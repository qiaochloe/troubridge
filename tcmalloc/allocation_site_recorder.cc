#include "tcmalloc/allocation_site_recorder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <thread>
#include <fcntl.h>
#include <unistd.h>

#include "absl/debugging/symbolize.h"
#include "absl/debugging/stacktrace.h"
#include "absl/synchronization/mutex.h"
#include "tcmalloc/internal/logging.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

// Thread-local flag to prevent reentrant calls. If we're already recording
// an allocation, skip nested calls to avoid deadlock when the hash map
// needs to allocate memory for resizing.
ABSL_CONST_INIT static thread_local bool recording_allocation = false;

ABSL_CONST_INIT static thread_local bool recording_free = false;

// Static pointer to the recorder for the atexit handler to call Shutdown().
// This is set when the tracking thread is created.
static AllocationSiteRecorder* g_recorder_for_shutdown = nullptr;

// atexit handler to shut down the recorder before cleanup runs.
static void ShutdownRecorderAtExit() {
  if (g_recorder_for_shutdown != nullptr) {
    g_recorder_for_shutdown->Shutdown();
  }
}

// Pagemap entry bits (from /proc/self/pagemap)
constexpr uint64_t PM_PRESENT = 1ULL << 63;       // Page is present in RAM
constexpr uint64_t PM_PFN_MASK = (1ULL << 55) - 1; // Mask for page frame number (bits 0-54)

constexpr size_t PAGE_SIZE = 4096;  // Typical page size, could use sysconf(_SC_PAGESIZE)

// File descriptors for /proc and /sys interfaces - opened once and reused
// -1 indicates not yet opened, -2 indicates open failed
static int pagemap_fd = -1;
static int page_idle_fd = -1;

// Initialize file descriptors for /proc and /sys interfaces
static void InitProcFds() {
  if (pagemap_fd == -1) {
    pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
    if (pagemap_fd < 0) pagemap_fd = -2;
  }
  if (page_idle_fd == -1) {
    // Idle Page Tracking: /sys/kernel/mm/page_idle/bitmap
    // Each bit represents one page frame. Writing 1 marks page as idle,
    // reading returns 1 if page is still idle (not accessed), 0 if accessed.
    page_idle_fd = open("/sys/kernel/mm/page_idle/bitmap", O_RDWR);
    if (page_idle_fd < 0) page_idle_fd = -2;
  }
}

// Get page frame number (PFN) for a virtual address from /proc/self/pagemap
// Returns 0 if the page is not present or on error
static uint64_t GetPageFrameNumber(void* addr) {
  if (pagemap_fd < 0) return 0;
  
  uintptr_t vaddr = reinterpret_cast<uintptr_t>(addr);
  uint64_t page_index = vaddr / PAGE_SIZE;
  off_t offset = page_index * sizeof(uint64_t);
  
  uint64_t pagemap_entry = 0;
  if (pread(pagemap_fd, &pagemap_entry, sizeof(pagemap_entry), offset) != sizeof(pagemap_entry)) {
    return 0;
  }
  
  // Check if page is present in RAM
  if (!(pagemap_entry & PM_PRESENT)) {
    return 0;
  }
  
  // Extract page frame number (bits 0-54)
  return pagemap_entry & PM_PFN_MASK;
}

// Check if a page was accessed since it was marked idle using Idle Page Tracking.
// The page_idle bitmap has one bit per page frame:
// - Writing 1 to a bit marks the page as idle
// - Reading returns 1 if still idle (not accessed), 0 if the page was accessed
// Returns true if the page was accessed (idle bit is 0), false otherwise.
static bool IsPageAccessed(uint64_t pfn) {
  if (page_idle_fd < 0 || pfn == 0) return false;
  
  // The bitmap is organized as 64-bit words, each bit representing one PFN
  // Offset in bitmap file = (pfn / 64) * 8 bytes
  // Bit position within word = pfn % 64
  uint64_t word_index = pfn / 64;
  uint64_t bit_index = pfn % 64;
  off_t offset = word_index * sizeof(uint64_t);
  
  uint64_t idle_bits = 0;
  if (pread(page_idle_fd, &idle_bits, sizeof(idle_bits), offset) != sizeof(idle_bits)) {
    return false;
  }
  
  // If the idle bit is 0, the page was accessed; if 1, still idle
  bool is_idle = (idle_bits >> bit_index) & 1;
  return !is_idle;  // Return true if accessed (not idle)
}

// Mark a page as idle using Idle Page Tracking
static void MarkPageIdle(uint64_t pfn) {
  if (page_idle_fd < 0 || pfn == 0) return;
  
  uint64_t word_index = pfn / 64;
  uint64_t bit_index = pfn % 64;
  off_t offset = word_index * sizeof(uint64_t);
  
  // Set the bit for this PFN to mark it as idle
  uint64_t idle_bits = 1ULL << bit_index;
  pwrite(page_idle_fd, &idle_bits, sizeof(idle_bits), offset);
}

// Check if a memory position has been accessed since it was last marked as idle.
// Returns true if the page was accessed, false otherwise.
bool recently_accessed(void* allocation) {
  if (allocation == nullptr) {
    return false;
  }
  
  // Initialize file descriptors on first call
  InitProcFds();
  
  // Get page frame number from pagemap
  uint64_t pfn = GetPageFrameNumber(allocation);
  if (pfn == 0) {
    // Page not present or error reading pagemap
    return false;
  }
  
  // Check if the page was accessed (idle bit cleared)
  return IsPageAccessed(pfn);
}

// Mark the page associated with an allocation idle
static void MarkAllocationIdle(void* allocation) {
  if (allocation == nullptr) {
    return false;
  }
  
  InitProcFds();
  
  uint64_t pfn = GetPageFrameNumber(allocation);
  if (pfn == 0) {
    return false;
  }
  
  // Mark as idle for next interval
  MarkPageIdle(pfn);
  
  return;
}

// Check if a page was accessed since it's marked idle
// Returns true if the page was accessed since the last mark, false otherwise.
static bool CheckAllocationAccess(void* allocation) {
  if (allocation == nullptr) {
    return false;
  }
  
  InitProcFds();
  
  uint64_t pfn = GetPageFrameNumber(allocation);
  if (pfn == 0) {
    return false;
  }
  
  // Check if accessed
  bool was_accessed = IsPageAccessed(pfn);
  
  return was_accessed;
}
const int access_checking_millisecond_interval = 200;
const int millisecond_interval_between_clear_and_check = 1000;
void AllocationSiteRecorder::PeriodicMemoryAccessTracking() {
  while (IsEnabled()) {
    {
      recording_free = true;
      absl::MutexLock lockm(&mutex_);
      absl::MutexLock lock(&freed_allocations_mutex_);
      for (auto& [trace, site] : sites_) {
        if (site.latest_allocation != nullptr) {
          if (freed_allocations_.find(site.latest_allocation) != freed_allocations_.end()) {
            freed_allocations_.erase(site.latest_allocation);
            site.latest_allocation = nullptr;
          } else {
            // Mark pages idle
            MarkAllocationIdle(site.latest_allocation)
          }
        }
      }
      recording_free = false;
    }
    absl::SleepFor(absl::Milliseconds(millisecond_interval_between_clear_and_check));
    {
      recording_free = true;
      absl::MutexLock lockm(&mutex_);
      absl::MutexLock lock(&freed_allocations_mutex_);
      for (auto& [trace, site] : sites_) {
        if (site.latest_allocation != nullptr) {
          if (freed_allocations_.find(site.latest_allocation) != freed_allocations_.end()) {
            freed_allocations_.erase(site.latest_allocation);
            site.latest_allocation = nullptr;
          } else {
            // Check if page was accessed since cleared
            if (CheckAllocationAccess(site.latest_allocation)) {
              site.sampled_accesses++;
            }
            site.sampled_intervals++;
          }
        }
      }
      recording_free = false;
    }
    absl::SleepFor(absl::Milliseconds(access_checking_millisecond_interval))
  }
}

void AllocationSiteRecorder::RecordAllocation(size_t size, void* allocated_address) {
  if (!IsEnabled()) {
    return;
  }

  if (!made_tracking_thread_) {
    made_tracking_thread_ = true;
    // Register the atexit handler to shut down before cleanup runs.
    // This prevents RecordFree from inserting into destroyed data structures.
    g_recorder_for_shutdown = this;
    std::atexit(ShutdownRecorderAtExit);
    std::thread tracking_thread([this]() { PeriodicMemoryAccessTracking(); });
    tracking_thread.detach();
  }


  // Prevent reentrant calls - if we're already inside RecordAllocation(),
  // skip this call to avoid deadlock when hash map operations trigger
  // memory allocations.
  if (recording_allocation) {
    return;
  }

  recording_allocation = true;

  // Capture stack trace, skipping this function and the allocation function
  StackTrace trace = {};
  trace.depth = absl::GetStackTrace(trace.stack, kMaxStackDepth, 2);
  trace.requested_size = size;
  trace.allocated_size = size;

  {
    absl::MutexLock lock(&mutex_);
    auto it = sites_.find(trace);
    if (it != sites_.end()) {
      // Update existing site
      it->second.total_count++;
      it->second.total_bytes += size;
      it->second.min_size = std::min(it->second.min_size, size);
      it->second.max_size = std::max(it->second.max_size, size);
      it->second.latest_allocation = allocated_address;
    } else {
      // Insert new site - this might allocate, but we've already set
      // recording_allocation to prevent reentrancy
      sites_.emplace(trace, AllocationSite(trace, size, allocated_address));
    }
  }

  recording_allocation = false;
}

void AllocationSiteRecorder::RecordFree(void* freed_address) {
  // Check shutdown flag first - during program cleanup, data structures may be
  // destroyed or in an invalid state, so skip all operations.
  if (IsShuttingDown()) {
    return;
  }
  if (freed_address == nullptr) {
    return;
  }
  if (recording_free) {
    return;
  }
  recording_free = true;
  recording_allocation = true;
  absl::MutexLock lock(&freed_allocations_mutex_);
  // Initialize the set with some buckets on first use if needed
  if (freed_allocations_.bucket_count() == 0) {
    freed_allocations_.reserve(1024);
  }
  freed_allocations_.insert(freed_address);
  recording_allocation = false;
  recording_free = false;
}

std::vector<AllocationSite> AllocationSiteRecorder::GetAllocationSites() const {
  size_t size;
  {
    absl::MutexLock lock(&mutex_);
    size = sites_.size();
  }
  
  // Allocate vector with exact size outside the lock to avoid deadlock
  std::vector<AllocationSite> result(size);
  
  // Now copy data while holding the lock using indexing (no allocations)
  {
    absl::MutexLock lock(&mutex_);
    size_t idx = 0;
    for (const auto& [trace, site] : sites_) {
      if (idx < result.size()) {
        result[idx] = site;
        idx++;
      }
    }
    // Resize in case size changed (but this should be rare)
    if (idx < result.size()) {
      result.resize(idx);
    }
  }
  return result;
}

size_t AllocationSiteRecorder::GetSiteCount() const {
  absl::MutexLock lock(&mutex_);
  return sites_.size();
}

void AllocationSiteRecorder::Clear() {
  absl::MutexLock lock(&mutex_);
  sites_.clear();
}

void AllocationSiteRecorder::PrintStats(Printer& out) const {
  // Use GetAllocationSites() which safely copies data while holding the lock
  // and releases it before returning, preventing deadlock
  std::vector<AllocationSite> sites_copy = GetAllocationSites();
  
  if (!IsEnabled()) {
    out.printf("Allocation Site Recorder: DISABLED\n");
    return;
  }

  if (sites_copy.empty()) {
    out.printf("Allocation Site Recorder: No allocation sites recorded\n");
    return;
  }

  // Sort sites by total_bytes (descending) for better readability
  // Don't reserve - avoid allocations that could trigger RecordAllocation
  std::vector<const AllocationSite*> sorted_sites;
  for (const auto& site : sites_copy) {
    sorted_sites.push_back(&site);
  }
  std::sort(sorted_sites.begin(), sorted_sites.end(),
             [](const AllocationSite* a, const AllocationSite* b) {
               return a->total_bytes > b->total_bytes;
             });

  out.printf("\n------------------------------------------------\n");
  out.printf("Allocation Site Recorder Statistics\n");
  out.printf("Total unique allocation sites: %zu\n", sites_copy.size());
  out.printf("------------------------------------------------\n\n");

  constexpr double kMiB = 1024.0 * 1024.0;

  for (size_t idx = 0; idx < sorted_sites.size(); ++idx) {
    const AllocationSite* site = sorted_sites[idx];
    const StackTrace& trace = site->stack_trace;

    out.printf("Site %zu:\n", idx + 1);
    out.printf("  Total allocations: %zu\n", site->total_count);
    out.printf("  Total bytes: %zu (%.2f MiB)\n", site->total_bytes,
                site->total_bytes / kMiB);
    out.printf("  Min size: %zu bytes\n", site->min_size);
    out.printf("  Max size: %zu bytes\n", site->max_size);
    out.printf("  Average size: %.2f bytes\n",
                site->total_count > 0
                    ? static_cast<double>(site->total_bytes) / site->total_count
                    : 0.0);
    out.printf("  Stack trace (depth=%zu):\n", trace.depth);

    for (size_t i = 0; i < trace.depth; ++i) {
      void* pc = trace.stack[i];
      char symbol_buf[1024];
      if (absl::Symbolize(pc, symbol_buf, sizeof(symbol_buf))) {
        out.printf("    #%zu %p %s\n", i, pc, symbol_buf);
      } else {
        out.printf("    #%zu %p\n", i, pc);
      }
    }
    out.printf("\n");
  }

  out.printf("------------------------------------------------\n");
}

void AllocationSiteRecorder::PrintMachineLearningStats(Printer& out) const {
  // Use GetAllocationSites() which safely copies data while holding the lock
  // and releases it before returning, preventing deadlock
  std::vector<AllocationSite> sites_copy = GetAllocationSites();
  
  if (!IsEnabled()) {
    out.printf("Allocation Site Recorder: DISABLED\n");
    return;
  }

  if (sites_copy.empty()) {
    out.printf("Allocation Site Recorder: No allocation sites recorded\n");
    return;
  }

  // Sort sites by total_bytes (descending) for better readability
  // Don't reserve - avoid allocations that could trigger RecordAllocation
  std::vector<const AllocationSite*> sorted_sites;
  for (const auto& site : sites_copy) {
    sorted_sites.push_back(&site);
  }
  std::sort(sorted_sites.begin(), sorted_sites.end(),
             [](const AllocationSite* a, const AllocationSite* b) {
               return a->total_bytes > b->total_bytes;
             });

  for (size_t idx = 0; idx < sorted_sites.size(); ++idx) {
    const AllocationSite* site = sorted_sites[idx];
    const StackTrace& trace = site->stack_trace;

    int average_size = site->total_bytes / site->total_count;
    out.printf("%d>|<%d>|<%d>|<", average_size, site->sampled_intervals, site->sampled_accesses);
    for (size_t i = 0; i < trace.depth; ++i) {
      void* pc = trace.stack[i];
      char symbol_buf[1024];
      if (absl::Symbolize(pc, symbol_buf, sizeof(symbol_buf))) {
        out.printf("%s>|<", symbol_buf);
      } else {
        out.printf("CANTSYMBOLIZE>|<");
      }
    }
    out.printf("\n");
  }
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

