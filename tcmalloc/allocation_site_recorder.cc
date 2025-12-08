#include "tcmalloc/allocation_site_recorder.h"

#include <algorithm>
#include <cstddef>
#include <vector>
#include <thread>

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

void AllocationSiteRecorder::PeriodicMemoryAccessTracking() {
  while (IsEnabled()) {
    {
      absl::MutexLock lockm(&mutex_);
      absl::MutexLock lock(&freed_allocations_mutex_);
      for (auto& [trace, site] : sites_) {
        if (site.latest_allocation != nullptr) {
          if (freed_allocations_.find(site.latest_allocation) != freed_allocations_.end()) {
            freed_allocations_.erase(site.latest_allocation);
            site.latest_allocation = nullptr;
          } else {
            site.sampled_accesses++;
            site.sampled_intervals++;
          }
        }
      }
    }
    absl::SleepFor(absl::Milliseconds(200));
  }
}

void AllocationSiteRecorder::RecordAllocation(size_t size, void* allocated_address) {
  if (!IsEnabled()) {
    return;
  }

  if (!made_tracking_thread_) {
    made_tracking_thread_ = true;
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
    } else {
      // Insert new site - this might allocate, but we've already set
      // recording_allocation to prevent reentrancy
      sites_.emplace(trace, AllocationSite(trace, size, allocated_address));
    }
  }

  recording_allocation = false;
}

void AllocationSiteRecorder::RecordFree(void* freed_address) {
  if (recording_free) {
    return;
  }
  recording_free = true;
  absl::MutexLock lock(&freed_allocations_mutex_);
  if (freed_allocations_.bucket_count() == 0) {
    recording_free = false;
    return;
  }
  freed_allocations_.insert(freed_address);
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

