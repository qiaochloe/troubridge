// Records all allocation sites with their stack traces.

#ifndef TCMALLOC_ALLOCATION_SITE_RECORDER_H_
#define TCMALLOC_ALLOCATION_SITE_RECORDER_H_

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/synchronization/mutex.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/logging.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

// Represents a single allocation site with its stack trace and statistics.
struct AllocationSite {
  StackTrace stack_trace;
  size_t total_count = 0;      // Total number of allocations at this site
  size_t total_bytes = 0;      // Total bytes allocated at this site
  size_t min_size = SIZE_MAX;  // Minimum allocation size
  size_t max_size = 0;         // Maximum allocation size
  void *latest_allocation = nullptr; // Latest allocation address, used for access frequency tracking
  size_t sampled_intervals = 0; // Number of intervals sampled on this site with an active latest allocation
  size_t sampled_accesses = 0; // Number of accesses to the latest allocation


  AllocationSite() = default;
  AllocationSite(const StackTrace& trace, size_t size)
      : stack_trace(trace), total_count(1), total_bytes(size), min_size(size), max_size(size), latest_allocation(nullptr), sampled_intervals(0), sampled_accesses(0) {}
  AllocationSite(const StackTrace& trace, size_t size, void* allocated_address)
      : stack_trace(trace), total_count(1), total_bytes(size), min_size(size), max_size(size), latest_allocation(allocated_address), sampled_intervals(0), sampled_accesses(0) {}
};

// Records all allocation sites with their stack traces.
// Thread-safe recorder that maintains statistics for each unique allocation site.
class AllocationSiteRecorder {
 public:
  AllocationSiteRecorder() = default;
  ~AllocationSiteRecorder() = default;

  // Record an allocation at the current call site.
  // Captures the stack trace and records statistics.
  void RecordAllocation(size_t size, void* allocated_address) ABSL_LOCKS_EXCLUDED(mutex_);

  // Get all recorded allocation sites.
  // Returns a copy of all allocation sites (for thread safety).
  std::vector<AllocationSite> GetAllocationSites() const ABSL_LOCKS_EXCLUDED(mutex_);

  // Get the number of unique allocation sites recorded.
  size_t GetSiteCount() const ABSL_LOCKS_EXCLUDED(mutex_);

  // Clear all recorded allocation sites.
  void Clear() ABSL_LOCKS_EXCLUDED(mutex_);

  // Check if recording is enabled.
  bool IsEnabled() const { return enabled_.load(std::memory_order_acquire); }

  // Enable or disable recording.
  void SetEnabled(bool enabled) { enabled_.store(enabled, std::memory_order_release); }

  // Print all recorded allocation sites to the printer.
  // Similar to MallocExtension::GetStats() format.
  void PrintStats(Printer& out) const ABSL_LOCKS_EXCLUDED(mutex_);

  // Print machine-learning friendly allocation sites to the printer.
  void PrintMachineLearningStats(Printer& out) const ABSL_LOCKS_EXCLUDED(mutex_);

  void RecordFree(void* freed_address);
  void PeriodicMemoryAccessTracking();

 private:
  // Hash function for stack traces.
  struct StackTraceHash {
    size_t operator()(const StackTrace& trace) const {
      size_t hash = 0;
      // Hash the stack trace depth and first few frames
      hash = absl::HashOf(trace.depth);
      for (size_t i = 0; i < trace.depth && i < 8; ++i) {
        hash = absl::HashOf(hash, trace.stack[i]);
      }
      return hash;
    }
  };

  // Equality function for stack traces.
  struct StackTraceEqual {
    bool operator()(const StackTrace& a, const StackTrace& b) const {
      if (a.depth != b.depth) return false;
      for (size_t i = 0; i < a.depth; ++i) {
        if (a.stack[i] != b.stack[i]) return false;
      }
      return true;
    }
  };

  mutable absl::Mutex mutex_;
  mutable absl::Mutex freed_allocations_mutex_;
  absl::flat_hash_map<StackTrace, AllocationSite, StackTraceHash, StackTraceEqual> sites_
      ABSL_GUARDED_BY(mutex_);
  std::unordered_set<void*> freed_allocations_ ABSL_GUARDED_BY(freed_allocations_mutex_);
  std::atomic<bool> enabled_{true};
  bool made_tracking_thread_ = false;
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_ALLOCATION_SITE_RECORDER_H_

