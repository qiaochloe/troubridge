#include "tcmalloc/allocation_site_recorder.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "absl/debugging/stacktrace.h"
#include "absl/synchronization/mutex.h"
#include "tcmalloc/internal/logging.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

void AllocationSiteRecorder::RecordAllocation(size_t size) {
  if (!IsEnabled()) {
    return;
  }

  // Capture stack trace, skipping this function and the allocation function
  StackTrace trace = {};
  trace.depth = absl::GetStackTrace(trace.stack, kMaxStackDepth, 2);
  trace.requested_size = size;
  trace.allocated_size = size;

  absl::MutexLock lock(&mutex_);
  auto it = sites_.find(trace);
  if (it != sites_.end()) {
    // Update existing site
    it->second.total_count++;
    it->second.total_bytes += size;
    it->second.min_size = std::min(it->second.min_size, size);
    it->second.max_size = std::max(it->second.max_size, size);
  } else {
    // Insert new site
    sites_.emplace(trace, AllocationSite(trace, size));
  }
}

std::vector<AllocationSite> AllocationSiteRecorder::GetAllocationSites() const {
  absl::MutexLock lock(&mutex_);
  std::vector<AllocationSite> result;
  result.reserve(sites_.size());
  for (const auto& [trace, site] : sites_) {
    result.push_back(site);
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

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

