#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "tcmalloc/malloc_extension.h"

namespace tcmalloc {
namespace {

// Helper function to make allocations from different call sites
void AllocateFromFunction1() {
  void* ptr1 = malloc(100);
  void* ptr2 = malloc(200);
  free(ptr1);
  free(ptr2);
}

void AllocateFromFunction2() {
  void* ptr = malloc(500);
  free(ptr);
}

void AllocateFromFunction3(size_t size) {
  void* ptr = malloc(size);
  free(ptr);
}

TEST(AllocationSiteStatsTest, BasicUsage) {
  // Make some allocations from different call sites
  AllocateFromFunction1();
  AllocateFromFunction1();  // Same site, multiple times
  AllocateFromFunction2();
  AllocateFromFunction3(1000);
  AllocateFromFunction3(2000);  // Same function, different sizes

  // Get allocation site statistics
  std::string stats = MallocExtension::GetAllocationSiteStats();

  // Verify we got some output
  EXPECT_FALSE(stats.empty());

  // Print the stats (useful for debugging)
  printf("\n=== Allocation Site Statistics ===\n");
  printf("%s\n", stats.c_str());
  printf("===================================\n\n");

  // Verify the output contains expected information
  EXPECT_TRUE(stats.find("Allocation Site Recorder") != std::string::npos ||
              stats.find("allocation sites") != std::string::npos);

  // Verify we have at least one allocation site recorded
  // (we made allocations from at least 3 different functions)
  size_t site_count = MallocExtension::GetAllocationSiteCount();
  EXPECT_GT(site_count, 0u);
}

TEST(AllocationSiteStatsTest, IncludedInGetStats) {
  // Make some allocations
  void* ptr1 = malloc(100);
  void* ptr2 = malloc(200);
  free(ptr1);
  free(ptr2);

  // Get full stats (which should include allocation sites)
  std::string all_stats = MallocExtension::GetStats();

  // Verify we got output
  EXPECT_FALSE(all_stats.empty());

  // The full stats should contain allocation site information
  // (Note: allocation sites are printed in GetStats output)
  printf("\n=== Full Stats (includes allocation sites) ===\n");
  printf("%s\n", all_stats.c_str());
  printf("==============================================\n\n");
}

TEST(AllocationSiteStatsTest, MultipleAllocationsSameSite) {
  // Make many allocations from the same call site
  std::vector<void*> ptrs;
  for (int i = 0; i < 10; ++i) {
    ptrs.push_back(malloc(100));
  }

  // Free all
  for (void* ptr : ptrs) {
    free(ptr);
  }

  // Get stats
  std::string stats = MallocExtension::GetAllocationSiteStats();

  // Should show this site with count = 10
  EXPECT_FALSE(stats.empty());
  printf("\n=== Stats after 10 allocations from same site ===\n");
  printf("%s\n", stats.c_str());
  printf("================================================\n\n");
}

TEST(AllocationSiteStatsTest, DifferentSizes) {
  // Make allocations of different sizes from the same function
  AllocateFromFunction3(10);
  AllocateFromFunction3(100);
  AllocateFromFunction3(1000);
  AllocateFromFunction3(10000);

  // Get stats
  std::string stats = MallocExtension::GetAllocationSiteStats();

  // Should show min/max/average sizes
  EXPECT_FALSE(stats.empty());
  printf("\n=== Stats with different allocation sizes ===\n");
  printf("%s\n", stats.c_str());
  printf("=============================================\n\n");
}

}  // namespace
}  // namespace tcmalloc

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

