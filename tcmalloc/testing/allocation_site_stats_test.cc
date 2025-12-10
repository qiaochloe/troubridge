#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <string>
#include <vector>
#include <fstream>

#include "gtest/gtest.h"
#include "tcmalloc/malloc_extension.h"

namespace tcmalloc {
namespace {

// Helper function to make allocations from different call sites
void WriteMachineLearningStats(const std::string& machine_learning_stats) {
  // In Bazel tests, write outputs into TEST_UNDECLARED_OUTPUTS_DIR so they are
  // collected under bazel-testlogs; fall back to CWD when run manually.
  const char* output_dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  std::string output_path = output_dir
                                ? std::string(output_dir) +
                                      "/machine_learning_stats.txt"
                                : "machine_learning_stats.txt";
  std::ofstream machine_learning_file(output_path,
                                      std::ios::out | std::ios::app);
  machine_learning_file << machine_learning_stats << std::endl;
}

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

void AllocateFromFunction4(size_t size) {
  int* ptrs[10];
  for (int i = 0; i < 10; i++) {
    ptrs[i] = (int*)malloc(size);
  }
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 5; j++) {
      *(ptrs[j]) = i * 10 + j;
      printf("%d\n", *(ptrs[j]));
      usleep(100000);
    }
  }
  for (int i = 0; i < 10; i++) {
    free(ptrs[i]);
  }
}

TEST(AllocationSiteStatsTest, BasicUsage) {
  AllocateFromFunction1();
  AllocateFromFunction1();
  AllocateFromFunction2();
  AllocateFromFunction3(1000);
  AllocateFromFunction3(2000);
  AllocateFromFunction4(4097);

  // Get allocation site statistics
  std::string stats = MallocExtension::GetAllocationSiteStats();

  // Verify we got some output
  EXPECT_FALSE(stats.empty());

  // Print the stats
  printf("\n=== Allocation Site Statistics ===\n");
  printf("%s\n", stats.c_str());
  printf("===================================\n\n");

  // Verify the output contains expected information
  EXPECT_TRUE(stats.find("Allocation Site Recorder") != std::string::npos ||
              stats.find("allocation sites") != std::string::npos);

  // Print machine-learning friendly stats to a file
  std::string machine_learning_stats = MallocExtension::GetMachineLearningAllocationSiteStats();
  EXPECT_FALSE(machine_learning_stats.empty());
  WriteMachineLearningStats(machine_learning_stats);

  // Verify we have at least one allocation site recorded
  size_t site_count = MallocExtension::GetAllocationSiteCount();
  EXPECT_GT(site_count, 0u);
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
  EXPECT_FALSE(stats.empty());
  printf("\n=== Allocation Site Statistics ===\n");
  printf("%s\n", stats.c_str());
  printf("===================================\n\n");

  // Print machine-learning friendly stats to a file
  std::string machine_learning_stats = MallocExtension::GetMachineLearningAllocationSiteStats();
  EXPECT_FALSE(machine_learning_stats.empty());
  WriteMachineLearningStats(machine_learning_stats);
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
  printf("\n=== Allocation Site Statistics ===\n");
  printf("%s\n", stats.c_str());
  printf("===================================\n\n");

  // Print machine-learning friendly stats to a file
  std::string machine_learning_stats = MallocExtension::GetMachineLearningAllocationSiteStats();
  EXPECT_FALSE(machine_learning_stats.empty());
  WriteMachineLearningStats(machine_learning_stats);
}

TEST(AllocationSiteStatsTest, Deduplication) {
  constexpr int kNumAllocations = 20;
  constexpr size_t kAllocSize = 100;
  
  std::vector<void*> ptrs;
  ptrs.reserve(kNumAllocations);
  
  // Make multiple allocations from the exact same call site (same line)
  for (int i = 0; i < kNumAllocations; ++i) {
    ptrs.push_back(malloc(kAllocSize));
  }

  // Free all
  for (void* ptr : ptrs) {
    free(ptr);
  }

  // Get stats
  std::string stats = MallocExtension::GetAllocationSiteStats();
  EXPECT_FALSE(stats.empty());

  // Should have one site with count = kNumAllocations
  // Search for the allocation count in the output
  std::string count_str = "Total allocations: " + std::to_string(kNumAllocations);
  bool found_deduplicated = stats.find(count_str) != std::string::npos;
  
  // Also check that total bytes matches
  size_t expected_total_bytes = kNumAllocations * kAllocSize;
  std::string bytes_str = "Total bytes: " + std::to_string(expected_total_bytes);
  bool found_correct_bytes = stats.find(bytes_str) != std::string::npos;

  printf("\n=== Allocation Site Statistics ===\n");
  printf("Expected: 1 site with %d allocations, %zu total bytes\n", 
         kNumAllocations, expected_total_bytes);
  printf("%s\n", stats.c_str());
  printf("==========================\n\n");

  // Verify deduplication occurred
  EXPECT_TRUE(found_deduplicated || found_correct_bytes) 
      << "Expected to find site with " << kNumAllocations 
      << " allocations or " << expected_total_bytes << " total bytes";

  // Print machine-learning friendly stats to a file
  std::string machine_learning_stats = MallocExtension::GetMachineLearningAllocationSiteStats();
  EXPECT_FALSE(machine_learning_stats.empty());
  WriteMachineLearningStats(machine_learning_stats);
  
  // Verify site count is reasonable
  size_t site_count = MallocExtension::GetAllocationSiteCount();
  EXPECT_GE(site_count, 1u);
}

}  // namespace
}  // namespace tcmalloc

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}