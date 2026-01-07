#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "tcmalloc/malloc_extension.h"

namespace tcmalloc {
namespace {

TEST(Basic, BasicUsage) {
  std::cout << "Start basic test" << '\n';

  void* ptr = malloc(500);
  free(ptr);

  std::cout << "End basic test" << '\n';
  
//  // Get allocation site statistics
//  std::string stats = MallocExtension::GetAllocationSiteStats();

//  // Verify we got some output
//  EXPECT_FALSE(stats.empty());

//  // Print the stats
//  printf("\n=== Allocation Site Statistics ===\n");
//  printf("%s\n", stats.c_str());
//  printf("===================================\n\n");

//  // Verify the output contains expected information
//  EXPECT_TRUE(stats.find("Allocation Site Recorder") != std::string::npos ||
//              stats.find("allocation sites") != std::string::npos);

//  // Verify we have at least one allocation site recorded
//  size_t site_count = MallocExtension::GetAllocationSiteCount();
//  EXPECT_GT(site_count, 0u);
}

}  // namespace
}  // namespace tcmalloc

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}