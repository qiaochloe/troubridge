#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "tcmalloc/malloc_extension.h"

namespace tcmalloc {
namespace {

// Measure single allocation time using absl::Time
TEST(Basic, MeasureAllocationTime) {
  std::cout << "\n=== Allocation Time Measurement ===\n";
  
  const size_t alloc_size = 1024;
  const int num_iterations = 1000;
  
  // With absl::Time
  absl::Time start = absl::Now();
  for (int i = 0; i < num_iterations; ++i) {
    void* ptr = malloc(alloc_size);
    free(ptr);
  }
  absl::Time end = absl::Now();
  absl::Duration elapsed = end - start;
  
  double avg_time_ns = absl::ToDoubleNanoseconds(elapsed) / num_iterations;
  std::cout << "Using absl::Time:\n";
  std::cout << "  Total time: " << absl::ToDoubleMilliseconds(elapsed)
            << " ms\n";
  std::cout << "  Average per allocation: " << avg_time_ns << " ns\n";
  std::cout << "  Average per allocation: " << avg_time_ns / 1000.0
            << " us\n";
  
  // With std::chrono
  auto chrono_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; ++i) {
    void* ptr = malloc(alloc_size);
    free(ptr);
  }
  auto chrono_end = std::chrono::high_resolution_clock::now();
  auto chrono_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      chrono_end - chrono_start);
  
  double avg_time_ns2 = chrono_elapsed.count() / static_cast<double>(num_iterations);
  std::cout << "\nUsing std::chrono:\n";
  std::cout << "  Total time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   chrono_elapsed).count()
            << " us\n";
  std::cout << "  Average per allocation: " << avg_time_ns2 << " ns\n";
  std::cout << "  Average per allocation: " << avg_time_ns2 / 1000.0
            << " us\n";
  
  std::cout << "===================================\n\n";
}

//// Measure allocation time for different sizes
//TEST(Basic, MeasureAllocationTimeBySize) {
//  std::cout << "\n=== Allocation Time by Size ===\n";
//  std::cout << "Size (bytes)\tTime (ns)\tTime (us)\n";
//  std::cout << "-----------------------------------\n";
  
//  const std::vector<size_t> sizes = {8, 16, 32, 64, 128, 256, 512, 1024,
//                                      4096, 8192, 16384, 65536, 1048576};
//  const int num_iterations = 10000;
  
//  for (size_t size : sizes) {
//    auto start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i < num_iterations; ++i) {
//      void* ptr = malloc(size);
//      free(ptr);
//    }
//    auto end = std::chrono::high_resolution_clock::now();
//    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
//        end - start);
    
//    double avg_ns = elapsed.count() / static_cast<double>(num_iterations);
//    std::cout << size << "\t\t" << avg_ns << "\t\t" << avg_ns / 1000.0
//              << "\n";
//  }
  
//  std::cout << "==================================\n\n";
//}

//// Measure allocation vs deallocation time separately
//TEST(Basic, MeasureAllocDeallocTimeSeparately) {
//  std::cout << "\n=== Allocation vs Deallocation Time ===\n";
  
//  const size_t alloc_size = 4096;
//  const int num_iterations = 10000;
//  std::vector<void*> ptrs;
//  ptrs.reserve(num_iterations);
  
//  // Measure allocation time
//  auto alloc_start = std::chrono::high_resolution_clock::now();
//  for (int i = 0; i < num_iterations; ++i) {
//    ptrs.push_back(malloc(alloc_size));
//  }
//  auto alloc_end = std::chrono::high_resolution_clock::now();
//  auto alloc_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
//      alloc_end - alloc_start);
  
//  // Measure deallocation time
//  auto dealloc_start = std::chrono::high_resolution_clock::now();
//  for (void* ptr : ptrs) {
//    free(ptr);
//  }
//  auto dealloc_end = std::chrono::high_resolution_clock::now();
//  auto dealloc_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
//      dealloc_end - dealloc_start);
  
//  double avg_alloc_ns = alloc_elapsed.count() / static_cast<double>(num_iterations);
//  double avg_dealloc_ns = dealloc_elapsed.count() / static_cast<double>(num_iterations);
  
//  std::cout << "Allocation time:\n";
//  std::cout << "  Total: " << alloc_elapsed.count() / 1000.0 << " us\n";
//  std::cout << "  Average: " << avg_alloc_ns << " ns (" << avg_alloc_ns / 1000.0
//            << " us)\n";
  
//  std::cout << "\nDeallocation time:\n";
//  std::cout << "  Total: " << dealloc_elapsed.count() / 1000.0 << " us\n";
//  std::cout << "  Average: " << avg_dealloc_ns << " ns (" << avg_dealloc_ns / 1000.0
//            << " us)\n";
  
//  std::cout << "\nTotal (alloc + dealloc):\n";
//  double total_avg_ns = avg_alloc_ns + avg_dealloc_ns;
//  std::cout << "  Average: " << total_avg_ns << " ns (" << total_avg_ns / 1000.0
//            << " us)\n";
  
//  std::cout << "==========================================\n\n";
//}

//}  // namespace
//}  // namespace tcmalloc

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}