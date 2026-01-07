#ifndef TCMALLOC_HOTNESS_PREDICTOR_H_
#define TCMALLOC_HOTNESS_PREDICTOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <sys/mman.h>

#include "absl/base/nullability.h"
#include "tcmalloc/internal/logging.h"

// Forward declaration
struct StackTrace;

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

// Hotness classes for access-frequency-aware allocation.
enum class HotnessClass : uint8_t {
  kCold = 0,
  kWarm = 1,
  kHot = 2,
  kCount = 3
};

// ML-based hotness predictor that uses a PyTorch model to predict
// allocation hotness based on stack trace and allocation size.
class HotnessPredictorML {
 public:
  HotnessPredictorML();
  ~HotnessPredictorML();

  // Initialize the predictor by loading the model and tokenizer.
  // Returns true on success, false on failure.
  // On failure, the predictor will fall back to returning kCold.
  bool Initialize();

  // Predict hotness class from stack trace and allocation size.
  // Returns kCold if model is not available or prediction fails.
  HotnessClass Predict(const StackTrace* stack_trace, size_t allocation_size);

  // Check if the predictor is initialized and ready to use.
  bool IsInitialized() const { return initialized_; }

 private:
  // Convert stack trace to string representation
  std::string StackTraceToString(const StackTrace* stack_trace);

  // Tokenize a stack trace string into token IDs
  std::vector<int64_t> Tokenize(const std::string& stack_trace_str);

  // Load tokenizer from JSON file
  bool LoadTokenizer(const std::string& tokenizer_path);

  bool initialized_;
  class Impl;
  
  // Custom deleter for Impl that uses munmap to avoid going through tcmalloc
  struct ImplMmapDeleter {
    void operator()(Impl* ptr) const;
  };
  
  std::unique_ptr<Impl, ImplMmapDeleter> impl_;
};

// Global instance (lazy-initialized)
HotnessPredictorML* GetHotnessPredictorML();

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_HOTNESS_PREDICTOR_H_