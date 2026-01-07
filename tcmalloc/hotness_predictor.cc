#include "tcmalloc/hotness_predictor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "absl/base/call_once.h"
#include "absl/debugging/symbolize.h"
#include "absl/strings/str_replace.h"
#include "absl/synchronization/mutex.h"
#include "tcmalloc/internal/logging.h"

#ifdef TCMALLOC_USE_ML_PREDICTOR
#include <torch/script.h>
#include <torch/torch.h>
#ifdef TCMALLOC_USE_TOKENIZERS_CPP
#include "tokenizers_cpp.h"
#endif
#endif

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

namespace {

// Helper function to check if file exists
// Uses stat() system call directly to avoid any allocations
bool FileExists(const std::string& path) {
  // Use access() instead of stat() - simpler and faster
  // F_OK tests for existence, R_OK would test for read permission
  return (access(path.c_str(), F_OK) == 0);
}

// Helper function to read file contents
std::vector<uint8_t> ReadFileBytes(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return {};
  }

  std::streamsize size = file.tellg();
  if (size <= 0) {
    return {};
  }
  
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    return {};
  }

  return buffer;
}

}  // namespace

#ifdef TCMALLOC_USE_ML_PREDICTOR

struct HotnessPredictorML::Impl {
#ifdef TCMALLOC_USE_TOKENIZERS_CPP
  std::unique_ptr<tokenizers::Tokenizer> tokenizer;
#endif
  // Use pointer to delay construction until model is loaded.
  // torch::jit::script::Module default constructor may allocate or initialize
  // global state, which could deadlock if called during allocator initialization.
  std::unique_ptr<torch::jit::script::Module> model;
  bool model_loaded;
  bool tokenizer_loaded;
  Impl() : model_loaded(false), tokenizer_loaded(false) {
  }
};

// Implementation of ImplMmapDeleter
void HotnessPredictorML::ImplMmapDeleter::operator()(HotnessPredictorML::Impl* ptr) const {
  if (ptr) {
    ptr->~Impl();
    size_t size = sizeof(HotnessPredictorML::Impl);
    munmap(ptr, size);
  }
}

HotnessPredictorML::HotnessPredictorML() : initialized_(false) {
  TC_LOG("[ML] HotnessPredictorML constructor called");
  TC_LOG("[ML] About to allocate Impl using mmap");
  
  // Use mmap to allocate Impl, bypassing tcmalloc to avoid reentrancy
  const size_t size = sizeof(Impl);
  void* raw_mem = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw_mem == MAP_FAILED) {
    TC_LOG("[ML] mmap failed for Impl allocation");
    impl_ = nullptr;  // Will be checked in Initialize()
    return;
  }
  
  TC_LOG("[ML] Using placement new to construct Impl");
  Impl* ptr = new (raw_mem) Impl();
  impl_ = std::unique_ptr<Impl, ImplMmapDeleter>(ptr);
  TC_LOG("[ML] Impl allocated successfully");
}

HotnessPredictorML::~HotnessPredictorML() {
  // impl_ will be automatically destroyed by unique_ptr with ImplMmapDeleter
}

bool HotnessPredictorML::Initialize() {
  TC_LOG("[ML] Initialize() called");
  if (initialized_) {
    TC_LOG("[ML] Already initialized, returning true");
    return true;
  }

  const char* model_path = "predictor/best_model.ts";
  const char* tokenizer_path = "predictor/tokenizer.json";

  TC_LOG("[ML] Checking file existence: %s", tokenizer_path);
  // Check if files exist before attempting to load
  bool tokenizer_exists = FileExists(tokenizer_path);
  TC_LOG("[ML] FileExists returned: %d", tokenizer_exists ? 1 : 0);
  if (!tokenizer_exists) {
    TC_LOG("[ML] Tokenizer file not found: %s - ML predictor disabled", tokenizer_path);
    return false;
  }

  TC_LOG("[ML] Checking file existence: %s", model_path);
  bool model_exists = FileExists(model_path);
  TC_LOG("[ML] FileExists returned: %d", model_exists ? 1 : 0);
  if (!model_exists) {
    TC_LOG("[ML] Model file not found: %s - ML predictor disabled", model_path);
    return false;
  }

#ifdef TCMALLOC_USE_TOKENIZERS_CPP
  // Load tokenizer using tokenizers-cpp
  TC_LOG("[ML] Reading tokenizer file");
  try {
    auto blob = ReadFileBytes(tokenizer_path);
    if (blob.empty()) {
      TC_LOG("[ML] Failed to read tokenizer file: %s", tokenizer_path);
      return false;
    }
    
    TC_LOG("[ML] Tokenizer file read, size: %zu bytes", blob.size());
    
    // Convert vector<uint8_t> to string
    std::string json_blob(reinterpret_cast<const char*>(blob.data()), blob.size());
    
    TC_LOG("[ML] Creating tokenizer from JSON blob");
    impl_->tokenizer = tokenizers::Tokenizer::FromBlobJSON(json_blob);
    if (!impl_->tokenizer) {
      TC_LOG("[ML] Failed to load tokenizer from %s", tokenizer_path);
      return false;
    }
    impl_->tokenizer_loaded = true;
    TC_LOG("[ML] Tokenizer loaded successfully");
  } catch (const std::exception& e) {
    TC_LOG("[ML] Exception loading tokenizer: %s", e.what());
    return false;
  }
#else
  TC_LOG("[ML] tokenizers-cpp not available - ML predictor disabled");
  return false;
#endif

  // Load model
  TC_LOG("[ML] Loading model from: %s", model_path);
  try {
    auto loaded_model = torch::jit::load(model_path);
    TC_LOG("[ML] Model loaded, setting to eval mode");
    loaded_model.eval();
    impl_->model = std::make_unique<torch::jit::script::Module>(std::move(loaded_model));
    impl_->model_loaded = true;
    TC_LOG("[ML] Model loaded successfully");
  } catch (const std::exception& e) {
    TC_LOG("[ML] Failed to load model from %s: %s", model_path, e.what());
    return false;
  }

  initialized_ = true;
  TC_LOG("[ML] ML Hotness Predictor initialized successfully");
  return true;
}

HotnessClass HotnessPredictorML::Predict(const StackTrace* stack_trace,
                                         size_t allocation_size) {
  if (!initialized_ || !impl_->model_loaded || !impl_->tokenizer_loaded ||
      !impl_->model) {
    return HotnessClass::kCold;
  }

  try {
    // Convert stack trace to string
    std::string stack_str = StackTraceToString(stack_trace);
    if (stack_str.empty()) {
      return HotnessClass::kCold;
    }

    // Tokenize using tokenizers-cpp
#ifdef TCMALLOC_USE_TOKENIZERS_CPP
    std::vector<int> token_ids_int = impl_->tokenizer->Encode(stack_str);
    // Convert to int64_t for PyTorch
    std::vector<int64_t> token_ids(token_ids_int.begin(), token_ids_int.end());
#else
    std::vector<int64_t> token_ids;
#endif

    if (token_ids.empty()) {
      return HotnessClass::kCold;
    }

    // Limit sequence length (model expects reasonable lengths)
    constexpr size_t kMaxSeqLen = 256;
    if (token_ids.size() > kMaxSeqLen) {
      token_ids.resize(kMaxSeqLen);
    }

    // Prepare inputs
    torch::NoGradGuard no_grad;

    // Create token tensor: [1, seq_len]
    torch::Tensor tokens_tensor =
        torch::from_blob(token_ids.data(),
                         {1, static_cast<int64_t>(token_ids.size())},
                         torch::kInt64)
            .clone();  // Clone to ensure ownership

    // Create size tensor: log1p(size) as [1, 1]
    double log_size = std::log1p(static_cast<double>(allocation_size));
    torch::Tensor size_tensor = torch::tensor({{log_size}}, torch::kFloat32);

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tokens_tensor);
    inputs.push_back(size_tensor);

    auto output = impl_->model->forward(inputs).toTensor();

    // Get predicted class (argmax)
    auto predicted = output.argmax(1).item<int64_t>();

    // Map to HotnessClass
    if (predicted == 0) {
      return HotnessClass::kCold;
    } else if (predicted == 1) {
      return HotnessClass::kWarm;
    } else {
      return HotnessClass::kHot;
    }
  } catch (const std::exception& e) {
    TC_LOG("ML prediction failed: %s", e.what());
    return HotnessClass::kCold;
  }
}

std::string HotnessPredictorML::StackTraceToString(
    const StackTrace* stack_trace) {
  if (stack_trace == nullptr || stack_trace->depth == 0) {
    return "";
  }

  std::ostringstream oss;
  for (size_t i = 0; i < stack_trace->depth && i < kMaxStackDepth; ++i) {
    void* pc = stack_trace->stack[i];
    char symbol_buf[1024];

    if (absl::Symbolize(pc, symbol_buf, sizeof(symbol_buf))) {
      std::string symbol(symbol_buf);
      // Remove file:line part if present
      size_t space_pos = symbol.find(' ');
      if (space_pos != std::string::npos) {
        symbol = symbol.substr(space_pos + 1);
      }
      // Remove template brackets and function parentheses for consistency
      absl::StrReplaceAll({{"<>", ""}, {"()", ""}}, &symbol);
      oss << symbol;
      if (i < stack_trace->depth - 1) {
        oss << "\n";
      }
    } else {
      // Fallback to address
      oss << "0x" << std::hex << reinterpret_cast<uintptr_t>(pc);
      if (i < stack_trace->depth - 1) {
        oss << "\n";
      }
    }
  }

  return oss.str();
}

#else  // TCMALLOC_USE_ML_PREDICTOR

// Stub implementation when ML predictor is disabled
struct HotnessPredictorML::Impl {};

HotnessPredictorML::HotnessPredictorML() : initialized_(false) {
  impl_ = std::make_unique<Impl>();
}

HotnessPredictorML::~HotnessPredictorML() = default;

bool HotnessPredictorML::Initialize() { return false; }

HotnessClass HotnessPredictorML::Predict(const StackTrace* stack_trace,
                                         size_t allocation_size) {
  (void)stack_trace;
  (void)allocation_size;
  return HotnessClass::kCold;
}

std::string HotnessPredictorML::StackTraceToString(
    const StackTrace* stack_trace) {
  (void)stack_trace;
  return "";
}

#endif  // TCMALLOC_USE_ML_PREDICTOR

namespace {

// Custom deleter for HotnessPredictorML that uses munmap instead of delete
struct MmapDeleter {
  void operator()(HotnessPredictorML* ptr) const {
    if (ptr) {
      ptr->~HotnessPredictorML();
      size_t size = sizeof(HotnessPredictorML);
      munmap(ptr, size);
    }
  }
};

// Global instance with thread-safe initialization
ABSL_CONST_INIT absl::once_flag init_flag;
std::unique_ptr<HotnessPredictorML, MmapDeleter> g_predictor;

// Atomic flag to track if initialization is in progress.
// This prevents reentrant calls during initialization that could cause
// deadlock when initialization operations (like model loading) trigger
// memory allocations.
ABSL_CONST_INIT static std::atomic<bool> initializing{false};

// Thread-local flag to detect same-thread reentrancy into call_once.
// If we're already initializing on this thread, we must not call call_once
// again as it will deadlock.
ABSL_CONST_INIT static thread_local bool in_init{false};

void InitPredictor() {
  TC_LOG("[ML] InitPredictor() called");
  
  // Set flag immediately to prevent reentrant calls during initialization.
  // This must be done before any allocations to prevent deadlock.
  bool expected = false;
  if (!initializing.compare_exchange_strong(expected, true)) {
    TC_LOG("[ML] InitPredictor() already in progress, skipping");
    return;
  }

  TC_LOG("[ML] Allocating HotnessPredictorML using mmap (bypassing tcmalloc)");
  // Use mmap directly to avoid going through tcmalloc, which would cause
  // reentrancy deadlock. We need enough space for HotnessPredictorML.
  const size_t size = sizeof(HotnessPredictorML);
  void* raw_mem = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw_mem == MAP_FAILED) {
    TC_LOG("[ML] mmap failed for HotnessPredictorML allocation");
    initializing.store(false);
    return;
  }
  
  TC_LOG("[ML] Using placement new to construct HotnessPredictorML");
  HotnessPredictorML* ptr = new (raw_mem) HotnessPredictorML();
  g_predictor = std::unique_ptr<HotnessPredictorML, MmapDeleter>(ptr);
  
  TC_LOG("[ML] Calling Initialize()");
  g_predictor->Initialize();
  
  TC_LOG("[ML] Initialization complete");
  initializing.store(false);
}

}  // namespace

HotnessPredictorML* GetHotnessPredictorML() {
  // If we're already initializing on THIS thread, return nullptr
  // immediately. Calling call_once reentrantly from the same thread will
  // deadlock because call_once uses a mutex internally.
  if (in_init) {
    TC_LOG("[ML] GetHotnessPredictorML() called during init (same thread), returning nullptr");
    return nullptr;
  }
  
  // If initialization is in progress on another thread, return nullptr
  // to avoid waiting (which could deadlock if that thread is waiting for us).
  if (initializing.load()) {
    TC_LOG("[ML] GetHotnessPredictorML() called during init (other thread), returning nullptr");
    return nullptr;
  }
  
  // Mark that we're entering call_once on this thread
  in_init = true;
  TC_LOG("[ML] GetHotnessPredictorML() calling call_once");
  
  absl::call_once(init_flag, InitPredictor);
  
  // Clear the thread-local flag
  in_init = false;
  
  TC_LOG("[ML] GetHotnessPredictorML() returning predictor");
  return g_predictor.get();
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
