#include "inference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "model_data.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace inference {
namespace {

static const char* TAG = "inference";

// The autoencoder is FullyConnected+ReLU dense layers (model.py's
// _dense_block); BatchNorm is folded into the FullyConnected weights by the
// TFLite converter, and the model's input/output tensors are already int8
// (export_tflite.py sets inference_input_type/output_type=int8), so no
// separate Quantize/Dequantize ops appear in the graph. If idf.py build fails
// with "Didn't find op", add the missing op here (see the TFLM op list at
// https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/kernels).
constexpr int kNumOps = 2;
tflite::MicroMutableOpResolver<kNumOps> resolver;

// Working memory for TFLM's tensors. A real run measured 15756 bytes used, so
// 24 KB gives comfortable headroom (weights live in flash, not here). Init()
// logs arena_used_bytes() on success; raise this if it ever reports too small.
constexpr int kTensorArenaSize = 24 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

int8_t QuantizeOne(float value, float scale, int zero_point) {
  int32_t q = static_cast<int32_t>(std::lround(value / scale)) + zero_point;
  return static_cast<int8_t>(std::clamp<int32_t>(q, -128, 127));
}

float Dequantize(int8_t value, float scale, int zero_point) {
  return (static_cast<int32_t>(value) - zero_point) * scale;
}

}  // namespace

bool Init() {
  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Model schema version %lu != supported %d",
             model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  if (resolver.AddFullyConnected() != kTfLiteOk) return false;
  if (resolver.AddRelu() != kTfLiteOk) return false;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors() failed -- increase kTensorArenaSize");
    return false;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  if (input_tensor->type != kTfLiteInt8 || output_tensor->type != kTfLiteInt8) {
    ESP_LOGE(TAG, "Expected an int8 in/out model (export_tflite.py sets this)");
    return false;
  }

  ESP_LOGI(TAG, "Model ready: input=%d floats, arena used=%u/%d bytes",
           static_cast<int>(input_tensor->bytes),
           static_cast<unsigned>(interpreter->arena_used_bytes()),
           kTensorArenaSize);
  return true;
}

int FloatInputSize() {
  // input_tensor->bytes counts int8 elements (1 byte each), which equals the
  // model's input element count -- i.e. how many float32s main.cc must read.
  return input_tensor ? static_cast<int>(input_tensor->bytes) : 0;
}

bool RunOnFloatVector(const float* input, int input_len, float* out_mse) {
  if (!interpreter || input_len != FloatInputSize()) {
    return false;
  }

  const float in_scale = input_tensor->params.scale;
  const int in_zero = input_tensor->params.zero_point;
  for (int i = 0; i < input_len; i++) {
    input_tensor->data.int8[i] = QuantizeOne(input[i], in_scale, in_zero);
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    ESP_LOGE(TAG, "Invoke() failed");
    return false;
  }

  const float out_scale = output_tensor->params.scale;
  const int out_zero = output_tensor->params.zero_point;

  double sum_sq_error = 0.0;
  for (int i = 0; i < input_len; i++) {
    const float out_val = Dequantize(output_tensor->data.int8[i], out_scale, out_zero);
    const float diff = input[i] - out_val;  // true float input, not re-dequantized
    sum_sq_error += static_cast<double>(diff) * diff;
  }
  *out_mse = static_cast<float>(sum_sq_error / input_len);
  return true;
}

bool BenchmarkLatency(int iterations, float* out_mean_us, float* out_min_us,
                      float* out_max_us) {
  if (!interpreter || iterations <= 0) return false;

  // Any fixed input works (latency is data-independent here); mid-scale zeros
  // keep it simple and deterministic.
  const int len = FloatInputSize();
  for (int i = 0; i < len; i++) input_tensor->data.int8[i] = 0;

  int64_t total_us = 0;
  int64_t min_us = INT64_MAX;
  int64_t max_us = 0;
  for (int i = 0; i < iterations; i++) {
    const int64_t start = esp_timer_get_time();
    if (interpreter->Invoke() != kTfLiteOk) {
      ESP_LOGE(TAG, "Invoke() failed during latency benchmark");
      return false;
    }
    const int64_t elapsed = esp_timer_get_time() - start;
    total_us += elapsed;
    if (elapsed < min_us) min_us = elapsed;
    if (elapsed > max_us) max_us = elapsed;
  }

  *out_mean_us = static_cast<float>(total_us) / iterations;
  *out_min_us = static_cast<float>(min_us);
  *out_max_us = static_cast<float>(max_us);
  return true;
}

}  // namespace inference
