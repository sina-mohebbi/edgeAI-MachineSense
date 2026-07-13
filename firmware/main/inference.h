// TFLite-Micro wrapper around the edgeAI-MachineSense autoencoder (see ml/model.py).
#pragma once

namespace inference {

// Builds the resolver + interpreter and allocates tensors. Call once at boot.
// Returns false (and logs why) if the model can't be loaded -- most commonly
// because the tensor arena in inference.cc is too small, or the resolver in
// inference.cc doesn't register every op the exported .tflite actually uses.
bool Init();

// Runs the autoencoder on one already-normalized float32 feature vector (640 =
// config.FEATURE_DIM in ml/, must match FloatInputSize()). Writes the
// reconstruction-error score to *out_mse.
//
// Quantizes `input` to int8 using the model's own input scale/zero_point
// (same math as ml/evaluate_tflite.py's Int8Autoencoder._quantize), runs
// inference, dequantizes the output, then scores dequantized-output against
// the ORIGINAL float `input` -- not a re-dequantized version of it. This
// matches ml/evaluate_per_id_tflite.py's methodology exactly (one
// quantization round-trip, on the output only), so on-device AUC should
// closely match the host int8 AUC. An earlier version instead compared
// dequantized-output to dequantized-input (quantizing twice, since the
// vector arrived over UART already int8), which added an extra noise term
// to both sides of the comparison and measurably degraded AUC.
bool RunOnFloatVector(const float* input, int input_len, float* out_mse);

// Number of float32 elements the model expects per call (should be
// FEATURE_DIM = 640).
int FloatInputSize();

// Times `iterations` back-to-back Invoke() calls and reports microseconds.
// Call once at boot, before the replay pipeline starts (it writes the input
// tensor). A synthetic input is representative here: the graph is a fixed-size
// dense int8 network with no data-dependent branching, so inference time does
// not vary with the input values. This isolates pure compute -- the replay
// throughput seen on the host is UART-bound (~222 ms/vector at 115200 baud),
// which would otherwise completely mask the real inference cost.
bool BenchmarkLatency(int iterations, float* out_mean_us, float* out_min_us,
                      float* out_max_us);

}  // namespace inference
