# edgeAI-MachineSense architecture

## Pipeline

```text
Edge device (ESP32 DevKit)
  log-mel vectors
    -> TFLite Micro autoencoder
    -> reconstruction MSE
    -> compare against threshold
    -> anomaly flag + LED
    -> 12-byte reply (score, anomaly flag, checksum)

Vectors arrive over USB/UART from the host replay client. A live build would
replace that source with an I2S microphone and on-device log-mel via esp-dsp.
```

## Design decisions

- Unsupervised autoencoder rather than a classifier. It trains on normal sound
  only, so no labelled failures are needed, which matches how predictive
  maintenance data usually looks.
- Sensor-free by default through replay mode. The ESP32 runs the real quantized
  model on pre-processed MIMII vectors, so validation needs no extra hardware.
- One autoencoder per machine ID. A device ships only the model for the machine
  it monitors.
- Full int8 quantization, required by TFLite Micro. It measured lossless against
  the float model, with a per-ID macro AUC of 0.768 either way.
- FreeRTOS task pipeline (rx, infer, tx) with a byte-sum checksum on the binary
  UART protocol. The checksum caught the console's CR/LF translation during
  bring-up.

## What this project adds over library-desk-sense

| | library-desk-sense | edgeAI-MachineSense |
|---|---|---|
| Intelligence | server-side analytics | on-device |
| Signal processing | low-rate sensing | log-mel features |
| Concurrency | basic tasks | FreeRTOS rx/infer/tx pipeline |
| Quantization | none | int8 TFLite Micro, lossless vs float |
| Testing and CI | none | dataset-free tests + GitHub Actions |

## Results

| Metric | Value |
|---|---|
| Per-ID macro AUC (float32) | 0.768 |
| Per-ID macro AUC (int8) | 0.768 |
| Deployed `id_02` AUC (int8, host) | 0.858 |
| `id_02` anomaly F1 at threshold | 0.81 (precision 0.85, recall 0.76) |
| int8 model size (`id_02`) | about 311 KiB |
| Tensor-arena RAM used | 15,756 of 24,576 bytes |
| Inference latency (ESP32 at 240 MHz) | 49.2 ms per vector (100-iteration mean, about 52 us spread) |
| Real-time margin | about 309 vectors per 10 s clip, so about 15 s of compute |
| On-device vs host agreement | 60-clip matched run: 0.8933 on board vs 0.8944 host, identical confusion matrix, 0 flag mismatches |

Latency is measured on the board at boot rather than inferred from replay
throughput. At 115200 baud the 2560-byte request alone takes about 222 ms, so
UART transfer would otherwise hide the compute cost entirely.

Two separate points about that number. Measured: it is not compiler-related,
since `-Og` to `-O2` moved it only from 49.3 ms to 49.2 ms. Inferred but not
profiled: the remaining cost is likely structural, because this is a dense-only
model on a classic ESP32 with no SIMD, and `esp-nn` mainly accelerates
convolution rather than fully-connected layers.
