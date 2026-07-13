# firmware: on-device inference and anomaly detection

Runs the quantized `id_02` autoencoder from [`ml/`](../ml/) on a real ESP32 using
TensorFlow Lite for Microcontrollers, organised as a FreeRTOS task pipeline, and
decides anomalies against a threshold while driving the onboard LED. Held-out
MIMII test vectors are streamed over the USB-serial cable, so no sensor is needed
and the on-device results can be checked against the host.

Status: built, flashed, and verified on real hardware (ESP-IDF v6.0.1, ESP32
WROOM). On-device AUC matches the host path, per-vector scores agree to within
about 0.001, and sustained multi-thousand-vector runs are crash-free with zero
checksum or anomaly-flag mismatches. Anomaly detection at the exported threshold
reaches F1 0.81 (precision 0.85, recall 0.76). Inference costs 49.2 ms per
vector, timed on the board at boot and printed as `MACHINESENSE_LATENCY`. See the
latency section in [`../README.md`](../README.md) for the real-time implication.

## Architecture

```text
replay_client.py (PC)                ESP32: 3 FreeRTOS tasks, 2 queues

load id_02 test WAVs                 rx_task     read 2560-byte float vector,
normalize each log-mel vector                    checksum it, push to queue
        |  2560 bytes ->             infer_task  quantize once, run Invoke,
        |                                        score, apply threshold,
        |                                        set LED, push to queue
        |  <- 12 bytes               tx_task     write score, anomaly, checksum
aggregate clip scores into AUC
plus precision/recall/F1
```

Protocol. Host to device: 640 little-endian `float32` values, 2560 bytes, one
normalized log-mel vector. Device to host: 12 bytes, made up of a `float32`
score, a `uint32` anomaly flag, and a `uint32` byte-sum checksum of the request
that was received.

Float32 is sent over the wire rather than int8 so the device quantizes once
internally and scores the reconstruction against the true input. That matches the
methodology used in `ml/` exactly, as described in `inference.h`.

The checksum earned its place: it caught the console UART silently translating
CR and LF bytes inside the binary stream, described in the notes below. It also
guards against transient bit errors.

## Setup

### 1. Generate the model and threshold

Both are git-ignored and produced from `ml/`.

```powershell
cd ..\ml
python export_per_id.py        # -> artifacts/per_id/id_02/model_data.cc/.h
python compute_threshold.py --machine-id id_02   # -> firmware/main/threshold.h
```

Copy the model in and rename its symbol to the generic name the firmware expects:

```powershell
cd ..\firmware
Copy-Item ..\ml\artifacts\per_id\id_02\model_data.cc main\model_data.cc
Copy-Item ..\ml\artifacts\per_id\id_02\model_data.h  main\model_data.h
(Get-Content main\model_data.cc) -replace 'g_model_data_id_02','g_model_data' | Set-Content main\model_data.cc
(Get-Content main\model_data.h)  -replace 'g_model_data_id_02','g_model_data' | Set-Content main\model_data.h
```

To target a different unit, such as `id_06` which is the strongest at AUC 0.9256,
swap `id_02` above and pass `--machine-id id_06` to both `compute_threshold.py`
and the replay client.

### 2. Install ESP-IDF

One-time setup with Espressif's
[Windows installer](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/windows-setup.html),
targeting esp32.

### 3. Build, flash, run

```powershell
cd firmware
idf.py set-target esp32          # first time only
idf.py -p COM7 build flash       # first build also fetches esp-tflite-micro

cd tools
pip install -r requirements.txt  # pyserial
python replay_client.py --port COM7 --machine-id id_02 --limit-files 20
```

Boot prints `MACHINESENSE_READY vector_len=640 threshold=0.594029` and then goes
silent, because logs are muted so they cannot corrupt the binary protocol. The
client reports on-device AUC, precision/recall/F1 at the threshold, and a
per-vector flag-integrity check. Full runs are slow at about 3.6 vectors per
second, so use `--limit-files` for spot checks.

With no board attached, `python replay_client.py --mock --machine-id id_02` runs
the same device math in Python. It validates the whole pipeline and reproduces
AUC 0.858 and F1 0.81.

## Notes from bring-up

All of these are fixed, but they are the non-obvious ones worth recording.

- UART CR/LF translation. The ESP-IDF console converts line endings on stdin and
  stdout by default, which silently corrupts any `0x0D` or `0x0A` byte in binary
  data. Disabled with `uart_vfs_dev_port_set_*_line_endings(..., ESP_LINE_ENDINGS_LF)`.
- Task Watchdog. The UART VFS read is non-blocking, so a naive `if (n <= 0) continue;`
  busy-spins and starves the idle task until the watchdog fires. Fixed with
  `vTaskDelay(1)` on an empty read.
- `Didn't find op` at build time means a kernel is missing from the resolver in
  `inference.cc`.
- `AllocateTensors() failed` means `kTensorArenaSize` is too small. It is
  currently 24 KB, and the boot log prints the real usage, which is about 15.8 KB.

## Files

| File | Role |
|---|---|
| `main/main.cc` | FreeRTOS pipeline (rx, infer, tx), threshold, LED, UART setup |
| `main/inference.h/.cc` | TFLite Micro interpreter, quantize-once and score |
| `main/model_data.h/.cc` | int8 model bytes, generated |
| `main/threshold.h` | anomaly threshold, generated by `compute_threshold.py` |
| `tools/replay_client.py` | PC driver: stream vectors, report AUC and anomaly metrics |
| `tools/diagnose_mismatch.py` | compare mock against real per-vector output during bring-up |
| `tools/tests/` | dataset-free, hardware-free tests |

## Possible extensions

On-device log-mel through `esp-dsp` with an I2S microphone would turn replay mode
into a fully live demo. The inference, threshold and LED path already in place
would be reused unchanged. See [`../docs/architecture.md`](../docs/architecture.md).
