# Image Compression Analysis — Sequential vs. OpenMP Parallel Processing

A C++ project that studies how to parallelize **the compression and decompression stages of one image** using OpenMP. The goal is to compare a single-threaded implementation with a multithreaded implementation while preserving the exact encoded output and reconstructed image.

> **Important scope note:** This project implements a custom JPEG-style codec for experimentation. Its `.bin` output uses the project's own `IMGC` bitstream format; it is **not a standards-compliant JPEG file** and should not be expected to open in ordinary image viewers. The input can be a JPG/PNG/BMP image, and the reconstructed output is saved as PNG.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [How the Project Works](#3-how-the-project-works)
4. [Sequential and Parallel Design](#4-sequential-and-parallel-design)
5. [Repository Structure](#5-repository-structure)
6. [Technology Stack](#6-technology-stack)
7. [System Requirements](#7-system-requirements)
8. [Setup on Windows with MSYS2 UCRT64](#8-setup-on-windows-with-msys2-ucrt64)
9. [Build Instructions](#9-build-instructions)
10. [Run a Single-Image Experiment](#10-run-a-single-image-experiment)
11. [Run the Batch Experiment](#11-run-the-batch-experiment)
12. [Run the Image-Size Scaling Experiment](#12-run-the-image-size-scaling-experiment)
13. [Verify Correctness](#13-verify-correctness)
14. [Generate Graphs](#14-generate-graphs)
15. [Output Files and CSV Columns](#15-output-files-and-csv-columns)
16. [Metrics and Formulas](#16-metrics-and-formulas)
17. [Understanding the Results](#17-understanding-the-results)
18. [Current Experimental Data](#18-current-experimental-data)
19. [Limitations and Interpretation](#19-limitations-and-interpretation)
20. [Troubleshooting](#20-troubleshooting)
21. [Recommended Project Improvements](#21-recommended-project-improvements)
22. [Reproducibility Checklist](#22-reproducibility-checklist)
23. [Academic and Technical Notes](#23-academic-and-technical-notes)

---

## 1. Project Overview

Image compression reduces the number of bytes needed to store an image. Compression algorithms usually contain multiple computational stages, and some stages can be performed independently while others have dependencies.

This project compares:

- **Sequential implementation:** each stage executes on one thread.
- **OpenMP implementation:** independent work inside the same image is distributed among threads.
- **Compression:** transforms image pixels into quantized coefficients and an entropy-coded bitstream.
- **Decompression:** reconstructs quantized coefficients, colour planes, and RGB pixels from that bitstream.
- **Correctness:** checks that parallel compression produces the same compressed bytes as the sequential baseline and that both reconstruct the same PNG.
- **Performance:** records per-stage timings, total compression/decompression time, speedup, and quality/storage metrics.

The parallel implementation does **not** achieve parallelism merely by giving a different image to each thread. The loops parallelize work within a single image, including image rows, 8×8 blocks, and fixed-size groups of blocks.

## 2. Objectives

1. Implement a sequential image compression and decompression pipeline.
2. Identify independent computations inside each pipeline stage.
3. use OpenMP to parallelize independent rows, blocks, and entropy-coding chunks.
4. Preserve deterministic output so the sequential and parallel compressed bitstreams can be compared byte for byte.
5. Measure compression and decompression times at different thread counts.
6. Calculate compression ratio, storage saving, MSE, PSNR, and SSIM-like image quality.
7. Evaluate scaling with image dimensions and with a collection of input images.
8. Identify serial bottlenecks, parallel overhead, and opportunities for further optimization.

## 3. How the Project Works

### 3.1 High-level workflow

```text
Input image (JPG / PNG / BMP)
          |
          v
      Read RGB pixels
          |
          v
   RGB -> YCbCr conversion
          |
          v
   Pad dimensions to multiples of 8
          |
          v
    Divide into 8 x 8 blocks
          |
          v
     2-D Discrete Cosine Transform
          |
          v
        Quantization
          |
          v
   Count symbols / build histograms
          |
          v
       Build Huffman tables
          |
          v
   Huffman encode block chunks
          |
          v
  Join chunks + write custom IMGC header
          |
          v
       Compressed .bin file
          |
          v
      Parse header / tables
          |
          v
       Huffman decode chunks
          |
          v
     Dequantization + inverse DCT
          |
          v
       YCbCr -> RGB conversion
          |
          v
     Reconstructed PNG + metrics
```

### 3.2 Codec stages

The shared implementation is in `src/codec_common.h`, so the sequential and parallel programs use the same transform, quantization, entropy-coding, decoder, and metric logic. Their main difference is how independent loops are executed.

1. **RGB to YCbCr:** converts each input pixel into luminance (Y) and chrominance (Cb and Cr) values. The current implementation uses 4:4:4 colour planes (no chroma subsampling).
2. **Padding and block geometry:** image dimensions are padded to multiples of eight for 8×8 transform blocks. The output reconstruction uses the original width and height.
3. **DCT:** transforms each 8×8 block from spatial pixel values to frequency coefficients.
4. **Quantization:** divides transform coefficients by quantization-table values and rounds them. The quality setting is currently fixed at `QUALITY = 75` in `codec_common.h`.
5. **Symbol statistics:** scans quantized coefficients to count symbols used by entropy coding.
6. **Huffman tables:** constructs tables from the collected symbol counts.
7. **Huffman encoding:** encodes the quantized coefficient data into a bitstream.
8. **Header and chunk metadata:** stores dimensions, quality, coding tables, and chunk-start metadata needed by this project's decoder.
9. **Decoding and reconstruction:** reverses the entropy coding, quantization, DCT, and colour conversion.
10. **Metrics:** compares the decoded RGB pixels against the input pixels.

### 3.3 Why use a custom codec?

Using a library that performs the entire JPEG encode/decode operation would make it difficult to time and parallelize the internal stages independently. The custom codec makes the main computational stages visible for this experiment.

This is an educational JPEG-style implementation, not a replacement for a mature JPEG implementation. Do not describe the output as a normal JPEG bitstream.

## 4. Sequential and Parallel Design

### 4.1 Compression-stage parallelism

| Stage | Sequential implementation | Parallel implementation | Dependency / reason |
|---|---|---|---|
| RGB → YCbCr | Processes rows in a loop | OpenMP parallel loop over rows | Each output row is independent |
| DCT + quantization | Processes each 8×8 block in a loop | OpenMP parallel loop over blocks | Each block writes to its own coefficient positions |
| Symbol histogram | Uses one histogram | Each thread has a private histogram; thread histograms are combined | Local counts avoid unsynchronized updates to shared counters |
| Build Huffman tables | Builds the tables after all counts are available | Remains serial | Requires the final global histogram; work is relatively small |
| Huffman encoding | Encodes all chunks in sequence | Encodes independent chunks into separate bit buffers | Each chunk has its own output buffer and known DC predictor state |
| Merge and header | Joins encoded data and writes metadata | Remains serial | Chunks must be concatenated in order and offsets calculated |

### 4.2 Decompression-stage parallelism

| Stage | Sequential implementation | Parallel implementation | Dependency / reason |
|---|---|---|---|
| Parse header | Parses header and prepares decoding tables | Remains serial | Header information must be read before decoding |
| Huffman decoding | Decodes chunks one after another | OpenMP parallel loop over chunks | The custom file stores chunk offsets and the preceding DC predictor values |
| Dequantization + inverse DCT | Processes each block in a loop | OpenMP parallel loop over blocks | Blocks write to independent regions of the output planes |
| YCbCr → RGB | Processes image rows in a loop | OpenMP parallel loop over rows | Each row is independent |

### 4.3 Why chunk metadata is important

Entropy-coded data is a variable-length bitstream. Without additional information, a decoder may need to decode earlier symbols to determine where a later region begins. This implementation groups blocks into chunks of **512 blocks** and stores chunk-start metadata, including a bit offset and DC predictor information. This enables the parallel decoder to start decoding a chunk independently.

The chunk size is a compile-time constant, `CHUNK_BLOCKS = 512`, in `codec_common.h`. It is not currently a user-configurable 1/4/16/64 partition-count parameter.

### 4.4 Thread safety

The parallel implementation uses separate output regions or buffers for independent work. For symbol statistics, each thread updates a local histogram before the results are combined. For encoding, each chunk writes to its own `BitWriter`. The merge occurs afterward in a deterministic order.

The implementation also uses synchronization for shared status/histogram updates. These choices are important because unsynchronized concurrent writes can cause data races or nondeterministic output.

## 5. Repository Structure

```text
Image_Compression_Analysis-main/
├── README.md
├── image_dataset/
│   ├── 01_landscape.jpg
│   ├── 02_geometric_art.jpg
│   ├── 03_noise_texture.jpg
│   ├── 04_checkerboard.jpg
│   ├── 05_forest_texture.jpg
│   ├── 06_water_ripples.jpg
│   ├── 07_brick_wall.jpg
│   ├── 08_color_waves.jpg
│   ├── 09_dot_pattern.jpg
│   └── 10_mixed_texture.jpg
├── scaling_images/
│   ├── img_500x333.jpg
│   ├── img_1000x667.jpg
│   ├── img_2000x1333.jpg
│   └── img_3000x2000.jpg
├── results/
│   ├── single_results.csv
│   ├── batch_results.csv
│   ├── scaling_size.csv
│   └── scaling_collection.csv
├── graphs/
├── output/
│   └── single/
└── src/
    ├── codec_common.h
    ├── sequential_single.cpp
    ├── parallel_single.cpp
    ├── run_batch.py
    ├── run_scaling.py
    ├── verify_single.py
    ├── generate_graphs.py
    ├── stb_image.h
    ├── stb_image_write.h
    └── README_single.md
```

### Main files

| File | Responsibility |
|---|---|
| `src/codec_common.h` | Shared codec implementation, image data structures, DCT/quantization, Huffman code, decoder, metrics, output and CSV writing |
| `src/sequential_single.cpp` | Single-threaded compression and decompression |
| `src/parallel_single.cpp` | OpenMP parallel compression and decompression within one image |
| `src/run_batch.py` | Runs the sequential and parallel programs across images and records comparison/correctness rows |
| `src/run_scaling.py` | Generates image-size scaling measurements and derives collection-size totals from batch results |
| `src/verify_single.py` | Compares compressed outputs and reconstructed PNG files, then prints stage speedups and efficiency |
| `src/generate_graphs.py` | Generates timing, speedup, compression-ratio, PSNR, and SSIM plots |
| `src/stb_image.h` / `src/stb_image_write.h` | Header-only image reading and PNG writing dependencies |
| `results/*.csv` | Saved experimental output; values depend on the run and machine |
| `graphs/*.png` | Generated charts |

## 6. Technology Stack

- **C++17:** codec implementation.
- **OpenMP:** shared-memory parallel execution.
- **g++ / MinGW-w64:** compiler used for the tested MSYS2 setup.
- **stb_image / stb_image_write:** included header-only image input and PNG output.
- **Python 3:** experiment automation, validation, and plotting scripts.
- **Pillow:** image resizing for the image-size experiment.
- **pandas and Matplotlib:** graph generation.

The supplied implementation does not require OpenCV or libjpeg-turbo to compile; the image I/O used by the code comes from the included `stb` headers.

## 7. System Requirements

Recommended/tested project configuration:

- Windows 11 with MSYS2 UCRT64 terminal.
- A recent `g++` supporting C++17 and OpenMP.
- Python 3.
- Python packages for scaling and graph generation:
  - `Pillow`
  - `pandas`
  - `matplotlib`

Large images require enough RAM for RGB pixels, three padded colour planes, transform coefficients, decoder buffers, and reconstructed pixels. The project can be run on smaller images, but thread overhead may outweigh the benefits for small workloads.

Check the environment:

```bash
g++ --version
python --version
python -m pip --version
```

Check that OpenMP is available by compiling the parallel program with `-fopenmp`. If compilation fails on `omp.h` or OpenMP symbols, check the compiler/toolchain installation.

## 8. Setup on Windows with MSYS2 UCRT64

### Step 1 — Open the correct terminal

Open **MSYS2 UCRT64** (not the plain MSYS terminal). Confirm that the compiler is available:

```bash
g++ --version
```

If the compiler is missing, install the UCRT64 toolchain from the MSYS2 UCRT64 terminal:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-gcc
```

### Step 2 — Extract the repository

Extract the ZIP so the project folder contains `image_dataset/` and `src/`. Open the UCRT64 terminal and change to the extracted `src` directory. For example, if the folder is in Downloads:

```bash
cd "/c/Users/<YOUR_WINDOWS_USERNAME>/Downloads/Image_Compression_Analysis-main/src"
```

Replace the example path with the actual path on your computer.

**Run the programs and scripts from `src/`.** The C++ programs write compressed files and reconstructed images to `../output/single/` and CSV data to `../results/`; the experiment scripts read and write CSV files under `../results/` and charts under `../graphs/`.

### Step 3 — Install Python packages

```bash
python -m pip install --user Pillow pandas matplotlib
```

The C++ codec itself uses the included `stb` headers. Python packages are needed for the scaling and graph scripts, not for compiling the two C++ executables.

## 9. Build Instructions

From the `src/` directory, compile the sequential program:

```bash
g++ sequential_single.cpp -o sequential_single.exe -std=c++17 -O2 -ffp-contract=off
```

Compile the OpenMP program:

```bash
g++ parallel_single.cpp -o parallel_single.exe -std=c++17 -O2 -fopenmp -ffp-contract=off
```

### Why these flags?

- `-std=c++17`: enables the language features used by the project.
- `-O2`: enables compiler optimizations suitable for benchmarking.
- `-fopenmp`: enables OpenMP for the parallel version.
- `-ffp-contract=off`: helps keep floating-point evaluation consistent between the two executables so exact-output comparisons are meaningful.

Use the **same optimization and floating-point flags** for both programs. Otherwise, a comparison may mix parallelization effects with compiler-configuration differences.

Successful compilation should create `sequential_single.exe` and `parallel_single.exe` in `src/`.

## 10. Run a Single-Image Experiment

Choose one image, for example `01_landscape.jpg`, and run from `src/`.

### Sequential baseline

```bash
./sequential_single.exe ../image_dataset/01_landscape.jpg 10
```

### Parallel versions

```bash
./parallel_single.exe ../image_dataset/01_landscape.jpg 1 10
./parallel_single.exe ../image_dataset/01_landscape.jpg 2 10
./parallel_single.exe ../image_dataset/01_landscape.jpg 4 10
./parallel_single.exe ../image_dataset/01_landscape.jpg 8 10
./parallel_single.exe ../image_dataset/01_landscape.jpg 12 10
```

The final argument is the number of repetitions. Each program reports the mean and best compression time and the mean and best decompression time. Input reading and quality-metric calculation are outside the main timed compression/decompression stages.

The parallel program's second argument is the requested OpenMP thread count. `1` thread is a useful control for comparing the parallel code path against the sequential baseline, but it is not a replacement for the separate sequential executable.

### Outputs

The program creates or updates:

- `../output/single/sequential.bin`
- `../output/single/sequential_reconstructed.png`
- `../output/single/parallel_t1.bin`
- `../output/single/parallel_t1_reconstructed.png`
- Equivalent `parallel_tN` files for each tested thread count.
- `../results/single_results.csv`

Paths above are relative to `src/`, where the programs are run. The output folder itself is at the project root, alongside `results/` and `graphs/`. Each new run overwrites the named `.bin` and `.png` outputs and **appends** a row to `../results/single_results.csv`. Keep that behavior in mind when comparing historical rows.

## 11. Run the Batch Experiment

The batch runner executes the sequential program and each requested parallel thread count for every supported image in the chosen folder. It compares each parallel `.bin` and reconstructed PNG against the sequential baseline for the same image.

Example: run ten images, thread counts 1/2/4/8/12, and ten repetitions per configuration:

```bash
python run_batch.py ../image_dataset 1,2,4,8,12 10
```

General form:

```bash
python run_batch.py <image_folder> <comma_separated_thread_counts> <repetitions>
```

If arguments are omitted, the script defaults to `image_dataset`, thread counts `1,2,4,8`, and five repetitions. Because the supplied dataset is located one directory above `src/`, use `../image_dataset` when running from `src/`.

The script writes:

```text
../results/batch_results.csv
```

It also prints a compact summary including dimensions, compression ratio, PSNR, SSIM, and compression/decompression speedups.

**Important:** `single_results.csv` is append-only, and the batch runner reads the last row after each run. If you are starting a fresh benchmark, archive or back up the existing CSV and then clear it before running the new experiment. Do not delete results you still need for the report.

## 12. Run the Image-Size Scaling Experiment

`run_scaling.py` resizes `01_landscape.jpg` to four dimensions and runs the sequential program and parallel configurations at 4, 8, and 12 threads.

From `src/`, on the Windows setup for which the script is currently written:

```bash
python run_scaling.py
```

The script expects `sequential_single.exe` and `parallel_single.exe` to exist in `src/`. It writes:

- `../scaling_images/` — resized image inputs.
- `../results/scaling_size.csv` — per-size timing rows.
- `../results/scaling_collection.csv` — sums derived from the existing batch results.

### Important interpretation of collection-size scaling

The collection-size script calculates totals by summing individual per-image measurements already stored in `batch_results.csv`. It is a **derived estimate for collections of 1, 2, 5, or 10 images**, not a new experiment in which multiple images are processed concurrently as one batch workload. Explain that distinction in reports and presentations.

The script currently uses fixed `.exe` paths and should be updated before use on Linux/macOS. The image-size CSV has no explicit repetition-count column even though the script configures five repetitions for each run; document the repetition configuration separately or improve the script to write it into the CSV.

## 13. Verify Correctness

After running the sequential baseline and the desired parallel configurations for the **same image**, run:

```bash
python verify_single.py
```

The verifier compares the sequential compressed bitstream with each available `parallel_tN.bin` file and compares reconstructed PNG file bytes. It also prints per-stage speedup and total efficiency based on the most recent matching rows in `single_results.csv`.

A successful comparison reports that the compressed file and reconstructed image are identical for each tested thread count.

### What the checks establish

- **Byte-identical `.bin`:** the parallel run emitted exactly the same encoded bytes as the sequential baseline for that input/configuration.
- **Byte-identical reconstructed PNG:** both runs saved exactly the same PNG bytes in the current environment.
- **Decoded coefficients identical to encoder:** the decoder reconstructed the same quantized coefficient arrays that were supplied to the encoder, as checked by the program.

### What the checks do not establish

- They do not prove that the custom format is compatible with standard JPEG software.
- They do not by themselves prove that the image is visually indistinguishable from the original; lossy compression is expected to introduce pixel differences.
- Byte-identical PNG comparisons can depend on using the same writer/library/version and settings.
- The verifier uses existing output files and the latest CSV rows; stale files or mixed runs can lead to misleading results. Run a clean, controlled experiment when preparing final evidence.

## 14. Generate Graphs

First ensure `batch_results.csv` contains the benchmark run you want to present. Then run:

```bash
python generate_graphs.py
```

The script reads `../results/batch_results.csv` and writes PNG charts under `../graphs/`:

- `compression_time.png`
- `decompression_time.png`
- `compression_speedup.png`
- `decompression_speedup.png`
- `compression_ratio.png`
- `psnr.png`
- `ssim.png`


### Graph-generation note

The current graph script uses `01_landscape.jpg` for timing/speedup charts and plots the thread rows present in the CSV. Make sure those rows come from one consistent experiment. The quality charts use the sequential rows across images. The charts are useful for exploration, but for a formal report it is better to calculate average results across images and include variability (for example, standard deviation or error bars).

## 15. Output Files and CSV Columns

### `single_results.csv`

One row is appended per invocation. It contains:

| Column | Meaning |
|---|---|
| `Mode` | `sequential` or `parallel` |
| `Threads` | Configured thread count; sequential is recorded as 1 |
| `Image` | Input image path |
| `Width`, `Height` | Original input dimensions |
| `Reps` | Number of timed repetitions configured |
| `Read_ms` | Image loading time; reported separately and excluded from compression/decompression totals |
| `Color_ms` | RGB → YCbCr stage mean |
| `DCT_Quant_ms` | DCT + quantization stage mean |
| `Hist_ms` | Symbol-statistics stage mean |
| `Huffman_ms` | Huffman-table construction stage mean |
| `Encode_ms` | Huffman-encoding stage mean |
| `Merge_ms` | Chunk merge + header creation stage mean |
| `Total_mean_ms` | Mean total compression-stage time |
| `Total_best_ms` | Best total compression time among repetitions |
| `Compressed_bytes` | Size of the custom compressed `.bin` file |
| `Ratio` | Raw RGB bytes divided by compressed bytes |
| `MSE` | Mean squared RGB pixel error between input and reconstruction |
| `PSNR_dB` | Peak signal-to-noise ratio derived from MSE |
| `SSIM` | Current implementation's global grayscale SSIM-like score |
| `Checksum` | FNV-1a checksum of the custom compressed bytes |
| `Original_bytes` | Size of the original input file on disk |
| `Parse_ms` | Header parsing / decoder-table preparation mean |
| `Decode_ms` | Huffman decoding mean |
| `IDCT_ms` | Dequantization + inverse DCT mean |
| `ColorBack_ms` | YCbCr → RGB reconstruction mean |
| `Decomp_mean_ms` | Mean total decompression-stage time |
| `Decomp_best_ms` | Best total decompression time among repetitions |

The `Ratio` field is **raw RGB size / compressed size**. The code also prints the original input-file size and original-file/compressed-file ratio to the console, but the CSV does not currently have a separate column for that second ratio.

### `batch_results.csv`

Contains per-image sequential/parallel rows, a `Correct` field, and `Comp_speedup` / `Decomp_speedup` fields relative to the sequential run for that image. A sequential row is labelled `baseline`; each parallel row should be `PASS` or `FAIL`.

### `scaling_size.csv`

Contains mode, thread count, width, height, compression time, and decompression time. The repetition count is not stored in the current schema.

### `scaling_collection.csv`

Contains totals derived by summing selected image rows from `batch_results.csv`. These totals are not a separate concurrent collection-processing measurement.

## 16. Metrics and Formulas

### Compression ratio

The CSV `Ratio` is:

\[
\text{Compression Ratio} =
\frac{\text{Raw RGB bytes}}{\text{Compressed file bytes}}
\]

The raw RGB byte count is width × height × 3 for an 8-bit, three-channel image. A larger value means the custom file is smaller relative to the uncompressed RGB representation.

The original JPG/PNG file is already encoded, so comparing its size with the custom output is a different measure. Do not mix these two denominators.

### Percentage of storage saved

The program prints raw-RGB storage saving:

\[
\text{Space Saving (\%)} =
100\left(1-\frac{\text{Compressed bytes}}{\text{Raw RGB bytes}}\right)
\]

This is relative to raw RGB data, not necessarily relative to the original JPG input file.

### Compression and decompression time

- **Compression time:** measured from the start of the compression stages to completion of the custom in-memory compressed representation. It includes the stage work and the merge/header construction, but excludes image loading, output-file writing in `finish_run`, and quality-metric calculations.
- **Decompression time:** measured from the start of parsing the in-memory compressed representation to completion of reconstructed RGB pixels. It excludes PNG output writing and quality-metric calculations.
- **Read time:** measured separately and printed/logged as `Read_ms`.
- **Mean time:** sum of the measured stage/total times divided by the number of repetitions.
- **Best time:** minimum total time observed across repetitions.

The current code does not record standard deviation or confidence intervals. Report mean and best values accurately; do not claim variability was measured unless you calculate it.

### Mean squared error (MSE)

For original pixel values \(I_i\) and reconstructed pixel values \(R_i\), across all RGB channel values:

\[
\mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N}(I_i-R_i)^2
\]

Here, \(N\) is the number of scalar RGB channel values, not simply the number of pixels.

### Peak signal-to-noise ratio (PSNR)

For 8-bit pixels with maximum value 255:

\[
\mathrm{PSNR} = 10\log_{10}\left(\frac{255^2}{\mathrm{MSE}}\right)
\]

PSNR is reported in decibels (dB). If MSE is zero, the code returns infinity.

### SSIM caveat

The current `calc_ssim` function computes a **global grayscale SSIM-like statistic** using whole-image means, variances, and covariance. Standard SSIM is commonly calculated over local windows and then aggregated. Therefore, describe the current metric precisely as the implementation's global grayscale SSIM score, or replace it with a well-tested standard SSIM implementation before claiming standard SSIM results.

### Speedup

For the same input image and same measured scope:

\[
\mathrm{Speedup}(p) =
\frac{T_{\mathrm{sequential}}}{T_{\mathrm{parallel},p}}
\]

A speedup greater than 1 means the parallel configuration was faster for that metric. A speedup below 1 means it was slower.

### Parallel efficiency

\[
\mathrm{Efficiency}(p) =
\frac{\mathrm{Speedup}(p)}{p}\times 100\%
\]

where \(p\) is the configured thread count. This expresses how effectively the additional threads are used relative to ideal linear scaling.

### Amdahl's law (optional analysis)

If a fraction \(s\) of the sequential work is inherently serial, the idealized maximum speedup for infinitely many processors is:

\[
S_{\max}=\frac{1}{s}
\]

Use this as an explanatory bound, not as a measured result. Actual performance also depends on memory bandwidth, thread scheduling, cache behavior, synchronization, and the parallel fraction of each stage.

## 17. Understanding the Results

Use the following approach when analysing a run:

1. Confirm that every parallel correctness row says `PASS`.
2. Compare sequential and parallel mean compression time for the same image.
3. Compare mean decompression time separately; compression and decompression may scale differently.
4. Calculate speedup using matching timing scopes and the same input.
5. Calculate efficiency using the actual configured thread count.
6. Inspect individual stages to see where time is spent and where parallelization helps.
7. Compare compressed size and quality metrics. The output size and quality should remain the same across thread counts if the bitstream is byte-identical.
8. Compare results across image sizes and content types. Noise-like images can compress differently from smooth images or regular patterns.
9. Avoid assuming that more threads always improve performance. Thread creation/scheduling, synchronization, serial stages, and memory bandwidth can limit scaling.
10. Use results collected on the same machine, compiler, build flags, and run conditions for fair comparisons.

## 18. Current Experimental Data

The ZIP supplied with this README contains saved CSV files. At the time this README was prepared:

- `batch_results.csv` contains **60 rows**: 10 images × (one sequential row + five parallel thread configurations).
- The parallel thread configurations represented in that file are **1, 2, 4, 8, and 12 threads**.
- The CSV contains **50 parallel rows marked `PASS`**, plus 10 sequential baseline rows.
- `scaling_size.csv` contains 16 rows: four image sizes × (one sequential configuration + parallel configurations at 4, 8, and 12 threads).
- `scaling_collection.csv` contains derived totals for collection sizes of 1, 2, 5, and 10 images.

These are the contents of the supplied files, not a guarantee that a fresh run on another machine will produce the same timing values. Treat the included CSVs and graphs as existing experimental artifacts and preserve them before regenerating them. Re-run the experiments on the final build if the report requires final submission results.

## 19. Limitations and Interpretation

1. **Custom codec, not standard JPEG:** the output is a project-specific `.bin` format. The JPEG-style transform stages do not make the bitstream a valid standard JPEG file.
2. **Fixed quality setting:** quality is set to 75 in source code rather than exposed as a command-line argument.
3. **No chroma subsampling:** the implementation uses three full-resolution colour planes (4:4:4).
4. **Fixed chunk size:** the entropy-coding chunk is 512 blocks; a separate configurable partition-count experiment is not implemented by this setting alone.
5. **Serial stages remain:** Huffman table construction and ordered stream merging are serial; header parsing is serial.
6. **Collection scaling is derived:** the collection CSV sums per-image measurements rather than timing a concurrently processed collection.
7. **Metrics are calculated after timed work:** output writing and quality calculations are excluded from the compression/decompression timings.
8. **SSIM is simplified:** the implementation uses global grayscale statistics rather than a conventional local-window SSIM implementation.
9. **CSV repetition metadata:** `scaling_size.csv` does not store its repetition count.
10. **Append-only result file:** `single_results.csv` accumulates rows across invocations, so mixed runs need to be handled carefully.
11. **Portability:** `run_scaling.py` currently assumes Windows `.exe` filenames.
12. **Benchmark stability:** the supplied scripts do not report standard deviation, confidence intervals, or machine load. Repetition counts and run conditions should be recorded for defensible performance analysis.

## 20. Troubleshooting

### `g++: command not found`

Open the MSYS2 **UCRT64** terminal and install/check the UCRT64 compiler. Make sure the compiler's `bin` directory is available in that terminal's PATH.

### `omp.h` or OpenMP link errors

Compile `parallel_single.cpp` with `-fopenmp`. Ensure that the selected compiler supports OpenMP.

### `Cannot read image`

Check that the input path is correct relative to the current working directory. From `src/`, the included dataset is at `../image_dataset/`.

### `python run_batch.py` cannot find an executable

Compile both executables first and run the Python script from `src/`. Check that the executable names match the platform.

### `run_scaling.py` cannot find `.exe` files

The current scaling script uses Windows-specific executable paths. On MSYS2/Windows, compile to `sequential_single.exe` and `parallel_single.exe`. For Linux/macOS, update the script to choose the executable suffix based on the platform.

### Missing Python module

Install the relevant packages:

```bash
python -m pip install --user Pillow pandas matplotlib
```

### Correctness says `FAIL`

- Ensure the sequential and parallel executables were compiled with matching `-O2` and `-ffp-contract=off` options.
- Re-run the sequential baseline and all desired parallel thread counts on the same image.
- Do not compare output files from different input images.
- Check whether output files are stale or a run failed.
- Inspect the latest rows in `single_results.csv` and the `Correct` column in `batch_results.csv`.

### Timings are unexpectedly slow or inconsistent

- Use a reasonably large image.
- Use the same build flags for both programs.
- Avoid running other CPU-heavy applications during measurement.
- Increase repetitions and report mean as well as best time.
- Do not compare results collected under different machine loads or compiler configurations as if they were one controlled experiment.

## 21. Recommended Project Improvements

The following are recommendations, not claims that these features already exist.

### High priority — strengthen experimental validity

1. **Add a clean benchmark mode.** Allow scripts to start a new CSV rather than silently appending to old data, while optionally preserving previous runs in timestamped files.
2. **Store run metadata.** Add a run ID, timestamp, repetition count, compiler version, build flags, operating system, CPU, image dimensions, and thread count to each result row or to a separate metadata file.
3. **Report variability.** Save every repetition or at least mean, standard deviation, median, and best time. Add error bars to plots.
4. **Make timing scope explicit.** Keep separate fields for loading, compression, decompression, writing output, and metric computation. Name totals so they cannot be mistaken for end-to-end time.
5. **Use a repeatable correctness suite.** Have one script run all selected thread counts and fail the run if any compressed output or reconstruction differs.
6. **Fix scaling portability.** Detect the executable suffix and invoke subprocesses with a consistent working directory.
7. **Make output paths explicit.** Pass an output/results directory to the programs or resolve all paths relative to the source script rather than the current shell directory.

### High priority — improve scientific interpretation

8. **Replace or clearly rename the simplified SSIM metric.** Use a validated implementation of standard SSIM and document the implementation/library/version.
9. **Separate raw-RGB and original-file ratios.** Add explicit CSV columns such as `RawRGB_bytes`, `Original_bytes`, `Compressed_bytes`, `Ratio_rawRGB`, `Ratio_original_file`, `Saving_rawRGB_pct`, and `Saving_original_file_pct`.
10. **Add standard deviation and confidence intervals.** Timing claims should not rely on one best result.
11. **Summarize across all images.** Report per-image results and aggregate mean/median results by thread count. Keep image-content differences visible rather than presenting one image as representative of all inputs.
12. **Make graphs data-driven.** Add charts for parallel efficiency, stage-level timings, and aggregate results; verify the thread-count rows and experiment ID before plotting.

### Medium priority — improve the codec experiment

13. **Expose quality as a command-line parameter.** Test several quality values, while ensuring sequential and parallel runs use the same value.
14. **Make chunk size configurable.** Compare multiple chunk sizes and record them in CSV. This is distinct from merely changing the number of OpenMP threads.
15. **Benchmark a partition-count experiment if required.** If the project rubric explicitly asks for 1/4/16/64 partitions, implement that as a separate configurable experiment and record the actual number of chunks/partitions. The current fixed `CHUNK_BLOCKS = 512` does not, by itself, satisfy that experiment.
16. **Measure serial bottlenecks.** Record stage timings and estimate the serial fraction; use the analysis to prioritize optimization.
17. **Add input validation and bounds checks.** Test empty, corrupt, truncated, very small, and odd-dimension images. Verify malformed custom bitstreams are rejected safely.
18. **Document the bitstream format.** Add a concise specification for the magic identifier, version, dimensions, quality, Huffman tables, chunk offsets, predictor values, and payload.
19. **Add unit tests.** Test DCT/IDCT behavior, quantization/dequantization, Huffman coding/decoding, padding/cropping, metric functions, and malformed inputs.
20. **Add memory-use measurements.** Large-image processing allocates several full-resolution buffers; measure peak memory as image dimensions increase.

### Optional — usability and maintainability

21. Add a `CMakeLists.txt` or build script to standardize compiler flags.
22. Add command-line options such as `--threads`, `--repetitions`, `--quality`, `--input`, `--output-dir`, and `--csv`.
23. Separate codec logic, command-line handling, and benchmark reporting into smaller modules.
24. Add a CI workflow that compiles both versions and runs a small correctness test.
25. Keep generated outputs and temporary benchmark artifacts out of source control unless they are intentionally included as reference results.
26. Add a small demo section showing the input image, reconstructed image, metrics, and correctness result.

## 22. Reproducibility Checklist

Before collecting final measurements:

- [ ] Record the CPU, OS, compiler version, and compiler flags.
- [ ] Compile sequential and parallel versions from the same source revision.
- [ ] Use the same optimization flags for both executables.
- [ ] Use the same input image and quality setting.
- [ ] Choose and record thread counts and repetitions.
- [ ] Preserve previous CSVs before starting a fresh run.
- [ ] Run the sequential baseline for each image.
- [ ] Run all parallel configurations for the same images.
- [ ] Verify byte-identical compressed output and reconstructed output.
- [ ] Check that no benchmark failed and all intended rows are present.
- [ ] Calculate speedup and efficiency using matching time columns.
- [ ] Regenerate graphs from the CSV corresponding to the reported run.
- [ ] Record whether collection-size results are measured or derived.
- [ ] Explain all limitations, including the custom file format and simplified SSIM.
- [ ] Do not include unverified results or claim planned experiments are complete.

## 23. Academic and Technical Notes

When presenting this project, emphasize that the core parallel-programming question is: **which independent parts of the compression and decompression of one image can be executed concurrently without changing the encoded result?**

A defensible report should distinguish:

- **Implementation facts:** what the C++ source actually does.
- **Measured results:** values recorded by a specific experiment.
- **Derived results:** quantities calculated from recorded measurements.
- **Planned improvements:** features not yet implemented or tested.

The current repository demonstrates within-image OpenMP parallelism for rows, 8×8 blocks, histograms, and fixed-size entropy-coding chunks. It also provides scripts and CSV files for timing, correctness checks, image-size scaling, and derived collection-size comparisons. The quality of the final evaluation will improve most by making runs reproducible, preserving raw repetitions, improving metric definitions, and clearly separating measured from derived results.
