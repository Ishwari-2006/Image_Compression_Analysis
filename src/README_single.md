# Single-image compression + decompression: sequential vs. parallel (inside one image)

Task: take **one** image, compress and decompress it sequentially, find which parts of the
compression and decompression process are independent of each other, parallelize exactly those parts
with OpenMP, and compare.

No OpenCV or libjpeg is needed - only `stb_image.h` / `stb_image_write.h`
(included, used just to read the JPEG and save PNGs). The codec itself is written
in `codec_common.h`, so every stage can be timed and parallelized.

## Files

| File | Purpose |
|---|---|
| `codec_common.h` | shared code: colour conversion, DCT, quantization, Huffman, decoder, metrics |
| `sequential_single.cpp` | compression + decompression in one thread, every stage timed |
| `parallel_single.cpp` | same pipeline, independent stages parallelized with OpenMP |
| `verify_single.py` | checks parallel output == sequential output, prints speedup table |

## Build and run (MSYS2 UCRT64 or any g++)

```
g++ sequential_single.cpp -o sequential_single -std=c++17 -O2 -ffp-contract=off
g++ parallel_single.cpp   -o parallel_single   -std=c++17 -O2 -fopenmp -ffp-contract=off

./sequential_single  <image> [repetitions]
./parallel_single    <image> <threads> [repetitions]
python verify_single.py
```

Example:

```
./sequential_single photo.jpg 10
./parallel_single   photo.jpg 1 10
./parallel_single   photo.jpg 2 10
./parallel_single   photo.jpg 4 10
./parallel_single   photo.jpg 8 10
python verify_single.py
```

* Use a **large image** (e.g. 3000x2000). A 512x480 image compresses in ~8 ms,
  which is too short for thread start-up cost to pay off.
* `repetitions` repeats only the compression and reports the mean, so timings are stable.
* Keep `-ffp-contract=off` for both programs so the floating-point results (and
  therefore the compressed bytes) are identical.

## Pipeline and dependency analysis

```
RGB image
  -> 1 colour conversion (RGB -> YCbCr)
  -> 2 8x8 DCT + quantization
  -> 3 symbol statistics (histogram)
  -> 4 build Huffman tables
  -> 5 Huffman encoding
  -> 6 join streams + header
  -> compressed file
```

| # | Stage | Depends on | Independent unit | Parallel? |
|---|---|---|---|---|
| 1 | Colour conversion | the same input pixel only | image row / pixel | **Yes** - `parallel for` over rows |
| 2 | DCT + quantization | its own 8x8 block only | 8x8 block (per component) | **Yes** - `parallel for` over blocks |
| 3 | Symbol statistics | quantized coefficients (all exist after stage 2) | block | **Yes** - private histogram per thread, added at the end |
| 4 | Build Huffman | **all** counts from stage 3 | none (tiny tree, <=256 symbols) | No - serial |
| 5 | Huffman encoding | tables from stage 4; not on the bits of other blocks | chunk of blocks | **Yes** - each chunk goes to its own bit buffer |
| 6 | Join streams + header | size of every earlier chunk (output position) | none | No - serial |

Barriers between stages are unavoidable: stage 2 needs stage 1's output, stage 4
needs the counts from every block, stage 5 needs the finished tables.
The DC coefficient is stored as a difference to the previous block, but all
quantized DC values already exist after stage 2, so this does not create a
dependency between blocks in stages 3 and 5.

## Decompression: dependency analysis

```
compressed file
  -> D0 parse header + build decoding tables
  -> D1 Huffman decoding
  -> D2 dequantization + inverse DCT
  -> D3 colour conversion YCbCr -> RGB
  -> reconstructed image
```

| # | Stage | Depends on | Independent unit | Parallel? |
|---|---|---|---|---|
| D0 | Parse header | the file header | none (tiny) | No - serial |
| D1 | Huffman decoding | where its chunk starts and the DC value before the chunk | chunk of 512 blocks | **Yes** |
| D2 | Dequantization + inverse DCT | its own 64 coefficients only | 8x8 block (per component) | **Yes** |
| D3 | Colour conversion | the same pixel of the three planes | image row / pixel | **Yes** |

A normal JPEG stream cannot be decoded in parallel, because the decoder only learns
where block *n+1* starts after decoding block *n*. To fix this, the compressed file
stores a small **chunk table** (bit position + DC predictor for every chunk of 512
blocks, about 14 bytes per chunk). The chunk size is fixed, so the compressed file is
the same for every thread count.

## Correctness

The chunked parallel encoder joins chunks bit-exactly, so the compressed file is
**byte-identical** to the sequential one for every thread count. `verify_single.py`
checks this for the `.bin` file and the reconstructed `.png`.
Each program also decodes its own file and checks that the decoded coefficients are
identical to the encoder's (`Decoded coefficients identical to encoder : YES`) before
computing PSNR/SSIM. The reconstructed `.png` from every parallel run is identical to the sequential one.

## Notes

* Compression ratio is reported two ways: against raw RGB (W x H x 3) and against
  the original input file (which is already a JPEG).
* The Huffman tables are stored in the header, so very small images have a visible header overhead.
* Reading the input file and the metric calculation (PSNR/SSIM) are not timed.
* Compression and decompression are timed separately; each is repeated `repetitions` times.
