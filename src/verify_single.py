"""
verify_single.py  -  correctness check + per-stage speedup table for the
single-image experiment.

Usage:  python verify_single.py            (run from src/, with output/ and results/ at the project root)

1. CORRECTNESS: the compressed file and the reconstructed image produced by every
   parallel run must be byte-for-byte identical to the sequential ones.
2. PERFORMANCE: reads results/single_results.csv (last row per mode/thread count)
   and prints speedup / efficiency per stage and for the whole compression.
"""
import csv, sys
from pathlib import Path

root = Path(__file__).resolve().parent
out = root.parent / "output" / "single"
csv_path = root.parent / "results" / "single_results.csv"

rows = list(csv.DictReader(open(csv_path, newline=""))) if csv_path.exists() else []
seq_rows = [r for r in rows if r["Mode"] == "sequential"]
if not seq_rows:
    print("ERROR: run sequential_single first (no sequential run found in single_results.csv)")
    sys.exit(1)
image_stem = Path(seq_rows[-1]["Image"]).stem
image_rows = [r for r in rows if Path(r["Image"]).stem == image_stem]
seq_bin = out / f"{image_stem}_sequential.bin"
seq_png = out / f"{image_stem}_sequential_reconstructed.png"
if not seq_bin.exists():
    print(f"ERROR: sequential output for {image_stem} is missing ({seq_bin})")
    sys.exit(1)

print("=" * 60)
print("CORRECTNESS  (parallel vs sequential)")
print("=" * 60)
all_ok = True
parallel_pattern = f"{image_stem}_parallel_t*.bin"
for pb in sorted(out.glob(parallel_pattern), key=lambda p: int(p.stem.rsplit("_t", 1)[1])):
    t = pb.stem.rsplit("_t", 1)[1]
    same_bin = pb.read_bytes() == seq_bin.read_bytes()
    pp = out / f"{image_stem}_parallel_t{t}_reconstructed.png"
    same_png = pp.exists() and seq_png.exists() and pp.read_bytes() == seq_png.read_bytes()
    ok = same_bin and same_png
    all_ok &= ok
    print(f"  threads={t:>2}  compressed file identical: {same_bin!s:5}  "
          f"reconstructed image identical: {same_png!s:5}  -> {'PASS' if ok else 'FAIL'}")
parallel_outputs = list(out.glob(parallel_pattern))
if not parallel_outputs:
    print("  (no parallel output found - run parallel_single first)")
    all_ok = False
print("OVERALL:", "PASS" if all_ok else "FAIL")

if not csv_path.exists():
    sys.exit(0 if all_ok else 1)

seq = [r for r in image_rows if r["Mode"] == "sequential"]
par = {}
for r in image_rows:
    if r["Mode"] == "parallel":
        par[int(r["Threads"])] = r          # last run per thread count wins
if not seq or not par:
    sys.exit(0 if all_ok else 1)
s = seq[-1]

comp_stages = [("Colour conversion", "Color_ms"), ("DCT + quantization", "DCT_Quant_ms"),
               ("Symbol statistics", "Hist_ms"), ("Build Huffman", "Huffman_ms"),
               ("Huffman encoding", "Encode_ms"), ("Merge + header", "Merge_ms"),
               ("TOTAL compression", "Total_mean_ms")]
decomp_stages = [("Parse header", "Parse_ms"), ("Huffman decoding", "Decode_ms"),
                 ("Dequant + inverse DCT", "IDCT_ms"), ("Colour YCbCr->RGB", "ColorBack_ms"),
                 ("TOTAL decompression", "Decomp_mean_ms")]
ths = sorted(par)

def report(title, stages, total_col, serial_cols, serial_label):
    print("\n" + "=" * 60)
    print(f"{title}   image: {s['Image']}  {s['Width']}x{s['Height']}")
    print("=" * 60)
    print("TIMES (ms)")
    print(f"{'Stage':<24}{'seq':>9}" + "".join(f"{('T=' + str(t)):>9}" for t in ths))
    for name, col in stages:
        print(f"{name:<24}{float(s[col]):>9.3f}" + "".join(f"{float(par[t][col]):>9.3f}" for t in ths))
    print("\nSPEEDUP = sequential time / parallel time")
    print(f"{'Stage':<24}" + "".join(f"{('T=' + str(t)):>9}" for t in ths))
    for name, col in stages:
        print(f"{name:<24}" + "".join(f"{float(s[col]) / max(float(par[t][col]), 1e-9):>9.2f}" for t in ths))
    tot = float(s[total_col])
    print("\nEFFICIENCY (%) = speedup / threads x 100   (TOTAL)")
    print("".join(f"  T={t}: {100 * tot / float(par[t][total_col]) / t:6.1f}%" for t in ths))
    ser = sum(float(s[c]) for c in serial_cols)
    print(f"\nSerial stages ({serial_label}) = {ser:.3f} ms = {100 * ser / tot:.2f}% of sequential total")
    if ser > 0:
        print(f"-> Amdahl upper bound on speedup  1/s = {tot / ser:.1f}x  (ignores thread start-up overhead)")

report("COMPRESSION", comp_stages, "Total_mean_ms", ["Huffman_ms", "Merge_ms"], "build Huffman + merge")
if "Decomp_mean_ms" in s:
    report("DECOMPRESSION", decomp_stages, "Decomp_mean_ms", ["Parse_ms"], "parse header")
sys.exit(0 if all_ok else 1)
