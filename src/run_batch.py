"""
run_batch.py - run sequential + parallel single-image codec on every image in a folder.

Usage (from the folder containing the compiled programs):
    python run_batch.py image_dataset            # defaults: threads 1 2 4 8, 5 reps
    python run_batch.py image_dataset 1,2,4,8 10

Output: results/batch_results.csv (one row per image x mode x thread count, with
speedups and a correctness flag) plus a summary table printed at the end.
"""
import csv, subprocess, sys, shutil
from pathlib import Path

root = Path(__file__).resolve().parent
folder = Path(sys.argv[1] if len(sys.argv) > 1 else "image_dataset")
threads = [int(t) for t in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4,8").split(",")]
reps = sys.argv[3] if len(sys.argv) > 3 else "5"
exe = ".exe" if sys.platform.startswith("win") or (root / "sequential_single.exe").exists() else ""
seq_exe, par_exe = str(root / f"sequential_single{exe}"), str(root / f"parallel_single{exe}")
src_csv = root / "results" / "single_results.csv"
out = root / "output" / "single"
dst_csv = root / "results" / "batch_results.csv"

imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
if not imgs:
    sys.exit(f"No images found in {folder}")

def last_row():
    return list(csv.DictReader(open(src_csv, newline="")))[-1]

rows = []
def norm(r):
    r["Image"] = Path(r["Image"]).name
    return r
for img in imgs:
    print(f"\n### {img.name}")
    subprocess.run([seq_exe, str(img), reps], check=True, stdout=subprocess.DEVNULL)
    s = norm(last_row())
    seq_bin = (out / "sequential.bin").read_bytes()
    seq_png = (out / "sequential_reconstructed.png").read_bytes()
    s["Correct"] = "baseline"
    rows.append(s)
    for t in threads:
        subprocess.run([par_exe, str(img), str(t), reps], check=True, stdout=subprocess.DEVNULL)
        p = norm(last_row())
        ok = ((out / f"parallel_t{t}.bin").read_bytes() == seq_bin and
              (out / f"parallel_t{t}_reconstructed.png").read_bytes() == seq_png)
        p["Correct"] = "PASS" if ok else "FAIL"
        rows.append(p)
        print(f"  t={t:<2} comp {float(p['Total_mean_ms']):8.2f} ms  "
              f"decomp {float(p['Decomp_mean_ms']):8.2f} ms  {p['Correct']}")

# speedups relative to the sequential run of the same image
seq_by_img = {r["Image"]: r for r in rows if r["Mode"] == "sequential"}
for r in rows:
    s = seq_by_img[r["Image"]]
    r["Comp_speedup"] = round(float(s["Total_mean_ms"]) / float(r["Total_mean_ms"]), 3)
    r["Decomp_speedup"] = round(float(s["Decomp_mean_ms"]) / float(r["Decomp_mean_ms"]), 3)

dst_csv.parent.mkdir(exist_ok=True)
with open(dst_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print("\n" + "=" * 78)
print(f"{'image':<22}{'WxH':>11}{'ratio':>8}{'PSNR':>8}{'SSIM':>8}  speedup (comp/decomp) by threads")
for img in imgs:
    rr = [r for r in rows if r["Image"] == img.name]
    s = rr[0]
    sp = "  ".join(f"{r['Threads']}t:{r['Comp_speedup']:.2f}/{r['Decomp_speedup']:.2f}"
                   for r in rr if r["Mode"] == "parallel")
    print(f"{img.name:<22}{s['Width']+'x'+s['Height']:>11}{float(s['Ratio']):>8.2f}"
          f"{float(s['PSNR_dB']):>8.2f}{float(s['SSIM']):>8.4f}  {sp}")
print("\nAll parallel outputs identical to sequential:",
      "YES" if all(r["Correct"] != "FAIL" for r in rows) else "NO - check batch_results.csv")
print("Saved:", dst_csv)
