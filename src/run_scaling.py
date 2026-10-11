import csv, subprocess
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parent
src_img = root.parent / "image_dataset" / "01_landscape.jpg"
tmp = root.parent / "scaling_images"; tmp.mkdir(exist_ok=True)
res = root.parent / "results"
sizes = [(500,333),(1000,667),(2000,1333),(3000,2000)]
threads = [4, 8, 12]
reps = "5"

def last_row():
    return list(csv.DictReader(open(res / "single_results.csv", newline="")))[-1]

# Experiment 1: image size
rows = []
base = Image.open(src_img).convert("RGB")
for w, h in sizes:
    p = tmp / f"img_{w}x{h}.jpg"
    base.resize((w, h), Image.LANCZOS).save(p, quality=95)
    subprocess.run([str(root/"sequential_single.exe"), str(p), reps], check=True, stdout=subprocess.DEVNULL)
    r = last_row(); rows.append(("sequential", 1, w, h, r["Total_mean_ms"], r["Decomp_mean_ms"]))
    for t in threads:
        subprocess.run([str(root/"parallel_single.exe"), str(p), str(t), reps], check=True, stdout=subprocess.DEVNULL)
        r = last_row(); rows.append(("parallel", t, w, h, r["Total_mean_ms"], r["Decomp_mean_ms"]))
    print("done", w, h, flush=True)

with open(res / "scaling_size.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["Mode","Threads","W","H","Comp_ms","Decomp_ms"]); wr.writerows(rows)

# Experiment 2: collection size, from batch_results.csv
b = list(csv.DictReader(open(res / "batch_results.csv", newline="")))
names = sorted({r["Image"] for r in b})
out = [["N_images","Mode","Threads","Total_comp_ms","Total_decomp_ms"]]
for n in (1, 2, 5, 10):
    sel = names[:n]
    for mode, t in [("sequential","1")] + [("parallel", str(x)) for x in (1,2,4,8,12)]:
        rr = [r for r in b if r["Image"] in sel and r["Mode"] == mode and (mode == "sequential" or r["Threads"] == t)]
        out.append([n, mode, t, round(sum(float(r["Total_mean_ms"]) for r in rr),2),
                    round(sum(float(r["Decomp_mean_ms"]) for r in rr),2)])
with open(res / "scaling_collection.csv", "w", newline="") as f:
    csv.writer(f).writerows(out)
print("Saved scaling_size.csv and scaling_collection.csv")
