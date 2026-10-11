import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

graphs = Path(__file__).resolve().parent.parent / "graphs"
graphs.mkdir(exist_ok=True)

df = pd.read_csv(Path(__file__).resolve().parent.parent / "results" / "batch_results.csv")

land = df[df["Image"] == "01_landscape.jpg"]
seq = land[land["Mode"] == "sequential"]
par = land[land["Mode"] == "parallel"]

# Compression Time
plt.figure(figsize=(8, 5))
plt.plot(par["Threads"], par["Total_mean_ms"], marker="o")
plt.axhline(seq["Total_mean_ms"].iloc[0], linestyle="--")
plt.xlabel("Threads")
plt.ylabel("Time (ms)")
plt.title("Compression Time vs Threads")
plt.grid(True)
plt.tight_layout()
plt.savefig(graphs / "compression_time.png", dpi=300)
plt.close()

# Decompression Time
plt.figure(figsize=(8, 5))
plt.plot(par["Threads"], par["Decomp_mean_ms"], marker="o")
plt.axhline(seq["Decomp_mean_ms"].iloc[0], linestyle="--")
plt.xlabel("Threads")
plt.ylabel("Time (ms)")
plt.title("Decompression Time vs Threads")
plt.grid(True)
plt.tight_layout()
plt.savefig(graphs / "decompression_time.png", dpi=300)
plt.close()

# Compression Speedup
plt.figure(figsize=(8, 5))
plt.plot(
    par["Threads"],
    par["Comp_speedup"],
    marker="o",
    label="Measured"
)
plt.plot(
    par["Threads"],
    par["Threads"],
    linestyle="--",
    label="Ideal"
)
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.title("Compression Speedup")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(graphs / "compression_speedup.png", dpi=300)
plt.close()

# Decompression Speedup
plt.figure(figsize=(8, 5))
plt.plot(
    par["Threads"],
    par["Decomp_speedup"],
    marker="o",
    label="Measured"
)
plt.plot(
    par["Threads"],
    par["Threads"],
    linestyle="--",
    label="Ideal"
)
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.title("Decompression Speedup")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(graphs / "decompression_speedup.png", dpi=300)
plt.close()

# Compression Ratio
quality = df[
    (df["Mode"] == "sequential") &
    (df["Threads"] == 1)
]

plt.figure(figsize=(12, 6))
plt.bar(quality["Image"], quality["Ratio"])
plt.xlabel("Image")
plt.ylabel("Compression Ratio")
plt.title("Compression Ratio Across Images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(graphs / "compression_ratio.png", dpi=300)
plt.close()

# PSNR
plt.figure(figsize=(12, 6))
plt.bar(quality["Image"], quality["PSNR_dB"])
plt.xlabel("Image")
plt.ylabel("PSNR (dB)")
plt.title("PSNR Across Images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(graphs / "psnr.png", dpi=300)
plt.close()

# SSIM
plt.figure(figsize=(12, 6))
plt.bar(quality["Image"], quality["SSIM"])
plt.xlabel("Image")
plt.ylabel("SSIM")
plt.title("SSIM Across Images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(graphs / "ssim.png", dpi=300)
plt.close()

print("GRAPH GENERATION COMPLETE")
print(f"Graphs saved in {graphs}/")

for f in sorted(graphs.glob("*.png")):
    print(f.name)