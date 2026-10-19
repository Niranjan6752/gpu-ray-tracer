# Compare two PPM images pixel by pixel (e.g. CPU vs GPU render).
# usage: python3 compare.py [a.ppm] [b.ppm]
import sys
import numpy as np
from PIL import Image

a_path = sys.argv[1] if len(sys.argv) > 1 else "../images/output/cpu.ppm"
b_path = sys.argv[2] if len(sys.argv) > 2 else "../images/output/gpu.ppm"

# int32, not uint8: uint8 subtraction wraps around (3 - 5 = 254)
a = np.asarray(Image.open(a_path), dtype=np.int32)
b = np.asarray(Image.open(b_path), dtype=np.int32)

if a.shape != b.shape:
    sys.exit(f"Different sizes: {a.shape} vs {b.shape}")

diff = np.abs(a - b)               # per channel, 0..255
px = diff.max(axis=2)              # worst channel per pixel
n = px.size

print(f"{a_path}  vs  {b_path}  ({a.shape[1]}x{a.shape[0]})")
print(f"max difference      : {px.max()}")
print(f"mean difference     : {diff.mean():.4f}")
print(f"pixels identical    : {100 * np.mean(px == 0):.3f}%")
print(f"pixels differ > 2   : {100 * np.mean(px > 2):.3f}%  ({int(np.sum(px > 2))} of {n})")

# Save a difference image: black = same, brighter = bigger difference (x10 so small diffs show)
Image.fromarray(np.clip(px * 10, 0, 255).astype(np.uint8)).save("../images/output/diff.png")
print("wrote ../images/output/diff.png")