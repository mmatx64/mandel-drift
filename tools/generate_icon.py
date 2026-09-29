"""Generate the app's mathematical Mandelbrot icon (requires NumPy and Pillow)."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

root = Path(__file__).resolve().parents[1]
assets = root / "assets"
assets.mkdir(exist_ok=True)
size = 768
y, x = np.mgrid[0:size, 0:size] / (size - 1)
c = (-2.25 + x * 3.0) + 1j * ((y - 0.5) * 3.0)
z = np.zeros_like(c)
alive = np.ones(c.shape, dtype=bool)
escape = np.zeros(c.shape)
for iteration in range(240):
    z[alive] = z[alive] ** 2 + c[alive]
    escaped = alive & (abs(z) > 16)
    escape[escaped] = iteration + 1 - np.log2(np.log2(abs(z[escaped])))
    alive[escaped] = False

mask = Image.fromarray((alive * 255).astype(np.uint8))
# A narrow luminous outline retains the familiar silhouette at taskbar sizes.
outer = np.asarray(mask.filter(ImageFilter.MaxFilter(9)), dtype=float) / 255
inner = np.asarray(mask, dtype=float) / 255
edge = np.maximum(0, outer - inner)
glow = np.asarray(Image.fromarray((edge * 255).astype(np.uint8)).filter(
    ImageFilter.GaussianBlur(10)), dtype=float) / 255
mix = np.clip(x * 0.8 + y * 0.4, 0, 1)[..., None]
neon = np.array([25, 233, 255]) * (1 - mix) + np.array([255, 50, 185]) * mix
background = np.zeros((size, size, 3)) + [12, 8, 32]
detail = np.where(alive, 0, np.clip((escape - 6) / 22, 0, 1))
color = background + neon * (edge[..., None] * .94 + glow[..., None] * 1.8
                            + detail[..., None] * .35)
color[alive] = np.array([5, 4, 17])
# Rounded-square tile with transparent corners.
qx = np.abs(x - .5) - .39
qy = np.abs(y - .5) - .39
distance = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - .08
alpha = np.clip(.5 - distance * size, 0, 1) * 255
rgba = np.dstack((np.clip(color, 0, 255), alpha)).astype(np.uint8)
icon = Image.fromarray(rgba).resize((256, 256), Image.Resampling.LANCZOS)
icon.save(assets / "mandeldrift.png")
icon.save(assets / "mandeldrift.ico", sizes=[(n, n) for n in (16, 24, 32, 48, 64, 128, 256)])
print(assets / "mandeldrift.ico")
