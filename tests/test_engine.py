#!/usr/bin/env python3
"""Regression tests for the Ortho Scenery Generator engine.

Covers the pure, non-GUI parts: the DDS size math, the BC1/DXT1 encoder and its
validators, the water placeholder, the gray-hole detector and the water/land
classifier. The GUI (tkinter) is stubbed so the module imports without a display.

Run:  python tests/test_engine.py          (plain stdlib, no pytest needed)
Exit code 0 = all passed, 1 = a failure, 77 = skipped (deps missing).
"""
import os
import sys
import types

# --- make the top-level module importable and stub tkinter -----------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _name in ("tkinter", "tkinter.ttk", "tkinter.messagebox",
              "tkinter.scrolledtext"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
_tk = sys.modules["tkinter"]
for _attr in ("Tk Frame Label Button Canvas Scrollbar StringVar IntVar DoubleVar "
              "BooleanVar Entry Radiobutton Checkbutton LEFT RIGHT X Y BOTH TOP "
              "BOTTOM FLAT DISABLED NORMAL END WORD").split():
    setattr(_tk, _attr, object)

try:
    import numpy as np
    from PIL import Image
    import ortho_scenery_generator as o
except Exception as exc:                       # deps missing -> skip, don't fail
    print(f"SKIP: engine import failed ({exc}). Install requirements.txt.")
    sys.exit(77)

SCRATCH = os.environ.get("TMPDIR", "/tmp")
_results = []


def check(name, cond):
    _results.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def _decode_bc1_base(path, px):
    """Decode the base mip level of a BC1 DDS to an (px,px,3) array."""
    with open(path, "rb") as f:
        data = f.read()
    nb = px // 4
    dec = np.zeros((px, px, 3), np.uint8)
    off = 128

    def d565(v):
        r = (v >> 11) & 31; g = (v >> 5) & 63; b = v & 31
        return [(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)]

    for by in range(nb):
        for bx in range(nb):
            c0 = data[off] | (data[off + 1] << 8)
            c1 = data[off + 2] | (data[off + 3] << 8)
            bits = (data[off + 4] | (data[off + 5] << 8)
                    | (data[off + 6] << 16) | (data[off + 7] << 24))
            off += 8
            e0, e1 = d565(c0), d565(c1)
            if c0 > c1:
                pal = [e0, e1,
                       [(2 * e0[k] + e1[k]) // 3 for k in range(3)],
                       [(e0[k] + 2 * e1[k]) // 3 for k in range(3)]]
            else:
                pal = [e0, e1,
                       [(e0[k] + e1[k]) // 2 for k in range(3)], [0, 0, 0]]
            for i in range(16):
                dec[by * 4 + i // 4, bx * 4 + i % 4] = pal[(bits >> (2 * i)) & 3]
    return dec


def test_dds_size_math():
    # A complete BC1 mip chain: base + 1/4 + 1/16 + ... ~= base * 4/3, + header.
    for px in (2048, 4096):
        base = (px // 4) ** 2 * 8
        total = o._dxt1_dds_size(px)
        check(f"_dxt1_dds_size({px}) > base+header",
              total > base + 128)
        check(f"_dxt1_dds_size({px}) ~ 4/3 base",
              abs(total - 128 - base * 4 / 3) < base * 0.01)


def test_bc1_roundtrip():
    rng = np.random.default_rng(1)
    for px in (2048, 4096):
        arr = rng.integers(20, 210, (px, px, 3)).astype(np.uint8)
        xx, _ = np.meshgrid(np.linspace(0, 60, px), np.linspace(0, 60, px))
        arr[:, :, 1] = np.clip(arr[:, :, 1].astype(int)
                               + (np.sin(xx) * 30).astype(int), 0, 255).astype(np.uint8)
        p = os.path.join(SCRATCH, f"_t_bc1_{px}.dds")
        o.write_dds_bc1(Image.fromarray(arr, "RGB"), p)
        check(f"BC1 {px}: dds_is_valid", o.dds_is_valid(p, px))
        check(f"BC1 {px}: dds_self_valid", o.dds_self_valid(p))
        check(f"BC1 {px}: size == _dxt1_dds_size",
              os.path.getsize(p) == o._dxt1_dds_size(px))
        check(f"BC1 {px}: no false gray holes", o.dds_gray_hole_tiles(p, px) == 0)
        os.remove(p)


def test_bc1_quality_on_smooth():
    px = 1024
    xx, yy = np.meshgrid(np.linspace(0, 255, px), np.linspace(0, 255, px))
    arr = np.stack([xx, yy, (xx + yy) / 2], -1).astype(np.uint8)
    p = os.path.join(SCRATCH, "_t_bc1_smooth.dds")
    o.write_dds_bc1(Image.fromarray(arr, "RGB"), p)
    err = float(np.abs(_decode_bc1_base(p, px).astype(int) - arr.astype(int)).mean())
    check(f"BC1 smooth mean abs error < 6 (got {err:.2f})", err < 6.0)
    os.remove(p)


def test_bc1_encoder_matches_bruteforce():
    # The one-endpoint-at-a-time index selection must match a full argmin.
    rng = np.random.default_rng(7)
    blocks = rng.integers(0, 255, (40, 40, 16, 3)).astype(np.int32)
    palette = rng.integers(0, 255, (40, 40, 4, 3)).astype(np.int32)
    diff = blocks[:, :, :, None, :] - palette[:, :, None, :, :]
    idx_brute = (diff * diff).sum(axis=-1).argmin(axis=-1)
    idx = np.zeros((40, 40, 16), dtype=np.uint32)
    best = ((blocks - palette[:, :, 0:1, :]) ** 2).sum(axis=-1)
    for k in range(1, 4):
        d = ((blocks - palette[:, :, k:k + 1, :]) ** 2).sum(axis=-1)
        take = d < best
        idx[take] = k
        best[take] = d[take]
    check("BC1 index selection matches brute-force argmin",
          bool((idx == idx_brute).all()))


def test_water_placeholder():
    p = os.path.join(SCRATCH, "_t_water.dds")
    ok = o.write_water_dds(p, o.WATER_RGB)
    check("water placeholder write returns True", ok)
    check("water placeholder dds_is_valid(32)", o.dds_is_valid(p, 32))
    check("water placeholder size exact",
          os.path.getsize(p) == o._dxt1_dds_size(o.WATER_PLACEHOLDER_PX))
    os.remove(p)


def test_gray_hole_detection():
    px = 2048
    gray = Image.new("RGB", (px, px), (60, 60, 60))
    p = os.path.join(SCRATCH, "_t_gray.dds")
    o.write_dds_bc1(gray, p)
    # GROUP_TILES x GROUP_TILES regions should all read as holes.
    check("all-gray group -> GROUP_TILES^2 holes",
          o.dds_gray_hole_tiles(p, px) == o.GROUP_TILES * o.GROUP_TILES)
    os.remove(p)


def test_classify_tile():
    ocean = Image.fromarray(np.tile(np.array([7, 39, 50], np.uint8), (128, 128, 1)), "RGB")
    land = Image.fromarray(np.tile(np.array([90, 80, 40], np.uint8), (128, 128, 1)), "RGB")
    check("ocean classifies as water", o.classify_tile(ocean)[0] == "water")
    check("land classifies as land", o.classify_tile(land)[0] == "land")
    # numpy and non-numpy paths agree on a solid tile
    o.HAS_NUMPY = False
    r_non = o.classify_tile(ocean)[1]
    o.HAS_NUMPY = True
    r_np = o.classify_tile(ocean)[1]
    check("numpy/non-numpy agree on solid ocean", abs(r_non - r_np) < 1e-9)


def test_slippy_roundtrip():
    for lon, lat, z in [(11.46, 51.75, 16), (-0.1, 51.5, 14), (2.3, 48.9, 17)]:
        x, y = o.lonlat_to_slippy(lon, lat, z)
        lon2, lat2 = o.tile_to_lonlat(x, y, z)
        # tile NW corner must be within one tile west/north of the point
        check(f"slippy roundtrip ({lon},{lat},z{z})",
              lon2 <= lon + 1e-6 and lat2 >= lat - 1e-6)


def main():
    for fn in (test_dds_size_math, test_bc1_roundtrip, test_bc1_quality_on_smooth,
               test_bc1_encoder_matches_bruteforce, test_water_placeholder,
               test_gray_hole_detection, test_classify_tile, test_slippy_roundtrip):
        print(f"\n{fn.__name__}:")
        fn()
    failed = [n for n, ok in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed.")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)
    print("All engine tests passed.")


if __name__ == "__main__":
    main()
