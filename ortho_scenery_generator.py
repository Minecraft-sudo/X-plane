# -*- coding: utf-8 -*-
"""
Ortho4XP-compatible Scenery Generator for X-Plane 12
=====================================================
Run directly:  python gemini-code-1783082591228.py

The GUI opens automatically:
  • Real Apple Maps satellite background (same source as tile generation)
  • Proper Web Mercator projection
  • Adaptive grid: cell size shrinks as you zoom in the map
      zoomed out  → 2°/1°  cells    (continent view)
      zoomed in   → 0.5°/0.25° cells (region view)
      very close  → 0.1°/0.05° cells (airport / partial tile)
  • Left-click / drag  → select cells
  • Right-click        → deselect
  • Ctrl+drag / scroll → pan & zoom
  • Download only the area you selected (no need to grab a full 1° tile)
  • Zoom levels 14 – 19  (new: 19 added)
  • Real-time progress bar with ETA
"""

import os, io, re, json, math, shutil, struct, time, threading, subprocess, random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import requests
from PIL import Image, ImageTk

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

Image.MAX_IMAGE_PIXELS = None


# ===========================================================================
#  GENERATION ENGINE
# ===========================================================================

class AppleTokenService:
    _DDGO  = "https://duckduckgo.com/local.js?get_mk_token=1"
    _APPLE = ("https://cdn.apple-mapkit.com/ma/bootstrap"
               "?apiVersion=2&mkjsVersion=5.79.95&poi=1")

    def __init__(self):
        self.access_key = None
        self.version    = None
        self._lock      = threading.Lock()
        self._refreshes = 0

    def _parse(self, body):
        for src in body.get("tileSources", []):
            if src.get("tileSource") == "satellite":
                p = src["path"]
                return {"version":    p.split("v=")[1].split("&")[0],
                        "access_key": p.split("accessKey=")[1].split("&")[0]}
        raise RuntimeError("No satellite source in Bootstrap response.")

    def refresh(self, attempts=4):
        # Thread-safe: when many parallel downloads hit a 401 at once, only the
        # first actually refreshes; the rest see the counter advanced and reuse
        # the fresh token instead of stampeding the (rate-limited) endpoint.
        before = self._refreshes
        with self._lock:
            if self._refreshes != before and self.access_key:
                return
            last = None
            for i in range(attempts):
                try:
                    dd = requests.get(self._DDGO, timeout=10)
                    dd.raise_for_status()
                    ap = requests.get(self._APPLE,
                                      headers={"Origin":        "https://duckduckgo.com",
                                               "Authorization": f"Bearer {dd.text.strip()}"},
                                      timeout=10)
                    ap.raise_for_status()
                    m = self._parse(ap.json())
                    self.access_key = m["access_key"]
                    self.version    = m["version"]
                    self._refreshes += 1
                    return
                except Exception as exc:
                    last = exc
                    if i < attempts - 1:
                        time.sleep(1.5 * (2 ** i))   # 1.5s, 3s, 6s
            raise RuntimeError(f"Apple token failed after {attempts} attempts: {last}")

    def tile_url(self, zoom, tx, ty, *, hd=True):
        sz = "2&scale=2" if hd else "1&scale=1"
        return (f"https://sat-cdn.apple-mapkit.com/tile?"
                f"style=7&size={sz}&z={zoom}&x={tx}&y={ty}"
                f"&v={self.version}&accessKey={self.access_key}")


# Parallelism for tile downloads. Apple's tile CDN (sat-cdn) handles many
# concurrent requests fine. Measured: 10->16 gives a real ~1.7x gain, beyond
# that we're bandwidth-bound on a 50 Mbit line (own sandbox saturates at the
# same ~50 Mbit regardless of worker count). Kept a bit above that measured
# plateau so a faster line (e.g. gigabit) has headroom to actually use it --
# raise further via "Parallel groups" in the GUI, which multiplies this.
TILE_WORKERS = 16


class _Bandwidth:
    """Global running total of downloaded bytes, for a live Mbit/s readout."""
    def __init__(self):
        self._lock = threading.Lock()
        self.total = 0
    def add(self, n):
        with self._lock:
            self.total += n
    def read(self):
        return self.total


BW = _Bandwidth()


def make_tile_session(pool_size=200):
    """One Session + big connection pool, meant to be shared across an ENTIRE
    generation run (all groups), not recreated per group/call. Recreating a
    Session per group throws away keep-alive connections between groups for
    no reason -- this lets connections stay warm across the whole run."""
    sess = requests.Session()
    sess.headers.update({"Referer": "https://duckduckgo.com/",
                         "User-Agent": "Mozilla/5.0"})
    sess.mount("https://", requests.adapters.HTTPAdapter(
        pool_connections=pool_size, pool_maxsize=pool_size))
    return sess


def download_tiles_parallel(tokens, coords, zoom, *, workers=TILE_WORKERS,
                            max_ret=3, hd=True, stop_event=None,
                            executor=None, session=None):
    """Download many tiles concurrently.

    coords: iterable of (tx, ty). Returns {(tx, ty): PIL.Image(RGB)} for every
    tile that succeeded (missing ones are simply absent).

    executor/session: pass a shared ThreadPoolExecutor/Session (see
    make_tile_session) to fetch into a run-wide pool instead of spinning up a
    fresh one per call. Without this, N groups running "in parallel" each open
    their OWN worker pool, so total OS threads = group_workers x tile_workers
    -- e.g. 24 groups x 16 = ~400 threads, which measurably HURTS throughput
    on Windows (verified: nested pools got slower at higher group_workers,
    while one flat pool kept scaling). A shared pool decouples "how many
    groups are being staged" from "how many raw connections are open," so a
    faster line has one clean, high-headroom knob instead of a multiplying one.
    """
    coords = list(coords)
    out = {}
    sess = session or requests.Session()
    if session is None:
        sess.headers.update({"Referer": "https://duckduckgo.com/",
                             "User-Agent": "Mozilla/5.0"})
        n = max(1, min(workers, len(coords)))
        sess.mount("https://", requests.adapters.HTTPAdapter(
            pool_connections=n, pool_maxsize=n))

    def fetch(coord):
        tx, ty = coord
        for attempt in range(1, max_ret + 1):
            if stop_event is not None and stop_event.is_set():
                return coord, None
            try:
                r = sess.get(tokens.tile_url(zoom, tx, ty, hd=hd), timeout=15)
                if r.status_code == 429:
                    # Rate-limited. Refreshing the token does NOT lift a rate
                    # limit, so retrying instantly just deepens the throttle:
                    # every worker keeps hammering, tiles start failing, and a
                    # single failed tile makes download_group discard the whole
                    # 64-tile group and fetch it again. Backing off with jitter
                    # lets the CDN recover, so sustained throughput stays high
                    # instead of collapsing into a retry storm.
                    tokens.refresh()
                    if attempt < max_ret:
                        time.sleep(0.5 * attempt + random.random() * 0.5)
                    continue
                if r.status_code in (401, 403):
                    tokens.refresh(); continue    # expired token -> retry at once
                if r.status_code >= 500:
                    # transient CDN error -- retry (used to give up instantly,
                    # which left permanent gray patches in the texture)
                    if attempt < max_ret:
                        time.sleep(0.4 * attempt)
                    continue
                if r.status_code != 200:
                    return coord, None      # 404 etc: tile permanently absent
                BW.add(len(r.content))
                return coord, Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception:
                if attempt < max_ret:
                    time.sleep(0.4 * attempt)
        return coord, None

    if not coords:
        return out
    if executor is not None:
        # Shared pool: submit and wait for just this batch's futures. Other
        # groups' tasks interleave in the same pool instead of each group
        # owning its own set of OS threads.
        futs = [executor.submit(fetch, c) for c in coords]
        for fut in futs:
            coord, im = fut.result()
            if im is not None:
                out[coord] = im
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(coords))) as ex:
            for coord, im in ex.map(fetch, coords):
                if im is not None:
                    out[coord] = im
    return out


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def tile_to_lonlat(x, y, zoom):
    n   = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def lonlat_to_slippy(lon, lat, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi) / 2.0 * n)
    return x, y


def slippy_range_for_bounds(lat_s, lon_w, lat_n, lon_e, zoom):
    """Slippy tile bounding box COVERING the bounds: every tile intersecting
    [lon_w,lon_e) x [lat_s,lat_n). The end tile comes from a point just
    INSIDE the east/south edge -- an edge falling mid-tile keeps that
    (partially covered) tile. The previous "x_se - 1" dropped it, and when
    that tile started a new 8-tile group the whole border group was skipped:
    verified 8-25%% of 1-deg cells had an uncovered stripe (up to ~1.8 km at
    z14; a hole in the ground in base-mesh mode)."""
    eps = 1e-9
    x_nw, y_nw = lonlat_to_slippy(lon_w, lat_n, zoom)
    x_se, y_se = lonlat_to_slippy(lon_e - eps, lat_s + eps, zoom)
    return x_nw, x_se, y_nw, y_se


def tile_range_for_degree(lat, lon, zoom):
    return slippy_range_for_bounds(lat, lon, lat + 1, lon + 1, zoom)


# ---------------------------------------------------------------------------
# Water / land classification
# ---------------------------------------------------------------------------

def classify_tile(img, threshold=0.60):
    """Water/land ratio from relative colour (blue channel clearly above red),
    not absolute brightness -- open ocean in Apple's imagery is very dark
    (RGB roughly 8,40,50) so a brightness floor like "blue > 60" misses it
    entirely; verified against real tiles (deep sea, coast, lakes, forest,
    desert, cities) before picking this threshold.
    """
    rgb = img.convert("RGB")
    if HAS_NUMPY:
        arr  = np.array(rgb, dtype=np.int16)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        mask = (b > r + 8) & (g >= r - 5)
        ratio = float(mask.sum()) / mask.size
    else:
        pixels = list(rgb.getdata())
        w, h   = rgb.size
        samp   = [pixels[yy*w+xx]
                  for yy in range(0, h, 8) for xx in range(0, w, 8)]
        ratio  = (sum(1 for (r,g,b) in samp if b > r+8 and g >= r-5)
                  / len(samp)) if samp else 0.0
    if ratio >= threshold: return "water", ratio
    if ratio >= 0.15:      return "mixed", ratio
    return "land", ratio


# ---------------------------------------------------------------------------
# DDS writer
# ---------------------------------------------------------------------------

def _dds_header(width, height, num_mipmaps):
    DDSD_CAPS=0x1; DDSD_HEIGHT=0x2; DDSD_WIDTH=0x4; DDSD_PITCH=0x8
    DDSD_PIXELFORMAT=0x1000; DDSD_MIPMAPCOUNT=0x20000
    DDSCAPS_TEXTURE=0x1000; DDSCAPS_COMPLEX=0x8; DDSCAPS_MIPMAP=0x400000
    DDPF_RGB=0x40; DDPF_ALPHAPIXELS=0x1
    flags = DDSD_CAPS|DDSD_HEIGHT|DDSD_WIDTH|DDSD_PITCH|DDSD_PIXELFORMAT
    caps  = DDSCAPS_TEXTURE
    if num_mipmaps > 1:
        flags |= DDSD_MIPMAPCOUNT
        caps  |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
    return struct.pack(
        "<4s" + "IIIIIII" + "11I" + "IIIIIIII" + "IIIII",
        b"DDS ", 124, flags, height, width, width*4, 0, num_mipmaps,
        *([0]*11),
        32, DDPF_RGB|DDPF_ALPHAPIXELS, 0, 32,
        0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000,
        caps, 0, 0, 0, 0)


def _dxt1_dds_size(px):
    """Total bytes of a complete DXT1/BC1 DDS with full mip chain (px x px)."""
    total = 128  # header
    s = px
    while True:
        blocks = ((s + 3) // 4) ** 2      # 4x4 blocks
        total += blocks * 8               # DXT1 = 8 bytes per block
        if s == 1:
            break
        s = max(1, s // 2)
    return total


def dds_is_valid(path, px):
    """True only if `path` is a COMPLETE, non-truncated DXT1 DDS of px x px.

    This is what makes resume safe: a run cancelled / killed mid-write leaves a
    partial .dds; checking only os.path.isfile() would wrongly treat it as done.
    Here we verify the magic, the dimensions, the format, and the EXACT file
    size. BC1/DXT1 is fixed-rate, so a complete texture is always byte-for-byte
    the same length (verified: texconv output == _dxt1_dds_size). Any other
    size = truncated (interrupted write) or trailing garbage = corrupt -> it
    gets re-downloaded.
    """
    try:
        if os.path.getsize(path) != _dxt1_dds_size(px):
            return False
        with open(path, "rb") as f:
            head = f.read(128)
        if head[:4] != b"DDS ":
            return False
        if struct.unpack_from("<I", head, 16)[0] != px:   # width
            return False
        if struct.unpack_from("<I", head, 12)[0] != px:   # height
            return False
        return head[84:88] == b"DXT1"
    except OSError:
        return False


def _dds_width(path):
    """Width (px) declared in a DDS header, or 0 if unreadable."""
    try:
        with open(path, "rb") as f:
            head = f.read(20)
        if head[:4] != b"DDS ":
            return 0
        return struct.unpack_from("<I", head, 16)[0]
    except OSError:
        return 0


def dds_self_valid(path):
    """Validate a DDS using the dimensions declared in its OWN header (so we
    don't need to know HD/SD up front) -- used when scanning an output folder
    to mark which cells already hold good data."""
    try:
        with open(path, "rb") as f:
            head = f.read(128)
        if head[:4] != b"DDS " or head[84:88] != b"DXT1":
            return False
        w = struct.unpack_from("<I", head, 16)[0]
        h = struct.unpack_from("<I", head, 12)[0]
        if w != h or w not in (2048, 4096):
            return False
        return os.path.getsize(path) == _dxt1_dds_size(w)
    except OSError:
        return False


def dds_gray_hole_tiles(path, px):
    """Count 'gray hole' tile regions inside a finished group texture.

    A tile whose download failed used to be pasted as uniform (60,60,60) gray
    into the stitched canvas, and older versions marked such groups complete
    anyway -- the byte-exact resume check can't see the difference. After BC1
    compression a gray region survives as blocks with color0 == color1 ~= gray,
    so this reads ONE small mip level (~32 KB regardless of texture size) and
    checks each of the 8x8 tile regions for that signature. Any hit means the
    group should be re-downloaded. Returns 0 if clean/unreadable/no numpy.
    """
    L = {4096: 4, 2048: 3}.get(px)      # mip level where the texture is 256px
    if L is None or not HAS_NUMPY:
        return 0
    off = 128
    s = px
    for _ in range(L):
        off += ((s + 3) // 4) ** 2 * 8
        s //= 2
    n = s // 4                          # BC1 blocks per side at that level
    try:
        with open(path, "rb") as f:
            f.seek(off)
            data = f.read(n * n * 8)
        if len(data) != n * n * 8:
            return 0
        u16 = np.frombuffer(data, dtype="<u2").reshape(n, n, 4)
        c0 = u16[:, :, 0].astype(np.int32)
        c1 = u16[:, :, 1].astype(np.int32)
        r = (c0 >> 11) & 31; g = (c0 >> 5) & 63; b = c0 & 31
        # RGB565 of (60,60,60) is (7,15,7); allow rounding slack. Real imagery
        # never BC1-encodes a whole tile with c0==c1 on every interior block.
        grayish = ((np.abs(r - 7) <= 1) & (np.abs(g - 15) <= 2)
                   & (np.abs(b - 7) <= 1) & (c0 == c1))
        bpt = n // GROUP_TILES          # blocks per tile region side
        holes = 0
        for j in range(GROUP_TILES):
            for i in range(GROUP_TILES):
                sub = grayish[j*bpt+1:(j+1)*bpt-1, i*bpt+1:(i+1)*bpt-1]
                if sub.size and sub.all():
                    holes += 1
        return holes
    except (OSError, ValueError):
        return 0


WATER_PLACEHOLDER_PX = 32


def write_water_dds(path, rgb):
    """Tiny solid-colour DXT1 DDS (full mip chain) for a skipped all-water
    group, written directly -- no staging BMP, no texconv. Solid colour looks
    identical at any resolution, so this is ~1 KB instead of the 11 MB (HD) a
    full-size placeholder would cost. Sized/formatted so dds_is_valid(path,
    WATER_PLACEHOLDER_PX) passes -- resume treats it as done while skip_water
    is on; with skip_water off it fails the full-size check and is redone
    with real imagery."""
    px = WATER_PLACEHOLDER_PX
    r, g, b = rgb
    c565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    block = struct.pack("<HHI", c565, c565, 0)     # c0==c1 -> solid colour
    DDSD = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000   # caps/h/w/pf/mips/linear
    mips = px.bit_length()                          # 32 -> 6 levels
    header = struct.pack(
        "<4s7I11I8I5I",
        b"DDS ", 124, DDSD, px, px, ((px + 3)//4)**2 * 8, 0, mips,
        *([0]*11),
        32, 0x4, int.from_bytes(b"DXT1", "little"), 0, 0, 0, 0, 0,
        0x1000 | 0x8 | 0x400000, 0, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(header)
        s = px
        while True:
            f.write(block * ((s + 3) // 4) ** 2)
            if s == 1:
                break
            s = max(1, s // 2)
    return dds_is_valid(path, px)


def write_dds(img, path, generate_mipmaps=True):
    # NOTE: no vertical flip here (Ortho4XP convention) -- the DSF/.ter UV
    # mapping (t=0 south / t=1 north) already accounts for north-up textures.
    # Flipping here + that UV convention would render everything upside down.
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    levels = [img]
    if generate_mipmaps:
        cur = levels[0]
        while max(cur.size) > 1:
            nw = max(1, cur.size[0]//2); nh = max(1, cur.size[1]//2)
            cur = cur.resize((nw, nh), Image.LANCZOS)
            levels.append(cur)
    w0, h0 = levels[0].size
    with open(path, "wb") as f:
        f.write(_dds_header(w0, h0, len(levels)))
        for lv in levels:
            f.write(lv.tobytes("raw", "BGRA"))


def _dds_header_dxt1(width, height, num_mipmaps):
    """DDS header for a BC1/DXT1 texture (FourCC 'DXT1'), laid out so
    dds_is_valid() accepts it and the total file matches _dxt1_dds_size()."""
    DDSD = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000   # caps/h/w/pf/mips/linsize
    caps = 0x1000                                          # DDSCAPS_TEXTURE
    if num_mipmaps > 1:
        caps |= 0x8 | 0x400000                            # COMPLEX | MIPMAP
    linsize = ((width + 3) // 4) * ((height + 3) // 4) * 8
    return struct.pack(
        "<4s7I11I8I5I",
        b"DDS ", 124, DDSD, height, width, linsize, 0, num_mipmaps,
        *([0] * 11),
        32, 0x4, int.from_bytes(b"DXT1", "little"), 0, 0, 0, 0, 0,
        caps, 0, 0, 0, 0)


def _bc1_compress_level(rgb):
    """Vectorised BC1/DXT1 encoder for ONE mip level (needs numpy).

    rgb: HxWx3 uint8 array -> block bytes. Bounding-box endpoints in 4-colour
    (opaque) mode: color0 is the channel-wise max, color1 the min, so in RGB565
    color0 >= color1 always holds and we never fall into BC1's 3-colour
    transparent mode (which would punch holes). This is the SAME 4-bit BC1
    format texconv emits -- 8x smaller than the uncompressed RGBA fallback and,
    unlike it, a texture X-Plane loads without trouble.
    """
    h, w, _ = rgb.shape
    ph = (4 - h % 4) % 4
    pw = (4 - w % 4) % 4
    if ph or pw:                                   # pad partial edge blocks
        rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
    H, W, _ = rgb.shape
    nby, nbx = H // 4, W // 4
    blocks = (rgb.reshape(nby, 4, nbx, 4, 3)
                 .transpose(0, 2, 1, 3, 4)
                 .reshape(nby, nbx, 16, 3).astype(np.int32))
    cmax = blocks.max(axis=2)
    cmin = blocks.min(axis=2)

    def to565(c):
        return (((c[..., 0] >> 3) & 0x1F) << 11) | \
               (((c[..., 1] >> 2) & 0x3F) << 5) | ((c[..., 2] >> 3) & 0x1F)

    def from565(v):
        r = (v >> 11) & 0x1F; g = (v >> 5) & 0x3F; b = v & 0x1F
        return np.stack([(r << 3) | (r >> 2),
                         (g << 2) | (g >> 4),
                         (b << 3) | (b >> 2)], axis=-1).astype(np.int32)

    c0 = to565(cmax); c1 = to565(cmin)             # c0 >= c1 guaranteed
    e0 = from565(c0); e1 = from565(c1)             # decode what the GPU sees
    palette = np.stack([e0, e1, (2 * e0 + e1) // 3, (e0 + 2 * e1) // 3], axis=2)
    diff = blocks[:, :, :, None, :] - palette[:, :, None, :, :]
    idx = (diff * diff).sum(axis=-1).argmin(axis=-1).astype(np.uint32)
    idx[c0 == c1] = 0                              # solid block -> index 0 only

    packed = np.zeros((nby, nbx), dtype=np.uint32)
    for i in range(16):
        packed |= (idx[:, :, i] & 0x3) << (2 * i)
    out = np.zeros((nby, nbx, 8), dtype=np.uint8)
    out[:, :, 0] = c0 & 0xFF;          out[:, :, 1] = (c0 >> 8) & 0xFF
    out[:, :, 2] = c1 & 0xFF;          out[:, :, 3] = (c1 >> 8) & 0xFF
    out[:, :, 4] = packed & 0xFF;      out[:, :, 5] = (packed >> 8) & 0xFF
    out[:, :, 6] = (packed >> 16) & 0xFF; out[:, :, 7] = (packed >> 24) & 0xFF
    return out.tobytes()


def write_dds_bc1(img, path):
    """Write a complete BC1/DXT1 DDS with a full mip chain, in pure Python.
    Used when texconv is unavailable: same compact format texconv produces
    (~1/8 the size of the old uncompressed fallback) so no-texconv machines
    (Linux/macOS, or Windows without the tool) still get storage-efficient,
    X-Plane-loadable textures instead of 8x-larger uncompressed ones."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    levels = [img]
    cur = img
    while max(cur.size) > 1:
        cur = cur.resize((max(1, cur.size[0] // 2), max(1, cur.size[1] // 2)),
                         Image.LANCZOS)
        levels.append(cur)
    w0, h0 = levels[0].size
    with open(path, "wb") as f:
        f.write(_dds_header_dxt1(w0, h0, len(levels)))
        for lv in levels:
            f.write(_bc1_compress_level(np.asarray(lv, dtype=np.uint8)))


# ---------------------------------------------------------------------------
# Terrain + DSF writers
# ---------------------------------------------------------------------------

def write_ter(path, base, lat_c, lon_c, size_m, pixels=4096):
    # Base-mesh terrain definition (Ortho4XP style). No PROJECTED: UVs come
    # explicitly from the DSF patch vertices.
    # ONLY safe for a FULL 1x1 deg tile run (all groups) -- a base-mesh DSF
    # (no sim/overlay) replaces terrain for the WHOLE cell; any area without
    # a patch becomes a hole (no ground at all -> aircraft falls through /
    # "spawns in the ground"). For partial/selective areas use write_pol +
    # write_dsf_overlay below instead, which drapes on top of the existing
    # terrain and never creates holes.
    with open(path, "w", encoding="ascii") as f:
        f.write("A\n800\nTERRAIN\n\n")
        f.write(f"LOAD_CENTER {lat_c:.6f} {lon_c:.6f} {size_m:.1f} {pixels}\n")
        f.write(f"BASE_TEX_NOWRAP ../textures/{base}.dds\n")
        f.write("NO_ALPHA\n")


def write_pol(path, base, lat_c, lon_c, size_m, pixels=4096):
    # Draped orthophoto polygon (official X-Plane way to overlay imagery on
    # top of EXISTING terrain, without replacing/removing it). Safe for any
    # partial area -- never creates holes. ST coords come from the DSF
    # (BEGIN_POLYGON param 65535), so SCALE is ignored.
    with open(path, "w", encoding="ascii") as f:
        f.write("A\n850\nDRAPED_POLYGON\n\n")
        f.write("LAYER_GROUP terrain 1\n")
        f.write(f"TEXTURE_NOWRAP ../textures/{base}.dds\n")
        f.write("SCALE 1.0 1.0\n")
        f.write(f"LOAD_CENTER {lat_c:.6f} {lon_c:.6f} {size_m:.1f} {pixels}\n")


def write_dsf_overlay(dsf_path, group_records, deg_lat, deg_lon):
    """Overlay DSF: draped polygons placed on top of whatever terrain already
    exists (default X-Plane mesh or another base mesh). No coverage
    requirement -- safe for any subset of the 1x1 deg cell."""
    with open(dsf_path, "w", encoding="ascii") as f:
        f.write("I\n800\nDSF\n\n")
        f.write(f"PROPERTY sim/west {deg_lon}\nPROPERTY sim/east {deg_lon+1}\n")
        f.write(f"PROPERTY sim/south {deg_lat}\nPROPERTY sim/north {deg_lat+1}\n")
        f.write("PROPERTY sim/planet earth\nPROPERTY sim/overlay 1\n")
        f.write("PROPERTY sim/creation_agent OrthoGenPy/2.0\n\n")
        for r in group_records:
            f.write(f"POLYGON_DEF terrain/{r['base']}.pol\n")
        f.write("\n")
        for idx, r in enumerate(group_records):
            lW, lN, lE, lS = r["lon_w"], r["lat_n"], r["lon_e"], r["lat_s"]
            # param 65535 = ST coords supplied per point, coord_depth 4 = lon lat s t
            f.write(f"BEGIN_POLYGON {idx} 65535 4\n")
            f.write("BEGIN_WINDING\n")
            f.write(f" POLYGON_POINT {lW:.7f} {lS:.7f} 0.0 0.0\n")  # SW
            f.write(f" POLYGON_POINT {lE:.7f} {lS:.7f} 1.0 0.0\n")  # SE
            f.write(f" POLYGON_POINT {lE:.7f} {lN:.7f} 1.0 1.0\n")  # NE
            f.write(f" POLYGON_POINT {lW:.7f} {lN:.7f} 0.0 1.0\n")  # NW
            f.write("END_WINDING\nEND_POLYGON\n")


class DemSampler:
    """Elevation from AWS terrarium DEM tiles (z12, ~24 m/px at 51°N),
    sampled with true bilinear interpolation across tile borders."""
    Z  = 12
    TS = 256          # terrarium tile size in pixels

    def __init__(self, cache_dir):
        self.cache = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._tiles = {}
        self._warned = False
        # Bare requests.get() per tile (previous behaviour) re-does the TLS
        # handshake every time -- measured ~1s/tile that way, ~halved with a
        # persistent session. Pool sized generously so prefetch() below (which
        # fetches many tiles in parallel) can actually use the concurrency
        # instead of urllib3 discarding connections past a small default pool.
        self._sess = requests.Session()
        self._sess.headers.update({"User-Agent": "Mozilla/5.0"})
        self._sess.mount("https://", requests.adapters.HTTPAdapter(
            pool_connections=32, pool_maxsize=32))

    def _fetch_tile_file(self, tx, ty):
        path = os.path.join(self.cache, f"{self.Z}_{tx}_{ty}.png")
        if not os.path.isfile(path):
            url = (f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/"
                   f"{self.Z}/{tx}/{ty}.png")
            r = self._sess.get(url, timeout=30)
            r.raise_for_status()
            # Atomic write: a run killed mid-write must never leave a truncated
            # PNG in the cache -- that used to silently turn all elevations in
            # the affected area into 0 m on every later run (verified).
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(r.content)
            os.replace(tmp, path)
        return path

    def prefetch(self, lat_s, lon_w, lat_n, lon_e, on_log=None, workers=16):
        """Download every DEM tile a full 1x1 deg cell will need, in parallel,
        before the (sequential) mesh-writing loop starts. For a full cell this
        is ~240 tiles -- fetched one at a time it's 4+ minutes; in parallel
        with a warm connection pool it's a few seconds, and scales further
        with a faster line since it's not CPU-bound."""
        x0, y0 = lonlat_to_slippy(lon_w, lat_n, self.Z)
        x1, y1 = lonlat_to_slippy(lon_e, lat_s, self.Z)
        coords = [(tx, ty) for ty in range(y0, y1 + 1) for tx in range(x0, x1 + 1)]
        t0 = time.time()

        def _one(c):
            # Best-effort: a single failed DEM tile must not abort the whole
            # cell's DSF (elevation() retries per-point and falls back itself).
            try:
                self._fetch_tile_file(*c)
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=min(workers, len(coords) or 1)) as ex:
            list(ex.map(_one, coords))
        if on_log:
            on_log(f"    DEM: {len(coords)} tiles prefetched in {time.time()-t0:.1f}s")

    def _tile(self, tx, ty):
        """Elevation array (or PIL image without numpy) for one DEM tile."""
        key = (tx, ty)
        a = self._tiles.get(key)
        if a is not None:
            return a
        path = self._fetch_tile_file(tx, ty)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Corrupt cache file (e.g. truncated by an old version's non-atomic
            # write): delete and refetch once instead of poisoning elevations.
            try: os.remove(path)
            except OSError: pass
            path = self._fetch_tile_file(tx, ty)
            img = Image.open(path).convert("RGB")
        if HAS_NUMPY:
            arr = np.asarray(img, dtype=np.float32)
            a = arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0 - 32768.0
        else:
            a = img
        self._tiles[key] = a
        return a

    def _px(self, X, Y):
        """Elevation at global DEM pixel (X, Y) -- crosses tile borders."""
        n = 2 ** self.Z
        tx = min(max(X // self.TS, 0), n - 1)
        ty = min(max(Y // self.TS, 0), n - 1)
        a  = self._tile(tx, ty)
        ix = min(max(X - tx * self.TS, 0), self.TS - 1)
        iy = min(max(Y - ty * self.TS, 0), self.TS - 1)
        if HAS_NUMPY:
            return float(a[iy, ix])
        R, G, B = a.getpixel((ix, iy))
        return R * 256.0 + G + B / 256.0 - 32768.0

    def elevation(self, lat, lon, on_log=None):
        try:
            n = 2.0 ** self.Z
            xf = (lon + 180.0) / 360.0 * n * self.TS
            r  = math.radians(max(-85.0, min(85.0, lat)))
            yf = ((1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi)
                  / 2.0 * n * self.TS)
            # pixel centres sit at +0.5, so shift before flooring
            x0 = int(math.floor(xf - 0.5)); fx = (xf - 0.5) - x0
            y0 = int(math.floor(yf - 0.5)); fy = (yf - 0.5) - y0
            e00 = self._px(x0,     y0)
            e10 = self._px(x0 + 1, y0)
            e01 = self._px(x0,     y0 + 1)
            e11 = self._px(x0 + 1, y0 + 1)
            return ((e00 * (1.0 - fx) + e10 * fx) * (1.0 - fy) +
                    (e01 * (1.0 - fx) + e11 * fx) * fy)
        except Exception as exc:
            if on_log and not self._warned:
                on_log(f"  [!] DEM fetch failed ({exc}); using 0 m for affected area.")
                self._warned = True
            return 0.0


def mesh_grid_for(group_records, deg_lat, spacing_m):
    """How many quads per group edge to hit the requested vertex spacing.

    One value for the WHOLE cell, so neighbouring groups share identical edge
    vertices (a per-group value could differ between rows and crack the mesh).
    """
    if not group_records:
        return 1
    r = group_records[0]
    lat_c = deg_lat + 0.5
    width_m = (r["lon_e"] - r["lon_w"]) * 111320.0 * math.cos(math.radians(lat_c))
    return max(1, min(160, int(round(width_m / float(spacing_m)))))


def write_dsf_basemesh(dsf_path, group_records, dem, deg_lat, deg_lon, grid, on_log):
    """Base-mesh DSF: physical terrain patches (flag=1) that REPLACE the
    default X-Plane terrain -- no sim/overlay. One BEGIN_PATCH per texture
    group, a grid x grid quad mesh, real DEM elevation, mercator-correct UV."""
    with open(dsf_path, "w", encoding="ascii") as f:
        f.write("I\n800\nDSF\n\n")
        f.write(f"PROPERTY sim/west {deg_lon}\nPROPERTY sim/east {deg_lon+1}\n")
        f.write(f"PROPERTY sim/south {deg_lat}\nPROPERTY sim/north {deg_lat+1}\n")
        f.write("PROPERTY sim/planet earth\n")
        f.write("PROPERTY sim/creation_agent OrthoGenPy/2.0\n\n")
        for r in group_records:
            f.write(f"TERRAIN_DEF terrain/{r['base']}.ter\n")
        f.write("\n")
        for idx, r in enumerate(group_records):
            lonW, lonE, latN, latS = r["lon_w"], r["lon_e"], r["lat_n"], r["lat_s"]
            cW = max(lonW, deg_lon); cE = min(lonE, deg_lon + 1)
            cS = max(latS, deg_lat); cN = min(latN, deg_lat + 1)
            if cE <= cW or cN <= cS:
                continue
            myS_full = _mercY(latS); myN_full = _mercY(latN)
            myS_c = _mercY(cS);      myN_c = _mercY(cN)

            # Precompute the vertex grid once: one DEM lookup per vertex
            # (emitting per-triangle would sample every vertex 6 times).
            rows = []
            for j in range(grid + 1):
                my  = myS_c + (myN_c - myS_c) * j / grid
                lat = _mercYInv(my)
                t   = (my - myS_full) / (myN_full - myS_full)
                row = []
                for i in range(grid + 1):
                    lon = cW + (cE - cW) * i / grid
                    s   = (lon - lonW) / (lonE - lonW)
                    e   = dem.elevation(lat, lon, on_log)
                    row.append(f"{lon:.7f} {lat:.7f} {e:.2f} 0.0 0.0 {s:.6f} {t:.6f}")
                rows.append(row)

            f.write(f"BEGIN_PATCH {idx} 0.0 -1.0 1 7\n")
            f.write("BEGIN_PRIMITIVE 0\n")
            for j in range(grid):
                for i in range(grid):
                    SW = rows[j][i];     SE = rows[j][i + 1]
                    NE = rows[j + 1][i + 1]; NW = rows[j + 1][i]
                    # CW winding (verified against X-Plane's own mesh, which is
                    # CW). CCW triangles are treated as back-facing / negative
                    # area and get silently skipped by X-Plane.
                    for v in (SW, NE, SE, SW, NW, NE):
                        f.write(f" PATCH_VERTEX {v}\n")
            f.write("END_PRIMITIVE\nEND_PATCH\n")


WATER_RGB = (7, 39, 50)   # matches Apple's real deep-ocean colour (measured)

# Skipping a group replaces it with a flat colour, so it must be used ONLY on
# groups that are (near-)entirely open water -- otherwise a small island or a
# sliver of coast inside a "mostly water" group would be flattened away. The
# user-facing water_threshold (default 0.60) is fine for painting the map, but
# for the irreversible skip we demand near-total coverage so real land detail
# is never dropped. This keeps skip-water a genuinely quality-neutral saver.
SKIP_WATER_MIN_RATIO = 0.97

def is_group_water(tokens, gx, gy, zoom, group, threshold, stop_event, session,
                   executor=None):
    """Cheap water pre-check: fetch ONE tile from `shift` zoom levels below
    (group = 2**shift), which covers exactly this group's footprint, instead
    of all group*group tiles. Returns True only if that preview is confidently
    water -- any ambiguity (mixed classification, fetch failure) returns
    False so the caller falls back to a real download instead of guessing.
    """
    shift = int(math.log2(group))
    preview_zoom = max(2, zoom - shift)
    tx = (gx * group) >> shift
    ty = (gy * group) >> shift
    tiles = download_tiles_parallel(tokens, [(tx, ty)], preview_zoom,
                                    workers=1, max_ret=2, hd=False,
                                    stop_event=stop_event, session=session,
                                    executor=executor)
    im = tiles.get((tx, ty))
    if im is None:
        return False
    cls, ratio = classify_tile(im, threshold=threshold)
    return cls == "water"


def download_group(tokens, gx, gy, zoom, group, texture_dir, base, max_ret,
                   stop_event=None, executor=None, session=None, hd=True):
    """Stage 1 (network): download group*group tiles, stitch, save a staging
    file. executor/session: shared run-wide pool (see make_tile_session);
    when given, this group's tile fetches interleave with every other
    in-flight group's fetches in the SAME pool instead of opening their own.

    hd: True = 512 px tiles (size=2&scale=2) -> 4096 px group texture. False =
    256 px tiles -> 2048 px texture: ~3.3x less to download and 4x less to
    store, at a small sharpness cost (measured: Apple's HD tile carries little
    real detail beyond 256 px). Roughly equivalent to half a zoom level.

    Returns (stage_path, got). stage_path is "SKIP" if a COMPLETE, valid .dds
    already exists (resume), or None if the whole group failed to download.
    """
    tpx = 512 if hd else 256
    tex_px = group * tpx
    dds_path = os.path.join(texture_dir, base + ".dds")
    if dds_is_valid(dds_path, tex_px):
        return "SKIP", group * group        # already done in a previous run
    if os.path.isfile(dds_path):
        # partial/corrupt leftover from an interrupted run -> discard & redo
        try: os.remove(dds_path)
        except OSError: pass
    coords = [(gx * group + i, gy * group + j)
              for j in range(group) for i in range(group)]
    tiles = download_tiles_parallel(tokens, coords, zoom, max_ret=max_ret,
                                    hd=hd, stop_event=stop_event,
                                    executor=executor, session=session)
    missing = [c for c in coords if c not in tiles]
    if missing and not (stop_event is not None and stop_event.is_set()):
        # One extra focused pass for stragglers (transient CDN hiccups).
        tiles.update(download_tiles_parallel(
            tokens, missing, zoom, max_ret=max_ret, hd=hd,
            stop_event=stop_event, executor=executor, session=session))
    if len(tiles) < len(coords):
        # NEVER bake an incomplete group: a partial texture written here would
        # pass the byte-exact resume check and keep its gray holes forever.
        # This also covers pause/close mid-group -- the group simply redoes
        # next run. Failing beats silently corrupting.
        return None, len(tiles)
    canvas = Image.new("RGB", (group * tpx, group * tpx), (60, 60, 60))
    got = 0
    for j in range(group):
        for i in range(group):
            im = tiles.get((gx * group + i, gy * group + j))
            if im is None:
                continue
            if im.size != (tpx, tpx):
                im = im.resize((tpx, tpx), Image.LANCZOS)
            canvas.paste(im, (i * tpx, j * tpx))
            got += 1
    # BMP, not PNG: measured on real imagery, PNG's deflate encode costs
    # ~2.2s for a 4096 canvas -- MORE than texconv's own BC1 compression
    # (~0.7-0.9s). BMP is uncompressed (~0.03s to write) and texconv reads it
    # directly, so the data isn't compressed twice for no reason.
    stage_path = os.path.join(texture_dir, base + ".bmp")
    canvas.save(stage_path)
    return stage_path, got


def compress_group(stage_path, dds_path, tex_px, on_log):
    """Stage 2 (CPU): texconv the staged image into a BC1 DDS with mips.
    Removes the staging file.

    Kept separate from the download so the two run on different resource pools
    and the network never idles while textures compress.
    """
    texconv = _find_texconv()
    try:
        if texconv:
            subprocess.run([texconv, "-f", "BC1_UNORM", "-m", "0", "-y",
                            "-o", os.path.dirname(dds_path), stage_path],
                           capture_output=True, text=True, timeout=180)
            # isfile() alone let a texconv crash/timeout that left a partial
            # .dds count as success -- the DSF then referenced a broken
            # texture. Full byte-exact validation is cheap here (one run per
            # group) and anything invalid simply reruns next time.
            return dds_is_valid(dds_path, tex_px)
        if HAS_NUMPY:
            # No texconv -> compress to the SAME BC1/DXT1 format in pure Python.
            # ~1/8 the size of an uncompressed DDS and, unlike it, one X-Plane
            # loads reliably. Slower than texconv but only used when it's absent.
            write_dds_bc1(Image.open(stage_path).convert("RGB"), dds_path)
            return dds_is_valid(dds_path, tex_px)
        on_log("  [!] texconv not found and numpy missing — writing uncompressed "
               "DDS (8x larger, may crash X-Plane). Install numpy or texconv.")
        write_dds(Image.open(stage_path).convert("RGBA"), dds_path,
                  generate_mipmaps=True)
        return os.path.isfile(dds_path)
    finally:
        try:
            os.remove(stage_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

_SBASE = 20  # 1 unit = 1/20 ° = 0.05 °

# Generation stitches GROUP_TILES x GROUP_TILES web tiles (512px each) into one
# 4096px texture per group.
GROUP_TILES = 8

# Base-mesh terrain detail: target distance between mesh vertices, in metres.
# This is INDEPENDENT of the texture zoom -- the grid per group is derived from
# it, so texture sharpness and terrain relief are separate knobs.
# The DEM itself is ~24 m/px, so anything down to ~50 m still adds real detail.
MESH_SPACING_M = 200


def _slippy_in_sel(tx, ty, zoom, sel_set):
    """True if slippy tile (tx,ty) overlaps at least one selected base cell."""
    lon_w, lat_n = tile_to_lonlat(tx,   ty,   zoom)
    lon_e, lat_s = tile_to_lonlat(tx+1, ty+1, zoom)
    la_s = int(math.floor(lat_s * _SBASE)) - 1
    la_n = int(math.ceil (lat_n * _SBASE)) + 1
    lo_w = int(math.floor(lon_w * _SBASE)) - 1
    lo_e = int(math.ceil (lon_e * _SBASE)) + 1
    for la in range(la_s, la_n + 1):
        for lo in range(lo_w, lo_e + 1):
            if (la, lo) in sel_set:
                return True
    return False


def _group_in_sel(gx, gy, zoom, sel_set, group=GROUP_TILES):
    """True if the group's web-tile block overlaps at least one selected cell."""
    lon_w, lat_n = tile_to_lonlat(gx * group,       gy * group,       zoom)
    lon_e, lat_s = tile_to_lonlat((gx + 1) * group, (gy + 1) * group, zoom)
    la_s = int(math.floor(lat_s * _SBASE)) - 1
    la_n = int(math.ceil (lat_n * _SBASE)) + 1
    lo_w = int(math.floor(lon_w * _SBASE)) - 1
    lo_e = int(math.ceil (lon_e * _SBASE)) + 1
    for la in range(la_s, la_n + 1):
        for lo in range(lo_w, lo_e + 1):
            if (la, lo) in sel_set:
                return True
    return False


def run_generation(cfg, on_progress, on_log, stop_event):
    """
    Draped-overlay generator: stitches GROUP_TILES x GROUP_TILES web tiles
    into one texture per group and writes an overlay DSF (sim/overlay 1) that
    drapes those textures over X-Plane's EXISTING terrain mesh.

    Elevation comes from X-Plane's own mesh -- the imagery follows whatever
    hills/valleys are already there. We do not inject our own DEM, so any
    partial selection is safe (a draped overlay can never leave a hole).

    The base-mesh path (write_ter + write_dsf_basemesh + DemSampler) is kept
    for a possible future FULL-1x1-deg-tile mode; it injects real DEM
    elevation but replaces the terrain, so it must cover the whole cell.

    cfg keys:
        sel_base_cells  – frozenset of (la_u, lo_u) at _SBASE resolution
        zoom            – int 14-19
        max_retries     – int
        output_dir      – str
        skip_water      – bool: pre-check each group with ONE low-zoom preview
                           tile before downloading it in full; a group that's
                           entirely open water gets a solid-colour placeholder
                           instead (overlay mode only -- see do_download).
        water_threshold – float 0-1, fraction of blue pixels in the preview
                           tile above which a group counts as "water".
    """
    sel_set  = cfg["sel_base_cells"]   # frozenset of (la_u, lo_u)
    zoom     = cfg["zoom"]
    max_ret  = cfg["max_retries"]
    out_base = cfg["output_dir"]
    base_mesh = cfg.get("base_mesh", False)
    skip_water      = cfg.get("skip_water", False)
    water_threshold = float(cfg.get("water_threshold", 0.60))
    spacing_m = int(cfg.get("mesh_spacing_m", MESH_SPACING_M))
    hd        = cfg.get("hd", True)

    # Group base cells by the 1°×1° DSF tile they belong to
    dsf_map = defaultdict(list)          # (deg_lat, deg_lon) → [base_cells]
    for (la_u, lo_u) in sel_set:
        deg_lat = int(math.floor(la_u / _SBASE))
        deg_lon = int(math.floor(lo_u / _SBASE))
        dsf_map[(deg_lat, deg_lon)].append((la_u, lo_u))

    mode = "BASE MESH (3D, replaces terrain)" if base_mesh else "draped overlay"
    on_log(f"Computing scope … ({len(sel_set)} base cells, "
           f"{len(dsf_map)} DSF tile(s), zoom {zoom}, mode: {mode})")
    dsf_order = sorted(dsf_map.keys())

    # Return value: True = ran to completion (or legitimately nothing to do),
    # False/None = aborted early. The GUI used to treat EVERY non-exception
    # end as success -- a failed token fetch showed "Done!" plus the success
    # popup and marked the session finished.
    if base_mesh:
        # SAFETY: a base-mesh DSF replaces terrain for the entire 1x1 deg
        # cell. Every cell must be FULLY selected (all _SBASE^2 base cells),
        # otherwise unselected areas become holes in the ground.
        for (deg_lat, deg_lon) in dsf_order:
            have = len(dsf_map[(deg_lat, deg_lon)])
            need = _SBASE * _SBASE
            if have < need:
                on_log(f"[-] ABORT: tile {deg_lat:+d}{deg_lon:+04d} only has "
                       f"{have}/{need} base cells selected. Base-mesh mode "
                       f"requires FULL 1x1 deg tiles (select whole tiles in "
                       f"the map; the grid snaps to 1 deg in this mode).")
                return False

    # Pre-compute, for every DSF cell, the group-aligned list of groups that
    # intersect the selection (group range = tile range // GROUP_TILES).
    cell_groups = {}
    total_groups = 0
    for (deg_lat, deg_lon) in dsf_order:
        cells = dsf_map[(deg_lat, deg_lon)]
        la_min = min(c[0] for c in cells); la_max = max(c[0] for c in cells) + 1
        lo_min = min(c[1] for c in cells); lo_max = max(c[1] for c in cells) + 1
        lat_s = la_min / _SBASE; lat_n = la_max / _SBASE
        lon_w = lo_min / _SBASE; lon_e = lo_max / _SBASE
        xs, xe, ys, ye = slippy_range_for_bounds(lat_s, lon_w, lat_n, lon_e, zoom)
        gxs, gxe = xs // GROUP_TILES, xe // GROUP_TILES
        gys, gye = ys // GROUP_TILES, ye // GROUP_TILES
        groups = [(gx, gy) for gy in range(gys, gye + 1)
                            for gx in range(gxs, gxe + 1)
                            if _group_in_sel(gx, gy, zoom, sel_set)]
        cell_groups[(deg_lat, deg_lon)] = groups
        total_groups += len(groups)

    on_log(f"  → {total_groups:,} texture groups "
           f"({total_groups * GROUP_TILES * GROUP_TILES:,} web tiles) to process.")

    if total_groups == 0:
        on_log("Nothing to do."); return True

    tokens = AppleTokenService()
    try:
        on_log("[*] Fetching Apple Maps token …")
        tokens.refresh()
        on_log("[+] Token OK.")
    except Exception as exc:
        on_log(f"[-] Token failed: {exc}"); return False

    if not _find_texconv():
        if HAS_NUMPY:
            on_log("[i] texconv not found — using the built-in BC1 compressor "
                   "instead (same compact DXT1 size, slightly lower quality). "
                   "For best quality get texconv from "
                   "https://github.com/microsoft/DirectXTex/releases")
        else:
            on_log("[!] texconv not found AND numpy missing — textures will be "
                   "uncompressed and ~8x larger. Install numpy, or get texconv "
                   "from https://github.com/microsoft/DirectXTex/releases")

    dem = DemSampler(os.path.join(out_base, "dem_cache")) if base_mesh else None
    dl_workers = max(1, int(cfg.get("group_workers", 6)))
    # Was hard-capped at 4, which becomes the bottleneck once downloads are no
    # longer the limiting factor (fast line): measured, texconv scales fine
    # well past 4 parallel invocations. Scale with actual cores instead, so
    # a bigger machine or a faster line both translate into real speedup.
    cpu_workers = max(2, (os.cpu_count() or 4) - 2)

    # ONE shared tile-fetch pool + Session for the whole run, instead of every
    # group opening its own. Measured: nested pools (group_workers separate
    # pools of TILE_WORKERS threads each) actively get SLOWER at higher
    # group_workers (~400 OS threads at 24x16) due to Windows scheduling
    # overhead, while a single flat pool kept scaling cleanly past 300
    # concurrent fetches. This decouples "groups being staged" (dl_workers,
    # bounded by local disk/memory) from "raw connections open" (this pool),
    # so a faster line has one clean knob with headroom instead of a
    # multiplying one.
    tile_pool_size = max(32, min(200, dl_workers * 12))
    tile_session  = make_tile_session(tile_pool_size)
    tile_executor = ThreadPoolExecutor(max_workers=tile_pool_size)
    # ONE compress pool for the whole run (was per-DSF-cell); shut down in
    # the finally below together with the tile pool so nothing ever leaks.
    compress_ex = ThreadPoolExecutor(max_workers=cpu_workers)
    on_log(f"[*] Pipeline: {dl_workers} groups staged concurrently, "
           f"{tile_pool_size} shared tile connections → {cpu_workers} texconv workers")
    prog = {"done": 0}
    prog_lock = threading.Lock()

    def bump_progress(tile_name, gx, gy):
        with prog_lock:
            prog["done"] += 1
            d = prog["done"]
        on_progress(d, total_groups, f"{tile_name}  group {gx},{gy}")

    try:
        for (deg_lat, deg_lon) in dsf_order:
            if stop_event.is_set():
                on_log("[!] Cancelled."); return False

            slat = f"+{deg_lat:02d}" if deg_lat >= 0 else f"{deg_lat:03d}"
            slon = f"+{deg_lon:03d}" if deg_lon >= 0 else f"{deg_lon:04d}"
            tile_name = f"{slat}{slon}"
            groups = cell_groups[(deg_lat, deg_lon)]
            on_log(f"\n--- DSF tile {tile_name}: {len(groups)} groups ---")

            lat_dir = (deg_lat // 10) * 10
            lon_dir = (deg_lon // 10) * 10
            s_lat10 = f"+{lat_dir:02d}" if lat_dir >= 0 else f"{lat_dir:03d}"
            s_lon10 = f"+{lon_dir:03d}" if lon_dir >= 0 else f"{lon_dir:04d}"
            scenery_dir = os.path.join(out_base, f"zOrtho4XP_{tile_name}")
            nav_dir     = os.path.join(scenery_dir, "Earth nav data",
                                       f"{s_lat10}{s_lon10}")
            terrain_dir = os.path.join(scenery_dir, "terrain")
            texture_dir = os.path.join(scenery_dir, "textures")
            for d in (nav_dir, terrain_dir, texture_dir):
                os.makedirs(d, exist_ok=True)

            # --- Resume scan: report what's already valid and clean partial junk.
            # Also HEALS textures from older versions: a group that baked failed
            # tiles as gray patches looks byte-valid, so check its content too
            # and re-download any group with gray holes.
            tex_px_scan = GROUP_TILES * (512 if hd else 256)
            already = healed = 0
            for (gx, gy) in groups:
                p = os.path.join(texture_dir, f"{tile_name}_z{zoom}_g{gx}_{gy}.dds")
                if dds_is_valid(p, tex_px_scan):
                    if dds_gray_hole_tiles(p, tex_px_scan):
                        try:
                            os.remove(p); healed += 1
                        except OSError:
                            already += 1
                    else:
                        already += 1
                elif (skip_water and not base_mesh
                      and dds_is_valid(p, WATER_PLACEHOLDER_PX)):
                    already += 1               # water placeholder counts as done
            # Remove leftover staging (.bmp) and any 0-byte/partial .dds from a run
            # that was cancelled or closed mid-write, so they get cleanly redone.
            cleaned = 0
            for fn in os.listdir(texture_dir):
                fp = os.path.join(texture_dir, fn)
                if fn.lower().endswith(".bmp"):
                    try: os.remove(fp); cleaned += 1
                    except OSError: pass
            if already or healed:
                on_log(f"    Resume: {already}/{len(groups)} groups already done & "
                       f"valid — {len(groups)-already} left"
                       + (f" (cleaned {cleaned} partial files)" if cleaned else "")
                       + (f" — {healed} mit grauen Fehlstellen erkannt → werden "
                          f"neu geladen" if healed else ""))

            abort_evt = threading.Event()
            results = {}
            res_lock = threading.Lock()
            # Backpressure: cap PNGs waiting on disk so downloads don't run far
            # ahead of compression (each 4096 PNG is ~30-40 MB).
            inflight = threading.Semaphore(dl_workers + cpu_workers + 2)

            def meta_for(gxy):
                gx, gy = gxy
                base = f"{tile_name}_z{zoom}_g{gx}_{gy}"
                lonW, latN = tile_to_lonlat(gx * GROUP_TILES,     gy * GROUP_TILES,     zoom)
                lonE, latS = tile_to_lonlat((gx+1) * GROUP_TILES, (gy+1) * GROUP_TILES, zoom)
                lat_c = (latN + latS) / 2; lon_c = (lonW + lonE) / 2
                size_m = (lonE - lonW) * 111320.0 * math.cos(math.radians(lat_c))
                return dict(gx=gx, gy=gy, base=base, lonW=lonW, lonE=lonE,
                            latN=latN, latS=latS, lat_c=lat_c, lon_c=lon_c, size_m=size_m)

            tex_px = GROUP_TILES * (512 if hd else 256)   # group texture resolution

            def finish(m, ok):
                if ok:
                    if base_mesh:
                        write_ter(os.path.join(terrain_dir, m["base"] + ".ter"),
                                  m["base"], m["lat_c"], m["lon_c"], m["size_m"], tex_px)
                    else:
                        write_pol(os.path.join(terrain_dir, m["base"] + ".pol"),
                                  m["base"], m["lat_c"], m["lon_c"], m["size_m"], tex_px)
                    with res_lock:
                        results[(m["gx"], m["gy"])] = {
                            "base": m["base"], "lon_w": m["lonW"], "lon_e": m["lonE"],
                            "lat_n": m["latN"], "lat_s": m["latS"]}
                else:
                    on_log(f"  [!] group {m['gx']},{m['gy']}: texture failed")
                    if base_mesh:
                        abort_evt.set()
                bump_progress(tile_name, m["gx"], m["gy"])

            comp_futs = []

            def do_compress(m, stage):
                try:
                    dds = os.path.join(texture_dir, m["base"] + ".dds")
                    ok = compress_group(stage, dds, tex_px, on_log)
                    finish(m, ok)
                except Exception as exc:
                    # Must never propagate: an uncaught error here used to
                    # surface at cf.result(), kill the whole run and leave the
                    # GUI frozen with a dead Start button.
                    on_log(f"  [!] compress {m['gx']},{m['gy']}: {exc}")
                    try: finish(m, False)
                    except Exception: pass
                finally:
                    inflight.release()

            # Stage 1 (network): download + stitch + save staging file, then hand
            # off to the texconv pool. Downloads keep flowing while textures compress.
            def do_download(gxy):
                if stop_event.is_set() or abort_evt.is_set():
                    return
                m = meta_for(gxy)
                inflight.acquire()
                released = False
                try:
                    dds_path = os.path.join(texture_dir, m["base"] + ".dds")
                    if not base_mesh and skip_water and not dds_is_valid(dds_path, tex_px):
                        if dds_is_valid(dds_path, WATER_PLACEHOLDER_PX):
                            released = True; inflight.release()
                            finish(m, True); return     # water placeholder from earlier run
                        try:
                            skip_thr = max(water_threshold, SKIP_WATER_MIN_RATIO)
                            is_water = is_group_water(tokens, m["gx"], m["gy"], zoom,
                                                      GROUP_TILES, skip_thr,
                                                      stop_event, tile_session,
                                                      executor=tile_executor)
                        except Exception:
                            is_water = False   # preview failed -> fall back to a real download
                        if is_water:
                            ok = write_water_dds(dds_path, WATER_RGB)
                            on_log(f"  [~] group {m['gx']},{m['gy']}: water — skipped "
                                   f"(1 preview tile, ~1 KB placeholder)")
                            released = True; inflight.release()
                            finish(m, ok); return
                    stage, got = download_group(tokens, m["gx"], m["gy"], zoom,
                                                GROUP_TILES, texture_dir, m["base"],
                                                max_ret, stop_event,
                                                executor=tile_executor, session=tile_session,
                                                hd=hd)
                    if stage == "SKIP":                 # already built (resume)
                        released = True; inflight.release()
                        finish(m, True); return
                    if stage is None:                   # incomplete/failed download
                        released = True; inflight.release()
                        if not stop_event.is_set():     # pause mid-group is not an error
                            finish(m, False)
                        return
                    comp_futs.append(compress_ex.submit(do_compress, m, stage))
                    released = True    # do_compress owns the semaphore slot from here
                except Exception as exc:
                    on_log(f"  [!] group {m['gx']},{m['gy']}: {exc}")
                    if not released:
                        inflight.release()
                        try: finish(m, False)
                        except Exception: pass

            with ThreadPoolExecutor(max_workers=dl_workers) as dex:
                list(dex.map(do_download, groups))     # waits for all downloads
            for cf in as_completed(list(comp_futs)):   # then drain compression
                try:
                    cf.result()
                except Exception as exc:               # belt & braces
                    on_log(f"  [!] compress worker: {exc}")

            if stop_event.is_set():
                on_log("[!] Cancelled."); return False
            if base_mesh and abort_evt.is_set():
                on_log(f"  [-] ABORT: base mesh cannot have missing groups (would "
                       f"leave a hole). Re-run to retry (finished textures are kept).")
                return False

            group_records = [results[g] for g in groups if g in results]
            if group_records:
                txt_path = os.path.join(nav_dir, f"{tile_name}.txt")
                dsf_path = os.path.join(nav_dir, f"{tile_name}.dsf")
                # Post-100% finalize phases: report via on_progress(-1, …) so the
                # GUI shows a live "still working" status + pulsing bar instead of
                # a frozen 100%. These phases (esp. base-mesh DEM+mesh+compile) can
                # take tens of seconds with no per-group progress to count.
                try:
                    if base_mesh:
                        grid = mesh_grid_for(group_records, deg_lat, spacing_m)
                        tris = len(group_records) * grid * grid * 2
                        on_log(f"    Mesh: {grid}x{grid} quads/group → {tris:,} triangles "
                               f"(~{spacing_m} m vertex spacing)")
                        on_progress(-1, 0, f"{tile_name}: Höhendaten (DEM) laden …")
                        dem.prefetch(deg_lat, deg_lon, deg_lat + 1, deg_lon + 1, on_log)
                        on_progress(-1, 0, f"{tile_name}: 3D-Mesh + Höhen schreiben "
                                           f"({tris:,} Dreiecke) …")
                        on_log(f"    Sampling DEM + writing mesh (this takes a while) …")
                        write_dsf_basemesh(txt_path, group_records, dem,
                                            deg_lat, deg_lon, grid, on_log)
                    else:
                        on_progress(-1, 0, f"{tile_name}: DSF zusammensetzen "
                                           f"({len(group_records):,} Kacheln) …")
                        write_dsf_overlay(txt_path, group_records, deg_lat, deg_lon)
                except Exception as exc:
                    on_log(f"[!] DSF write failed for {tile_name}: {exc}")
                    continue

                on_log(f"[+] {tile_name}: {len(group_records)} groups written")

                dsftool = _find_dsftool(out_base)
                if dsftool:
                    on_progress(-1, 0, f"{tile_name}: DSF kompilieren (DSFTool) …")
                    on_log(f"    Compiling DSF with DSFTool …")
                    try:
                        r = subprocess.run(
                            [dsftool, "--text2dsf", txt_path, dsf_path],
                            capture_output=True, text=True, timeout=600)
                        if r.returncode == 0:
                            os.remove(txt_path)
                            on_log(f"    DSF compiled OK ({os.path.getsize(dsf_path)//1024} KB)")
                        else:
                            on_log(f"    DSFTool error (code {r.returncode}): {r.stderr[:200]}")
                            on_log(f"    Text DSF kept at: {txt_path}")
                    except Exception as exc:
                        on_log(f"    DSFTool failed: {exc}")
                        on_log(f"    Text DSF kept at: {txt_path}")
                else:
                    on_log(f"    No DSFTool found — text DSF saved.")
                    on_log(f"    To compile:  DSFTool --text2dsf \"{txt_path}\" \"{dsf_path}\"")
                    on_log(f"    DSFTool.exe is in your Downloads folder.")
            else:
                on_log(f"[!] {tile_name}: no groups succeeded")

    finally:
        # Always shut the shared pools down -- even when a DSF cell
        # raises unexpectedly -- so no threads outlive the run.
        compress_ex.shutdown(wait=True)
        tile_executor.shutdown(wait=True)
    on_log("\n[+] Generation complete.")
    return True


def _find_dsftool(out_base):
    """Return path to DSFTool.exe if found, else None."""
    candidates = [
        os.path.join(os.path.dirname(out_base), "DSFTool.exe"),
        os.path.join(out_base, "DSFTool.exe"),
        os.path.expanduser(r"~\Downloads\DSFTool.exe"),
        os.path.expanduser(r"~\Desktop\DSFTool.exe"),
        r"C:\Program Files (x86)\Steam\steamapps\common\X-Plane 12\tools\DSFTool.exe",
        r"C:\Program Files\X-Plane 12\tools\DSFTool.exe",
    ]
    return next((p for p in candidates if os.path.isfile(p)), None)


def _find_texconv():
    """Return path to texconv (Microsoft DirectXTex) if found, else None.
    Checks the usual Windows drop spots plus PATH / a bare `texconv` binary,
    so a non-Windows build gets real texconv compression when it's installed."""
    candidates = [
        os.path.expanduser(r"~\Downloads\texconv.exe"),
        os.path.expanduser(r"~\Desktop\texconv.exe"),
        r"C:\tools\texconv.exe",
    ]
    hit = next((p for p in candidates if os.path.isfile(p)), None)
    return hit or shutil.which("texconv") or shutil.which("texconv.exe")


# ===========================================================================
#  MERCATOR HELPERS
# ===========================================================================

def _mercY(lat):
    r = math.radians(max(-85.05, min(85.05, lat)))
    return math.log(math.tan(math.pi / 4 + r / 2))


def _mercYInv(my):
    return math.degrees(2.0 * math.atan(math.exp(my)) - math.pi / 2)


# ===========================================================================
#  CITIES
# ===========================================================================

CITIES = [
    ("London",       51.5,  -0.1), ("Paris",      48.9,   2.3),
    ("Berlin",       52.5,  13.4), ("Hamburg",    53.6,  10.0),
    ("Munich",       48.1,  11.6), ("Vienna",     48.2,  16.4),
    ("Rome",         41.9,  12.5), ("Madrid",     40.4,  -3.7),
    ("Warsaw",       52.2,  21.0), ("Amsterdam",  52.4,   4.9),
    ("Prague",       50.1,  14.5), ("Brussels",   50.8,   4.4),
    ("Oslo",         59.9,  10.7), ("Stockholm",  59.3,  18.1),
    ("Budapest",     47.5,  19.0), ("Zurich",     47.4,   8.5),
    ("Copenhagen",   55.7,  12.6), ("Helsinki",   60.2,  25.0),
    ("Athens",       37.9,  23.7), ("Lisbon",     38.7,  -9.1),
    ("Aschersleben", 51.75, 11.46),
]


# ===========================================================================
#  GUI
# ===========================================================================

class OrthoApp:
    C_BG    = "#1b2438"
    C_PANEL = "#232d42"
    C_HDR   = "#1a6b28"
    C_ACC   = "#4af078"
    C_FG    = "#dce8f0"
    C_DIM   = "#7888a0"
    C_SEL   = "#ffffff"   # current selection
    C_DONE  = "#20d24e"   # fully downloaded (every group of the cell is on disk)
    C_TODO  = "#ffd11a"   # started but INCOMPLETE (paused / aborted run)
    C_BTN   = "#1e7030"
    C_RED   = "#c83030"

    _DV = dict(lat_min=34.0, lat_max=72.0, lon_min=-12.0, lon_max=48.0)

    # Selection base resolution: 1 unit = 0.05° (= 1/_SBASE degrees)
    _SBASE = _SBASE

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Ortho4XP-style Scenery Generator  –  X-Plane 12")
        root.configure(bg=self.C_BG)
        root.geometry("1320x760")
        root.minsize(960, 600)

        self._v  = dict(self._DV)
        # _sel: set of (la_u, lo_u) integers at _SBASE resolution (0.05°)
        self._sel: set = set()
        # _hov: coarse cell key (la_u, lo_u) aligned to current _cell_deg()
        self._hov = None
        self._drag_start = None   # coarse cell key where drag began
        self._drag_mode  = None   # "add" | "remove"
        self._pan_last   = None
        self._rdrag_origin = None  # (x, y) where right button went down
        self._rdrag_is_pan = False # right-drag exceeded click threshold -> panning
        # A base cell is DONE only when every texture group covering it exists
        # on disk. Cells that hold some but not all of their groups are TODO
        # (an interrupted run) -- they are what "resume" still has to fetch.
        self._done_base = set()    # base cells fully downloaded          (green)
        self._todo_base = set()    # base cells started but incomplete    (yellow)
        self._done_info = {}       # base cell -> (zoom, px) of existing data
        self._todo_cfg  = None     # (zoom, px) the unfinished data was made with
        self._todo_flags = None    # base_mesh/skip_water of the unfinished run
        self._done_coarse = None   # (cd_u, {done,todo}) cache -- see _coarse_has
        self._scan_ran  = False    # first scan restores the interrupted settings
        self._bg_draw_key = None   # cache key of the last _draw_bg crop/resize
        self._bg_off = (0, 0)
        self._running = False      # a generation run is active
        self._cw, self._ch = 900, 600

        # Satellite background — dynamic, resolution follows the map zoom
        self._bg_mosaic  = None   # PIL Image covering the current view
        self._bg_photo   = None   # ImageTk.PhotoImage (keep ref to avoid GC)
        self._bg_bounds  = None   # (lon_w, lat_s, lon_e, lat_n) of the mosaic
        self._bg_zoom    = None   # slippy tile zoom of the current mosaic
        self._bg_loading = False
        self._bg_error   = None   # str, shown on the canvas when load failed
        self._bg_tokens  = None   # shared AppleTokenService
        self._bg_cache   = {}     # (z,x,y) -> PIL tile, reused across pans/zooms
        self._bg_gen     = 0      # request generation; stale fetches are dropped
        self._bg_after   = None   # pending debounce timer id

        # Generation
        self._stop_event = None   # threading.Event
        self._gen_start  = None   # float timestamp
        self._pb_pulsing = False  # progress bar in indeterminate finalize mode

        self._build_ui()
        # Load the background at the resolution that matches the initial view;
        # zooming/panning then refetches sharper tiles automatically.
        root.after(300, lambda: self._refresh_bg(force=True))
        self._scan_after = None
        root.after(400, self._scan_existing)   # mark already-downloaded cells
        root.after(80,  self._redraw)
        # Live bandwidth readout
        self._bw_last_t = None
        self._bw_last_b = 0
        self._bw_peak   = 0.0
        self._bw_hist   = []      # rolling window of instantaneous Mbit/s
        self._bw_after  = root.after(700, self._poll_bw)
        # Closing via the window's X mid-run: signal the worker to stop so it
        # exits cleanly. Everything already downloaded stays on disk; on the
        # next run the resume scan re-validates it and continues where it
        # stopped, so closing anytime is safe.
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        if self._running:
            # Closing mid-run: keep the run's frozen scope so the next launch
            # marks the unfinished area yellow and can continue it.
            self._session_save("paused", getattr(self, "_run_snap", None))
        if self._stop_event is not None and not self._stop_event.is_set():
            self._stop_event.set()          # tell the generation thread to stop
        # Drop pending timers first: one firing into a half-destroyed window
        # prints a Tcl "invalid command name" on every exit.
        for attr in ("_bw_after", "_bg_after", "_scan_after"):
            aid = getattr(self, attr, None)
            if aid is not None:
                try: self.root.after_cancel(aid)
                except Exception: pass
                setattr(self, attr, None)
        self.root.destroy()

    # ------------------------------------------------------------------
    # Mercator canvas helpers
    # ------------------------------------------------------------------

    def _merc_n(self): return _mercY(self._v["lat_max"])
    def _merc_s(self): return _mercY(self._v["lat_min"])

    def _geo_to_xy(self, lat, lon):
        """Geographic point → canvas pixel (x, y). Y=0 is north."""
        v = self._v
        my_n = self._merc_n(); my_s = self._merc_s()
        x = (lon - v["lon_min"]) / (v["lon_max"] - v["lon_min"]) * self._cw
        y = (my_n - _mercY(lat)) / (my_n - my_s) * self._ch
        return x, y

    def _xy_to_geo(self, cx, cy):
        """Canvas pixel → geographic (lat, lon)."""
        v = self._v
        lon = cx / self._cw * (v["lon_max"] - v["lon_min"]) + v["lon_min"]
        my_n = self._merc_n(); my_s = self._merc_s()
        my   = my_n - cy / self._ch * (my_n - my_s)
        return _mercYInv(my), lon

    def _lon_per_px(self):
        v = self._v
        return (v["lon_max"] - v["lon_min"]) / self._cw

    def _enforce_aspect(self):
        """Prevent horizontal/vertical stretch: make the view's lon-span vs
        mercator lat-span match the canvas pixel aspect (square tiles stay
        square). Longitude span is authoritative; latitude is derived."""
        v = self._v
        if self._cw < 2 or self._ch < 2:
            return
        lon_span = v["lon_max"] - v["lon_min"]
        my_c = (self._merc_n() + self._merc_s()) / 2.0
        # x uses degrees of lon, y uses mercator radians -> convert with radians()
        my_span = math.radians(lon_span) * self._ch / self._cw
        top, bot = _mercY(85.0), _mercY(-85.0)
        my_n = min(top, my_c + my_span / 2.0)
        my_s = max(bot, my_c - my_span / 2.0)
        v["lat_max"] = _mercYInv(my_n)
        v["lat_min"] = _mercYInv(my_s)

    # ------------------------------------------------------------------
    # Adaptive grid
    # ------------------------------------------------------------------

    def _cell_deg(self):
        """Current grid cell size in degrees, based on how far in we're zoomed."""
        # Base-mesh mode replaces terrain for the WHOLE 1x1 deg cell -- partial
        # coverage would leave holes in the ground. So selection granularity is
        # locked to full 1-degree tiles in that mode.
        if getattr(self, "_mesh_var", None) is not None and self._mesh_var.get():
            return 1.0
        # Fixed cell size selected in the UI -> don't shrink with zoom.
        csv = getattr(self, "_cellsize_var", None)
        if csv is not None and csv.get() != "Auto":
            try:
                return float(csv.get())
            except ValueError:
                pass
        lon_span = self._v["lon_max"] - self._v["lon_min"]
        if lon_span > 50:  return 2.0
        if lon_span > 25:  return 1.0
        if lon_span > 12:  return 0.5
        if lon_span > 6:   return 0.25
        if lon_span > 3:   return 0.1
        return 0.05

    def _on_mesh_toggle(self):
        """Base-mesh mode toggled: selection granularity changes to full 1-deg
        tiles, so any existing partial selection would be invalid -- clear it."""
        self._sel.clear()
        on = self._mesh_var.get()
        self._spacing_combo.config(state="readonly" if on else "disabled")
        # In base-mesh mode the cell is always a full 1°, so the cell-size
        # selector has no effect -> grey it out.
        if hasattr(self, "_cellsize_combo"):
            self._cellsize_combo.config(state="disabled" if on else "readonly")
        self._update_mesh_info()
        self._update_sel_info()
        self._redraw()

    def _update_mesh_info(self):
        """Show what the chosen vertex spacing costs in triangles."""
        if not self._mesh_var.get():
            self._mesh_info.config(text="Overlay mode: imagery drapes over "
                                        "X-Plane's own 3D terrain.", fg="#88aacc")
            return
        try:
            s = max(20, int(self._spacing_var.get()))
        except (tk.TclError, ValueError):
            return
        n_tiles = max(1, len(self._sel) // (self._SBASE * self._SBASE))
        lat_c = 51.5
        if self._sel:
            lat_c = (min(k[0] for k in self._sel) / self._SBASE) + 0.5
        cw = 111320.0 * math.cos(math.radians(lat_c))
        tris = int(2 * (cw / s) * (111320.0 / s)) * n_tiles
        warn = tris > 1_200_000
        self._mesh_info.config(
            text=(f"≈{tris:,} triangles"
                  + (f" for {n_tiles} tiles" if n_tiles > 1 else "")
                  + f"  (X-Plane's own: ~242,000)"
                  + ("\n⚠ very heavy — may hurt FPS / DSF build time" if warn else "")),
            fg="#ff9955" if warn else "#88aacc")

    def _update_hd_info(self):
        """Explain, for the CURRENT zoom + selection latitude, what the HD/SD
        switch actually buys: effective ground resolution and Apple's native
        detail limit (~30-50 cm/px), so the trade-off is concrete not abstract."""
        if not hasattr(self, "_hd_info"):
            return
        try:
            z = int(self._zoom_var.get())
        except (tk.TclError, ValueError):
            return
        lat_c = 51.0
        if self._sel:
            lat_c = (min(k[0] for k in self._sel) / self._SBASE) + 0.5
        m_per_tile = 40075016.686 / (2 ** z) * math.cos(math.radians(lat_c))
        cm = (m_per_tile / (512 if self._hd_var.get() else 256)) * 100.0
        if self._hd_var.get():
            # HD asks for finer than Apple's ~30-50 cm/px source -> mostly upscaled
            note = ("hier meist über Apples echter Detailgrenze → SD spart 75 % "
                    "Platz bei kaum sichtbarem Verlust" if cm < 30 else
                    "volle Schärfe der Quelle")
            self._hd_info.config(
                text=f"HD ≈ {cm:.0f} cm/px bei Zoom {z}. {note}.", fg="#88aacc")
        else:
            note = ("nahe Apples echter Quellauflösung → kaum Verlust gegenüber HD"
                    if cm <= 55 else "sichtbar weicher; für Nahansicht ggf. HD")
            self._hd_info.config(
                text=f"SD ≈ {cm:.0f} cm/px bei Zoom {z}, ¼ Speicher. {note}.",
                fg="#9fe0ff")

    def _cell_deg_u(self):
        """Cell size in base units (integer)."""
        return round(self._cell_deg() * self._SBASE)

    def _snap_coarse(self, lat, lon):
        """Snap (lat, lon) to nearest coarse-cell SW corner → (la_u, lo_u)."""
        cd_u = self._cell_deg_u()
        la_u = int(math.floor(lat * self._SBASE))
        lo_u = int(math.floor(lon * self._SBASE))
        return ((la_u // cd_u) * cd_u, (lo_u // cd_u) * cd_u)

    def _xy_to_coarse(self, cx, cy):
        lat, lon = self._xy_to_geo(cx, cy)
        return self._snap_coarse(lat, lon)

    def _key_geo(self, key):
        """Coarse key → (lat_sw, lon_sw) geographic SW corner in degrees."""
        return key[0] / self._SBASE, key[1] / self._SBASE

    # ---- Selection helpers ----

    def _toggle_coarse(self, coarse_key, mode):
        """Add or remove ALL base cells within a coarse cell."""
        cd_u = self._cell_deg_u()
        la_u, lo_u = coarse_key
        for dla in range(cd_u):
            for dlo in range(cd_u):
                k = (la_u + dla, lo_u + dlo)
                if mode == "add":
                    self._sel.add(k)
                else:
                    self._sel.discard(k)

    def _coarse_is_sel(self, la_u, lo_u):
        """Quick check: is the coarse cell selected? Uses center base cell."""
        cd_u = self._cell_deg_u()
        return (la_u + cd_u // 2, lo_u + cd_u // 2) in self._sel

    # ---- Tile count estimate ----

    def _estimate_tiles(self, zoom):
        """Estimate total slippy tiles for current selection at given zoom."""
        if not self._sel:
            return 0
        dsf_cells = defaultdict(list)
        for (la_u, lo_u) in self._sel:
            dg_la = int(math.floor(la_u / self._SBASE))
            dg_lo = int(math.floor(lo_u / self._SBASE))
            dsf_cells[(dg_la, dg_lo)].append((la_u, lo_u))
        total = 0
        for cells in dsf_cells.values():
            la_min = min(c[0] for c in cells)
            la_max = max(c[0] for c in cells) + 1
            lo_min = min(c[1] for c in cells)
            lo_max = max(c[1] for c in cells) + 1
            xs, xe, ys, ye = slippy_range_for_bounds(
                la_min / self._SBASE, lo_min / self._SBASE,
                la_max / self._SBASE, lo_max / self._SBASE, zoom)
            total += max(0, xe - xs + 1) * max(0, ye - ys + 1)
        return total

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _hdr(self, p, txt):
        f = tk.Frame(p, bg=self.C_HDR)
        f.pack(fill=tk.X, pady=(8, 0))
        tk.Label(f, text=txt, bg=self.C_HDR, fg="white",
                 font=("Segoe UI", 9, "bold"), padx=5, pady=2).pack(anchor="w")

    def _btn(self, p, txt, cmd, bg=None):
        b = tk.Button(p, text=txt, command=cmd,
                      bg=bg or self.C_BTN, fg="white", relief=tk.FLAT,
                      activebackground="#2aaa44", activeforeground="white",
                      cursor="hand2", font=("Segoe UI", 9, "bold"))
        b.pack(fill=tk.X, padx=8, pady=3)
        return b

    def _build_ui(self):
        outer = tk.Frame(self.root, bg=self.C_PANEL, width=245)
        outer.pack(side=tk.LEFT, fill=tk.Y)
        outer.pack_propagate(False)
        # Sidebar scrolls: in a small window every control stays reachable.
        sb = tk.Scrollbar(outer, orient="vertical", width=12)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        sc = tk.Canvas(outer, bg=self.C_PANEL, highlightthickness=0,
                       yscrollcommand=sb.set)
        sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=sc.yview)
        left = tk.Frame(sc, bg=self.C_PANEL)
        sc.create_window((0, 0), window=left, anchor="nw", width=233)
        left.bind("<Configure>",
                  lambda e: sc.configure(scrollregion=(0, 0, 233, e.height)))
        self._side_canvas = sc
        self._side_frame  = outer
        # Wheel events don't bubble in Tk -- bind_all and scroll only when the
        # pointer is over the sidebar (but not over the log, which scrolls
        # itself). The map canvas keeps its own wheel-zoom binding.
        self.root.bind_all("<MouseWheel>", self._on_side_wheel, add="+")
        self._build_left(left)
        right = tk.Frame(self.root, bg=self.C_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_map(right)
        self._build_bottom(right)

    def _on_side_wheel(self, e):
        try:
            w = self.root.winfo_containing(e.x_root, e.y_root)
        except (KeyError, tk.TclError):
            return
        over_log = False
        while w is not None:
            if w is self._log_box:
                over_log = True
            if w is self._side_frame:
                if not over_log:
                    self._side_canvas.yview_scroll(-1 if e.delta > 0 else 1,
                                                   "units")
                return
            w = getattr(w, "master", None)

    def _build_left(self, p):
        tk.Label(p, text="Ortho Region Selector",
                 bg=self.C_PANEL, fg=self.C_ACC,
                 font=("Segoe UI", 12, "bold")).pack(pady=(10, 0))
        tk.Label(p, text="X-Plane 12  •  Ortho4XP style",
                 bg=self.C_PANEL, fg=self.C_DIM,
                 font=("Segoe UI", 8)).pack()

        self._hdr(p, "Active Cell")
        self._active_var = tk.StringVar(value="(none)")
        tk.Label(p, textvariable=self._active_var, bg=self.C_PANEL, fg="#ffff80",
                 font=("Courier New", 10, "bold")).pack(pady=3)
        self._cell_size_var = tk.StringVar(value="")
        tk.Label(p, textvariable=self._cell_size_var, bg=self.C_PANEL, fg=self.C_DIM,
                 font=("Segoe UI", 8)).pack()

        cf = tk.Frame(p, bg=self.C_PANEL)
        cf.pack(fill=tk.X, padx=8, pady=(2, 2))
        tk.Label(cf, text="Cell size:", bg=self.C_PANEL, fg=self.C_FG,
                 font=("Segoe UI", 8), anchor="w").pack(side=tk.LEFT)
        # 1° = one X-Plane DSF tile = one zOrtho4XP_ folder: the unit downloads
        # and resume actually work in. "Auto" shrinks the grid with the map
        # zoom, a fixed value keeps it constant however far you zoom in.
        self._cellsize_var = tk.StringVar(value="1.0")
        self._cellsize_combo = ttk.Combobox(
            cf, textvariable=self._cellsize_var, width=8, state="readonly",
            values=["Auto", "2.0", "1.0", "0.5", "0.25", "0.1", "0.05"])
        self._cellsize_combo.pack(side=tk.RIGHT)
        self._cellsize_var.trace_add(
            "write", lambda *_: (self._update_sel_info(), self._redraw()))

        self._hdr(p, "Selection")
        self._sel_count = tk.StringVar(value="0 cells selected")
        tk.Label(p, textvariable=self._sel_count, bg=self.C_PANEL, fg="#aaffaa",
                 font=("Segoe UI", 9)).pack(anchor="w", padx=8)
        self._sel_range = tk.StringVar(value="")
        tk.Label(p, textvariable=self._sel_range, bg=self.C_PANEL, fg=self.C_DIM,
                 font=("Segoe UI", 8)).pack(anchor="w", padx=8)
        self._est_var = tk.StringVar(value="")
        tk.Label(p, textvariable=self._est_var, bg=self.C_PANEL, fg="#ff9955",
                 font=("Segoe UI", 8, "italic"),
                 wraplength=225, justify="left").pack(anchor="w", padx=8, pady=2)

        self._hdr(p, "Build Options")
        self._zoom_var  = tk.IntVar(value=16)
        self._water_var = tk.DoubleVar(value=0.60)
        self._skip_var  = tk.BooleanVar(value=False)
        self._retry_var = tk.IntVar(value=3)
        self._mesh_var  = tk.BooleanVar(value=False)

        self._spacing_var = tk.IntVar(value=MESH_SPACING_M)

        tk.Label(p, text="Terrain / Gelände:", bg=self.C_PANEL, fg="#ffcc66",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8, pady=(2, 0))

        tk.Radiobutton(
            p, text="X-Plane's mesh nutzen  (Bilder als Overlay)\n"
                    "• jeder Bereich wählbar, nie Löcher\n"
                    "• folgt X-Planes vorhandenen Hügeln",
            variable=self._mesh_var, value=False, command=self._on_mesh_toggle,
            bg=self.C_PANEL, fg=self.C_FG, selectcolor="#2d3a50",
            activebackground=self.C_PANEL, activeforeground=self.C_FG,
            justify="left", anchor="w", font=("Segoe UI", 9)).pack(
                fill=tk.X, padx=8, pady=(2, 0))

        tk.Radiobutton(
            p, text="Eigenes 3D-Mesh aus DEM  (ersetzt XP-Terrain)\n"
                    "• nur ganze 1°×1°-Kacheln\n"
                    "• eigene Höhen, Detail unten wählbar",
            variable=self._mesh_var, value=True, command=self._on_mesh_toggle,
            bg=self.C_PANEL, fg="#ffcc66", selectcolor="#2d3a50",
            activebackground=self.C_PANEL, activeforeground="#ffcc66",
            justify="left", anchor="w", font=("Segoe UI", 9)).pack(
                fill=tk.X, padx=8, pady=(2, 4))

        mf = tk.Frame(p, bg=self.C_PANEL)
        mf.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(mf, text="Mesh detail (m):", bg=self.C_PANEL, fg=self.C_FG,
                 font=("Segoe UI", 9), width=17, anchor="w").pack(side=tk.LEFT)
        self._spacing_combo = ttk.Combobox(
            mf, textvariable=self._spacing_var, values=[300, 200, 150, 100, 60],
            width=5, state="disabled")
        self._spacing_combo.pack(side=tk.RIGHT)
        self._mesh_info = tk.Label(p, text="", bg=self.C_PANEL, fg="#88aacc",
                                    font=("Segoe UI", 8, "italic"),
                                    wraplength=225, justify="left")
        self._mesh_info.pack(anchor="w", padx=8, pady=(0, 3))
        self._spacing_var.trace_add("write", lambda *_: self._update_mesh_info())
        self._update_mesh_info()   # set initial hint for the default (overlay)

        self._pargrp_var = tk.IntVar(value=6)
        for label, var, kind, opts in [
            ("Zoom level",      self._zoom_var,   "combo", [14,15,16,17,18,19]),
            ("Parallel groups", self._pargrp_var, "combo", [1,2,3,4,6,8,12,16,24]),
            ("Water threshold", self._water_var,  "entry", None),
            ("Max retries",     self._retry_var,  "combo", [1,2,3,5]),
        ]:
            f = tk.Frame(p, bg=self.C_PANEL)
            f.pack(fill=tk.X, padx=8, pady=2)
            tk.Label(f, text=label+":", bg=self.C_PANEL, fg=self.C_FG,
                     font=("Segoe UI", 9), width=17, anchor="w").pack(side=tk.LEFT)
            if kind == "combo":
                ttk.Combobox(f, textvariable=var, values=opts,
                             width=5, state="readonly").pack(side=tk.RIGHT)
            else:
                tk.Entry(f, textvariable=var, width=7, bg="#2d3a50",
                         fg="white", insertbackground="white",
                         relief=tk.FLAT).pack(side=tk.RIGHT)

        # (the old "Generate mipmaps" checkbox was removed: texconv always
        # builds the full mip chain, the switch never did anything)
        # Image resolution HD/SD switch. True = HD (4096 px group texture),
        # False = SD (2048 px). BooleanVar so the rest of the code (resume /
        # auto-sync / estimate / cfg) keeps working unchanged; only the widget
        # changed from a lone checkbox to an explicit, labelled toggle.
        self._hd_var = tk.BooleanVar(value=True)
        tk.Label(p, text="Bild-Auflösung / Resolution:", bg=self.C_PANEL,
                 fg="#ffcc66", font=("Segoe UI", 9, "bold")).pack(
                     anchor="w", padx=8, pady=(4, 0))
        tk.Radiobutton(
            p, text="HD  –  4096 px  (~11 MB/Gruppe)\n• volle Schärfe",
            variable=self._hd_var, value=True,
            bg=self.C_PANEL, fg=self.C_FG, selectcolor="#2d3a50",
            activebackground=self.C_PANEL, activeforeground=self.C_FG,
            justify="left", anchor="w", font=("Segoe UI", 9)).pack(
                fill=tk.X, padx=8, pady=(2, 0))
        tk.Radiobutton(
            p, text="SD  –  2048 px  (~2,8 MB/Gruppe, ¼ Speicher)\n"
                    "• ~½ Zoomstufe weicher; ideal ab Zoom 17–18",
            variable=self._hd_var, value=False,
            bg=self.C_PANEL, fg="#9fe0ff", selectcolor="#2d3a50",
            activebackground=self.C_PANEL, activeforeground="#9fe0ff",
            justify="left", anchor="w", font=("Segoe UI", 9)).pack(
                fill=tk.X, padx=8, pady=(0, 2))
        self._hd_info = tk.Label(p, text="", bg=self.C_PANEL, fg="#88aacc",
                                 font=("Segoe UI", 8, "italic"),
                                 wraplength=225, justify="left")
        self._hd_info.pack(anchor="w", padx=8, pady=(0, 2))
        # Flipping HD/SD must refresh the live disk/DL estimate and the hint.
        self._hd_var.trace_add(
            "write", lambda *_: (self._update_hd_info(), self._update_sel_info()))

        tk.Checkbutton(p, text="Skip pure-water tiles "
                               "(1-KB-Platzhalter statt Download)",
                       variable=self._skip_var, bg=self.C_PANEL, fg=self.C_FG,
                       selectcolor="#2d3a50", wraplength=225, justify="left",
                       activebackground=self.C_PANEL, activeforeground=self.C_FG,
                       font=("Segoe UI", 9)).pack(anchor="w", padx=8, pady=1)
        self._update_hd_info()

        self._hdr(p, "Output Folder")
        self._out_var = tk.StringVar(value=os.path.expanduser("~\\Desktop"))
        tk.Entry(p, textvariable=self._out_var, bg="#2d3a50", fg="white",
                 insertbackground="white", relief=tk.FLAT,
                 font=("Segoe UI", 8)).pack(fill=tk.X, padx=8, pady=4)
        # Re-scan for already-downloaded cells whenever the folder changes
        # (debounced so typing doesn't scan on every keystroke).
        self._out_var.trace_add("write", lambda *_: self._schedule_scan())

        self._hdr(p, "Actions")
        self._resume_btn = self._btn(p, "↩  Unvollständiges auswählen",
                                     self._select_todo, bg="#8a6a10")
        self._resume_btn.config(state=tk.DISABLED)
        self._start_btn  = self._btn(p, "▶  Start / Fortsetzen", self._start)
        self._pause_btn  = self._btn(p, "⏸  Pause (resumable)", self._pause, bg="#b07020")
        self._pause_btn.config(state=tk.DISABLED)
        self._btn(p, "Clear Selection", self._clear, bg="#444455")

        self._hdr(p, "Map Controls")
        vf = tk.Frame(p, bg=self.C_PANEL)
        vf.pack(fill=tk.X, padx=8, pady=4)
        for txt, cmd in [("+ Zoom", self._zoom_in),
                         ("− Zoom", self._zoom_out),
                         ("Reset",  self._reset_view)]:
            tk.Button(vf, text=txt, command=cmd, bg="#2d3a50", fg=self.C_FG,
                      relief=tk.FLAT, cursor="hand2",
                      font=("Segoe UI", 8), width=7).pack(side=tk.LEFT, padx=2)
        self._btn(p, "↻  Reload map", self._start_bg_load, bg="#2d3a50")

        self._hdr(p, "Legend")
        for color, lbl in [(self.C_SEL,  "Auswahl (weiß)"),
                            (self.C_DONE, "Fertig geladen (grün)"),
                            (self.C_TODO, "Unvollständig / pausiert (gelb)"),
                            ("#cccccc", "Grid line (zoom to shrink)"),
                            ("#ff9900", "Aschersleben"),
                            ("#ff4444", "Reference city")]:
            lf = tk.Frame(p, bg=self.C_PANEL)
            lf.pack(anchor="w", padx=8, pady=1)
            tk.Frame(lf, bg=color, width=14, height=14).pack(side=tk.LEFT, padx=(0,5))
            tk.Label(lf, text=lbl, bg=self.C_PANEL, fg=self.C_FG,
                     font=("Segoe UI", 8)).pack(side=tk.LEFT)

        self._hdr(p, "Log")
        self._log_box = scrolledtext.ScrolledText(
            p, height=10, bg="#0c1120", fg="#88ffaa",
            font=("Courier New", 7), relief=tk.FLAT,
            state=tk.DISABLED, wrap=tk.WORD)
        self._log_box.pack(fill=tk.BOTH, padx=4, pady=4, expand=True)
        self._zoom_var.trace_add("write", lambda *_: self._update_sel_info())

    def _build_map(self, parent):
        tk.Label(parent,
                 text="Left-click/drag = select  •  Right-click = deselect  "
                      "•  Right-drag / Ctrl+drag = pan  •  Scroll = zoom map  "
                      "(cells get smaller as you zoom in, unless Cell size is fixed)",
                 bg=self.C_BG, fg=self.C_DIM,
                 font=("Segoe UI", 8)).pack(side=tk.TOP, anchor="w", padx=6, pady=(4,0))
        self._canvas = tk.Canvas(parent, bg="#1a3050", cursor="crosshair",
                                 highlightthickness=1,
                                 highlightbackground=self.C_ACC)
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        c = self._canvas
        c.bind("<Configure>",         self._on_resize)
        c.bind("<Button-1>",          self._on_lclick)
        c.bind("<B1-Motion>",         self._on_ldrag)
        c.bind("<ButtonRelease-1>",   self._on_lrelease)
        c.bind("<Motion>",            self._on_motion)
        c.bind("<Button-3>",          self._on_rpress)
        c.bind("<B3-Motion>",         self._on_rmotion)
        c.bind("<ButtonRelease-3>",   self._on_rrelease)
        c.bind("<MouseWheel>",        self._on_scroll)
        c.bind("<Leave>",             lambda _: self._set_hover(None))
        c.bind("<Control-Button-1>",  self._on_pan_start)
        c.bind("<Control-B1-Motion>", self._on_pan)

    def _build_bottom(self, parent):
        bottom = tk.Frame(parent, bg=self.C_BG)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=(0,4))
        self._pb = ttk.Progressbar(bottom, mode="determinate", maximum=100)
        self._pb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,8))
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TProgressbar", troughcolor="#2a3550",
                        background=self.C_ACC, thickness=18)
        self._status_var = tk.StringVar(value="Ready – select cells then click Start")
        tk.Label(bottom, textvariable=self._status_var, bg=self.C_BG, fg=self.C_FG,
                 font=("Segoe UI", 9), width=45, anchor="w").pack(side=tk.LEFT)
        self._pct_var = tk.StringVar(value="")
        tk.Label(bottom, textvariable=self._pct_var, bg=self.C_BG, fg=self.C_ACC,
                 font=("Courier New", 9, "bold"), width=10).pack(side=tk.LEFT)
        self._eta_var = tk.StringVar(value="")
        tk.Label(bottom, textvariable=self._eta_var, bg=self.C_BG, fg=self.C_DIM,
                 font=("Segoe UI", 8), width=14).pack(side=tk.LEFT)
        # Live download speed (shows whether the connection is saturated)
        self._mbit_var = tk.StringVar(value="")
        tk.Label(bottom, textvariable=self._mbit_var, bg=self.C_BG, fg="#66ddff",
                 font=("Courier New", 9, "bold"), width=16, anchor="e").pack(side=tk.LEFT)

    def _poll_bw(self):
        """Sample the global byte counter every 0.5 s → live Mbit/s readout,
        smoothed over the last ~3 s. A bursty line (dropping to 10-20 Mbit for
        <1 s) would make an instantaneous readout jump around alarmingly; the
        rolling average shows the sustained rate that actually matters."""
        now = time.time()
        total = BW.read()
        if self._bw_last_t is not None:
            dt = now - self._bw_last_t
            if dt > 0:
                inst = (total - self._bw_last_b) * 8.0 / 1e6 / dt
                self._bw_hist.append(inst)
                if len(self._bw_hist) > 6:          # ~3 s window at 0.5 s ticks
                    self._bw_hist.pop(0)
                avg = sum(self._bw_hist) / len(self._bw_hist)
                if avg > 0.05:
                    self._bw_peak = max(self._bw_peak, avg)
                    self._mbit_var.set(f"{avg:5.0f} Mbit/s")
                else:
                    self._mbit_var.set("")
        self._bw_last_t = now
        self._bw_last_b = total
        self._bw_after = self.root.after(500, self._poll_bw)

    # ------------------------------------------------------------------
    # Resume scan: which cells already hold valid downloads
    # ------------------------------------------------------------------

    def _schedule_scan(self):
        """Debounced rescan (e.g. while typing the output path)."""
        if getattr(self, "_scan_after", None) is not None:
            try: self.root.after_cancel(self._scan_after)
            except Exception: pass
        self._scan_after = self.root.after(600, self._scan_existing)

    # ---- Interrupted-run memory -------------------------------------------
    # An aborted run only leaves zOrtho4XP_ folders for the tiles it actually
    # REACHED. Pausing early therefore loses every tile that was selected but
    # not started yet -- so the selection is also written to a session file and
    # replayed on the next launch. Kept out of the output folder (the user's
    # Desktop) on purpose.

    def _session_path(self):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        d = os.path.join(base, "OrthoRegionSelector")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "session.json")

    def _run_settings(self):
        """Everything a resume needs, captured NOW."""
        return dict(out=self._out_var.get().strip(),
                    zoom=int(self._zoom_var.get()),
                    hd=bool(self._hd_var.get()),
                    skip_water=bool(self._skip_var.get()),
                    base_mesh=bool(self._mesh_var.get()),
                    cells=sorted(self._sel))

    def _session_save(self, status, snap=None):
        """snap: the settings frozen at Start. The map stays interactive during
        a run (clicks change the selection and even flip the zoom control via
        the auto-sync), so sampling the LIVE widgets on pause/close saved
        whatever the user had fiddled with -- not what the run actually covers
        -- and the resume then missed cells or restored the wrong zoom."""
        try:
            data = dict(self._run_settings() if snap is None else snap)
            data.update(status=status, ts=time.time())
            p = self._session_path()
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, p)          # atomic: never leave a half-written file
        except (OSError, tk.TclError, ValueError):
            pass                        # a lost session file must never break a run

    def _session_load(self, out):
        """The last run's exact scope, or None. A FINISHED session is still
        returned: it is what proves its tiles need nothing more, and without it
        the folder fallback would re-flag the rest of a partly-covered degree
        tile as unfinished right after a successful run."""
        try:
            with open(self._session_path(), encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            return None
        # normpath too: a trailing backslash in the typed output folder used
        # to silently unmatch the session (-> over-broad folder fallback).
        def norm(p): return os.path.normcase(os.path.normpath(str(p) or "."))
        if norm(d.get("out", "")) != norm(out):
            return None                 # session belongs to a different folder
        return d

    def _scan_existing(self):
        """Scan the output folder (in the background) for already-downloaded,
        VALID group textures and mark those map cells. Runs on startup, when
        the output folder changes, and after each generation/pause."""
        # Cancel (not just forget) any pending debounced rescan: the startup
        # scan used to null the id of a still-pending timer, which then fired
        # a duplicate scan -- or fired into the destroyed window on exit.
        if getattr(self, "_scan_after", None) is not None:
            try: self.root.after_cancel(self._scan_after)
            except Exception: pass
        self._scan_after = None
        out = self._out_var.get().strip()
        if not self._running:
            self._status_var.set("Prüfe vorhandene Downloads …")
        threading.Thread(target=self._scan_worker, args=(out,), daemon=True).start()

    def _scan_worker(self, out):
        # Any unexpected error here used to kill the thread silently: no
        # marking ever appeared and the status stayed on "Prüfe vorhandene
        # Downloads …" forever. Report it and reset to an empty marking.
        try:
            self._scan_worker_inner(out)
        except Exception:
            import traceback
            err = traceback.format_exc()
            try:
                self.root.after(0, lambda: (
                    self._log("[!] Scan des Ausgabeordners fehlgeschlagen:\n" + err),
                    self._apply_scan(set(), {}, 0, 0, frozenset(), None)))
            except Exception:
                pass          # window closed while scanning

    def _scan_worker_inner(self, out):
        zmap = {}                     # base cell -> {zoom: px} of existing data
        groups = set()                # (deg_lat, deg_lon, zoom, gx, gy) on disk
        touched = set()               # base cells holding at least one group
        tiles = set()                 # (deg_lat, deg_lon) folders that hold data
        corrupt = 0
        n_ph = 0                      # tiny water placeholders found
        pat = re.compile(r"_z(\d+)_g(-?\d+)_(-?\d+)\.dds$", re.IGNORECASE)
        fpat = re.compile(r"^zOrtho4XP_([+-]\d+)([+-]\d{3})$")
        # A complete DXT1 DDS has an EXACT, constant byte size per resolution,
        # so scandir's stat data alone separates good from partial files.
        # No per-file open(): a cold-cache scan of tens of thousands of
        # textures finishes in seconds instead of minutes (the strict header
        # check still guards every group at download/resume time).
        px_by_size = {_dxt1_dds_size(4096): 4096,
                      _dxt1_dds_size(2048): 2048,
                      _dxt1_dds_size(WATER_PLACEHOLDER_PX): WATER_PLACEHOLDER_PX}
        try:
            for name in os.listdir(out):
                fm = fpat.match(name)
                if not fm:
                    continue
                tla, tlo = int(fm.group(1)), int(fm.group(2))
                texdir = os.path.join(out, name, "textures")
                try:
                    entries = list(os.scandir(texdir))
                except OSError:
                    continue
                for de in entries:
                    m = pat.search(de.name)
                    if not m:
                        continue
                    try:
                        px = px_by_size.get(de.stat().st_size, 0)
                    except OSError:
                        continue
                    if px == 0:
                        corrupt += 1      # partial write -> re-downloaded later
                        continue
                    is_water_ph = (px == WATER_PLACEHOLDER_PX)
                    z = int(m.group(1)); gx = int(m.group(2)); gy = int(m.group(3))
                    if is_water_ph:
                        n_ph += 1
                    # Keyed INCLUDING the folder's tile: groups straddling a 1°
                    # border exist once per folder (own name prefix each). With
                    # a bare (z,gx,gy) the neighbour's copy stood in for a
                    # missing one here -- border cells showed green although
                    # THIS tile's file was absent, resume skipped them, and the
                    # DSF ended up with a missing stripe at the tile edge.
                    groups.add((tla, tlo, z, gx, gy))
                    lonW, latN = tile_to_lonlat(gx * GROUP_TILES,       gy * GROUP_TILES,       z)
                    lonE, latS = tile_to_lonlat((gx + 1) * GROUP_TILES, (gy + 1) * GROUP_TILES, z)
                    la0 = int(math.floor(latS * self._SBASE))
                    la1 = int(math.floor((latN - 1e-9) * self._SBASE))
                    lo0 = int(math.floor(lonW * self._SBASE))
                    lo1 = int(math.floor((lonE - 1e-9) * self._SBASE))
                    for la in range(la0, la1 + 1):
                        for lo in range(lo0, lo1 + 1):
                            touched.add((la, lo))
                            if not is_water_ph:
                                # placeholders carry no real HD/SD info --
                                # don't let them drive zoom/HD auto-sync or
                                # the resolution-conflict popup
                                zmap.setdefault((la, lo), {})[z] = px
                # An existing zOrtho4XP_ folder means that 1x1 deg tile was
                # part of a run -- even an empty one (created, then the run
                # stopped). Its cells count as "intended", so an aborted tile
                # shows up as unfinished instead of silently vanishing.
                tiles.add((tla, tlo))
        except OSError:
            pass

        # A cell is DONE only if EVERY group covering it exists in ITS OWN
        # tile's folder. Marking a cell done because it holds ONE of its ~30
        # groups painted the whole 0.05 deg cell (and a fringe beyond the
        # selection) as finished -- that was the "green is too big / hangs
        # over the selection" bug.
        zooms = sorted({g[2] for g in groups})
        done = set()
        done_z = {}                   # cell -> the zoom that completed it
        for (la, lo) in touched:
            ta, to = la // self._SBASE, lo // self._SBASE
            for z in zooms:
                xs, xe, ys, ye = slippy_range_for_bounds(
                    la / self._SBASE, lo / self._SBASE,
                    (la + 1) / self._SBASE, (lo + 1) / self._SBASE, z)
                if all((ta, to, z, gx, gy) in groups
                       for gy in range(ys // GROUP_TILES, ye // GROUP_TILES + 1)
                       for gx in range(xs // GROUP_TILES, xe // GROUP_TILES + 1)):
                    done.add((la, lo))
                    done_z[(la, lo)] = z
                    break

        # Per-cell (zoom, px) for auto-sync + the resolution-conflict popup.
        # Prefer the zoom that actually COMPLETED the cell: keeping whichever
        # file was scanned last meant a handful of stray z19 test textures made
        # finished z18 cells report z19 -- wrong auto-zoom and a bogus
        # "Andere Auflösung überschreiben?" popup.
        z_counts = defaultdict(int)
        for g in groups:
            z_counts[g[2]] += 1
        z_major = max(z_counts, key=z_counts.get) if z_counts else None
        info = {}
        for cell, zd in zmap.items():
            z = done_z.get(cell)
            if z is None or z not in zd:
                z = z_major if z_major in zd else next(iter(zd))
            info[cell] = (z, zd[z])

        # Everything the last run INTENDED to cover but that is not done is what
        # a resume still has to fetch -> shown yellow.
        #
        # For the tiles it names, the saved selection IS the intent and wins:
        # deriving intent from the folders rounds a partial run up to the whole
        # 1x1 deg tile (verified: a 0.2x0.2 deg run resumed as the full degree,
        # 25x the tiles -- and a *finished* partial run came back flagged as
        # unfinished). For every other tile that holds data -- older downloads,
        # a lost session file -- the folder is all we have, and there the whole
        # degree tile is the best guess at what was wanted.
        sess = self._session_load(out)
        sess_cells = ({tuple(c) for c in sess.get("cells", []) if len(c) == 2}
                      if sess else set())
        sess_tiles = {(la // self._SBASE, lo // self._SBASE)
                      for (la, lo) in sess_cells}
        intent = set(sess_cells)
        for (dla, dlo) in tiles:
            if (dla, dlo) in sess_tiles:
                continue                     # the session describes this tile exactly
            la0 = dla * self._SBASE; lo0 = dlo * self._SBASE
            for la in range(la0, la0 + self._SBASE):
                for lo in range(lo0, lo0 + self._SBASE):
                    intent.add((la, lo))
        todo = intent - done

        # The settings the unfinished run used -> restored on startup so
        # "resume" continues at the same zoom/HD (and the same overlay/
        # base-mesh mode) instead of overwriting.
        cfg = None
        flags = None
        if sess and sess.get("status") != "done" and sess_cells - done:
            try:
                cfg = (int(sess["zoom"]), 4096 if sess.get("hd") else 2048)
            except (KeyError, TypeError, ValueError):
                cfg = None            # old/corrupt session file: fall back below
            flags = dict(base_mesh=bool(sess.get("base_mesh")),
                         skip_water=bool(sess.get("skip_water")))
        if cfg is None:
            cands = [info[k] for k in touched if k in info]
            if cands:
                cfg = max(set(cands), key=cands.count)
        try:
            self.root.after(0, lambda: self._apply_scan(
                done, info, corrupt, n_ph, todo, cfg, flags))
        except Exception:
            pass                  # window closed while scanning

    def _apply_scan(self, done, info, corrupt, n_ph=0, todo=frozenset(),
                    cfg=None, flags=None):
        first = not self._scan_ran
        self._scan_ran = True
        self._done_base = done
        self._todo_base = todo
        self._done_info = info
        self._todo_cfg  = cfg
        self._todo_flags = flags          # base_mesh/skip_water of that run
        self._done_coarse = None          # invalidate the coarse-lookup cache
        if corrupt:
            self._log(f"[!] {corrupt} beschädigte/unvollständige Texturen gefunden "
                      f"— werden beim Start automatisch neu geladen.")
        if n_ph and not self._skip_var.get():
            # The folder was built with skip_water on; keep honouring the
            # placeholders instead of silently re-downloading them as imagery.
            self._skip_var.set(True)
            self._log(f"[i] {n_ph} Wasser-Platzhalter gefunden → »Skip "
                      f"pure-water tiles« wieder aktiviert. (Ausschalten lädt "
                      f"dort echtes Wasser-Bildmaterial.)")
        if done:
            self._log(f"[i] Fertig (grün): {len(done)} Zellen "
                      f"= {len(done)/(self._SBASE**2):.2f}°².")
        if todo and cfg and first and not self._running:
            # Restore the settings of the interrupted run so pressing
            # "Fortsetzen" really continues instead of starting a second,
            # differently-configured download over the same ground.
            z, px = cfg
            self._zoom_var.set(z)
            self._hd_var.set(px >= 4096)
            self._apply_todo_flags()
            self._log(f"[i] Unvollständig (gelb): {len(todo)} Zellen "
                      f"= {len(todo)/(self._SBASE**2):.2f}°². "
                      f"Zoom {z} / {'HD' if px >= 4096 else 'SD'} übernommen.")
            self._log("[i] »↩ Unvollständiges auswählen« + »Fortsetzen« "
                      "lädt genau das Fehlende nach.")
        elif todo:
            self._log(f"[i] Unvollständig (gelb): {len(todo)} Zellen.")
        self._resume_btn.config(
            state=tk.NORMAL if todo else tk.DISABLED,
            text=(f"↩  Unvollständiges auswählen ({len(todo)})" if todo
                  else "↩  Unvollständiges auswählen"))
        if not self._running:
            # The post-run rescan lands seconds after _on_done set its verdict:
            # never let it overwrite "Done!" or the "Fehler — siehe Log" hint.
            cur = self._status_var.get()
            if todo and not cur.startswith("Fehler"):
                self._status_var.set(
                    f"{len(todo)} Zellen unvollständig — fortsetzbar")
            elif not todo and cur.startswith("Prüfe"):
                self._status_var.set("Ready – select cells then click Start")
        self._update_sel_info()
        self._redraw()

    def _apply_todo_flags(self):
        """Re-apply the interrupted run's mode switches (base-mesh, skip-water).
        Zoom/HD alone were restored before -- a paused BASE-MESH run would
        quietly resume as a draped overlay, producing the wrong DSF type."""
        flags = getattr(self, "_todo_flags", None)
        if not flags:
            return
        if bool(flags.get("base_mesh")) != bool(self._mesh_var.get()) \
                and not self._sel:      # don't stomp a selection the user made
            self._mesh_var.set(bool(flags.get("base_mesh")))
            self._on_mesh_toggle()
            self._log("[i] Modus des unterbrochenen Laufs übernommen: "
                      + ("eigenes 3D-Mesh." if flags.get("base_mesh")
                         else "Overlay."))
        if flags.get("skip_water") and not self._skip_var.get():
            self._skip_var.set(True)

    def _select_todo(self):
        """Select exactly the cells an interrupted run still owes us, at the
        zoom/HD that run used -- one click, then Start continues seamlessly."""
        if not self._todo_base:
            return
        # Mode first: _on_mesh_toggle clears the selection, so it must run
        # BEFORE the todo cells are added.
        self._apply_todo_flags()
        self._sel |= set(self._todo_base)
        if self._todo_cfg:
            z, px = self._todo_cfg
            self._zoom_var.set(z)
            self._hd_var.set(px >= 4096)
        # Frame the restored area so it is actually visible on the map.
        la = [k[0] for k in self._todo_base]; lo = [k[1] for k in self._todo_base]
        lat_s = min(la) / self._SBASE; lat_n = (max(la) + 1) / self._SBASE
        lon_w = min(lo) / self._SBASE; lon_e = (max(lo) + 1) / self._SBASE
        pad = max(0.25, (lon_e - lon_w) * 0.25)
        self._v["lon_min"] = lon_w - pad; self._v["lon_max"] = lon_e + pad
        self._v["lat_min"] = lat_s - pad; self._v["lat_max"] = lat_n + pad
        self._enforce_aspect()
        self._log(f"[i] {len(self._todo_base)} unvollständige Zellen ausgewählt. "
                  f"»Fortsetzen« lädt nur die fehlenden Gruppen.")
        self._update_sel_info()
        self._redraw()
        self._schedule_bg_refresh()

    def _coarse_has(self, la_u, lo_u, which):
        """Does the coarse cell at (la_u,lo_u) overlap done/todo base cells?

        Uses a precomputed aggregation instead of probing every base cell
        inside the coarse cell (up to 1600 set lookups PER CELL, ~52 ms per
        redraw at continent view with a few downloaded tiles -- and hovering
        redraws on every mouse move; measured 103x faster this way)."""
        src = self._done_base if which == "done" else self._todo_base
        if not src:
            return False
        cd_u = self._cell_deg_u()
        if self._done_coarse is None or self._done_coarse[0] != cd_u:
            self._done_coarse = (cd_u, {
                "done": {((la // cd_u) * cd_u, (lo // cd_u) * cd_u)
                         for (la, lo) in self._done_base},
                "todo": {((la // cd_u) * cd_u, (lo // cd_u) * cd_u)
                         for (la, lo) in self._todo_base},
            })
        return (la_u, lo_u) in self._done_coarse[1][which]

    # ------------------------------------------------------------------
    # Satellite background loading
    # ------------------------------------------------------------------

    _BG_TILE_PX = 256
    _BG_MAX_TILES = 140     # safety cap per refresh

    def _start_bg_load(self):
        """Reload-map button: force a fresh fetch of the current view."""
        self._bg_error = None
        self._refresh_bg(force=True)

    def _ideal_tile_zoom(self):
        """Slippy zoom so tiles land ~256 px on screen at the current view."""
        lon_span = max(1e-6, self._v["lon_max"] - self._v["lon_min"])
        z = math.log2(360.0 * self._cw / (self._BG_TILE_PX * lon_span))
        return max(2, min(17, int(round(z))))

    def _schedule_bg_refresh(self):
        """Debounced: fires ~350 ms after the last zoom/pan so rapid input
        coalesces into a single fetch (keeps Apple's rate limit happy)."""
        if self._bg_after is not None:
            try: self.root.after_cancel(self._bg_after)
            except Exception: pass
        self._bg_after = self.root.after(350, self._refresh_bg)

    def _refresh_bg(self, force=False):
        self._bg_after = None
        v = self._v
        z = self._ideal_tile_zoom()
        nmax = 2 ** z
        x0, y0 = lonlat_to_slippy(v["lon_min"], v["lat_max"], z)
        x1, y1 = lonlat_to_slippy(v["lon_max"], v["lat_min"], z)
        x0 -= 1; y0 -= 1; x1 += 1; y1 += 1          # one-tile margin for panning
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(nmax - 1, x1); y1 = min(nmax - 1, y1)
        # shrink from the edges if the window is somehow huge (safety)
        while (x1 - x0 + 1) * (y1 - y0 + 1) > self._BG_MAX_TILES and x1 > x0 and y1 > y0:
            x0 += 1; x1 -= 1; y0 += 1; y1 -= 1

        req = [(z, tx, ty) for ty in range(y0, y1 + 1) for tx in range(x0, x1 + 1)]
        # Nothing to do if we already show this zoom and every tile is cached
        if (not force and z == self._bg_zoom and self._bg_mosaic is not None
                and all(k in self._bg_cache for k in req)
                and self._bounds_cover_view()):
            return

        self._bg_gen += 1
        gen = self._bg_gen
        self._bg_loading = (self._bg_mosaic is None)   # only show "loading" on first ever
        if self._bg_mosaic is None:
            self._redraw()
        threading.Thread(target=self._fetch_tiles,
                         args=(gen, z, x0, y0, x1, y1), daemon=True).start()

    def _bounds_cover_view(self):
        if not self._bg_bounds:
            return False
        lw, ls, le, ln = self._bg_bounds
        v = self._v
        return (lw <= v["lon_min"] and le >= v["lon_max"]
                and ls <= v["lat_min"] and ln >= v["lat_max"])

    def _fetch_tiles(self, gen, z, x0, y0, x1, y1):
        try:
            if self._bg_tokens is None:
                self._bg_tokens = AppleTokenService()
                self._bg_tokens.refresh()
        except Exception as e:
            try:
                self.root.after(0, lambda err=str(e), g=gen: self._on_bg_done(
                    None, None, err, g, None))
            except (RuntimeError, tk.TclError):
                pass      # window closed while this fetch was still in flight
            return
        TS = self._BG_TILE_PX
        # Fetch only the tiles not already cached -- in parallel.
        missing = [(tx, ty) for ty in range(y0, y1 + 1) for tx in range(x0, x1 + 1)
                   if (z, tx, ty) not in self._bg_cache]
        if missing and gen == self._bg_gen:
            fetched = download_tiles_parallel(self._bg_tokens, missing, z,
                                              workers=8, max_ret=2, hd=False)
            # ~200 KB per decoded 256px tile -> cap at ~120 MB, and evict
            # other zoom levels first; never dump the view we're building.
            if len(self._bg_cache) > 600:
                for k in [k for k in self._bg_cache if k[0] != z]:
                    del self._bg_cache[k]
                if len(self._bg_cache) > 600:
                    self._bg_cache.clear()
            for (tx, ty), im in fetched.items():
                if im.size != (TS, TS):
                    im = im.resize((TS, TS), Image.LANCZOS)
                self._bg_cache[(z, tx, ty)] = im
        if gen != self._bg_gen:
            return                              # superseded by a newer view
        cols, rows = x1 - x0 + 1, y1 - y0 + 1
        mosaic = Image.new("RGB", (cols * TS, rows * TS), (20, 35, 70))
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                im = self._bg_cache.get((z, tx, ty))
                if im is not None:
                    mosaic.paste(im, ((tx - x0) * TS, (ty - y0) * TS))
        lon_w, lat_n = tile_to_lonlat(x0,     y0,     z)
        lon_e, lat_s = tile_to_lonlat(x1 + 1, y1 + 1, z)
        bounds = (lon_w, lat_s, lon_e, lat_n)
        try:
            self.root.after(0, lambda m=mosaic, b=bounds, g=gen, zz=z:
                            self._on_bg_done(m, b, None, g, zz))
        except (RuntimeError, tk.TclError):
            pass          # window closed while this fetch was still in flight

    def _on_bg_done(self, mosaic, bounds, err, gen=None, zoom=None):
        # Ignore results from a request that a newer zoom/pan already replaced.
        if gen is not None and gen != self._bg_gen:
            return
        self._bg_loading = False
        if err:
            self._bg_error = str(err)
            self._log(f"[!] Karte: {err}")
        else:
            self._bg_mosaic = mosaic
            self._bg_bounds = bounds
            self._bg_zoom   = zoom
            self._bg_error  = None
        self._redraw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _redraw(self):
        c = self._canvas
        c.delete("all")
        if self._bg_mosaic is not None:
            self._draw_bg()
        else:
            c.create_rectangle(0, 0, self._cw, self._ch, fill="#1a3050", outline="")
            if self._bg_loading:
                c.create_text(self._cw//2, self._ch//2,
                              text="Lade Satellitenkarte …",
                              fill="white", font=("Segoe UI", 14))
            elif self._bg_error:
                c.create_text(self._cw//2, self._ch//2 - 14,
                              text="Satellitenkarte konnte nicht geladen werden",
                              fill="#ff9955", font=("Segoe UI", 13, "bold"))
                c.create_text(self._cw//2, self._ch//2 + 10,
                              text=self._bg_error[:110],
                              fill="#cccccc", font=("Segoe UI", 8))
                c.create_text(self._cw//2, self._ch//2 + 34,
                              text="Apple drosselt nach vielen Downloads. "
                                   "„Reload map“ klicken (Auswahl bleibt erhalten).",
                              fill="#88aacc", font=("Segoe UI", 9))
        self._draw_cells(self._todo_base, self.C_TODO)  # yellow: unfinished
        self._draw_cells(self._done_base, self.C_DONE)  # green:  complete
        # The selection is drawn at its TRUE 0.05° resolution too: with a 1°
        # grid, a partially selected degree cell (exactly what resuming an
        # aborted tile produces) showed no selection at all before.
        self._draw_cells(self._sel, self.C_SEL, stipple="gray25")
        self._draw_grid()
        self._draw_axes()
        self._draw_cities()
        if self._sel:
            self._draw_bbox()

    def _draw_cells(self, cells, color, stipple="gray50"):
        """Draw the TRUE footprint of a base-cell set in `color`.

        At high zoom every 0.05° base cell is drawn individually. Zoomed out
        (base cells sub-pixel) k×k base cells are merged into blocks just big
        enough to stay visible (~2 px), and a block is only drawn when EVERY
        base cell in it belongs to the set -- so the painted area never grows
        beyond the real footprint, at any zoom."""
        if not cells:
            return
        v = self._v
        b = 1.0 / self._SBASE
        px_per_deg = self._cw / (v["lon_max"] - v["lon_min"])
        cell_px = max(b * px_per_deg, 1e-9)
        k = max(1, int(math.ceil(2.0 / cell_px)))
        la_min = int(math.floor(v["lat_min"] * self._SBASE)) - k
        la_max = int(math.ceil (v["lat_max"] * self._SBASE)) + k
        lo_min = int(math.floor(v["lon_min"] * self._SBASE)) - k
        lo_max = int(math.ceil (v["lon_max"] * self._SBASE)) + k
        # Per block keep the bounding box of the cells actually in it: a fully
        # covered block yields the block itself, an edge block only the covered
        # strip. Rounding the block up would repaint the old "green hangs over
        # the selection" bug at continent zoom.
        boxes = {}
        for (la, lo) in cells:
            if not (la_min <= la <= la_max and lo_min <= lo <= lo_max):
                continue
            key = (la // k, lo // k)
            bx = boxes.get(key)
            if bx is None:
                boxes[key] = [la, la, lo, lo]
            else:
                if la < bx[0]: bx[0] = la
                if la > bx[1]: bx[1] = la
                if lo < bx[2]: bx[2] = lo
                if lo > bx[3]: bx[3] = lo
        c = self._canvas
        for i, (la0, la1, lo0, lo1) in enumerate(boxes.values()):
            if i > 6000:                # safety cap on huge downloads
                break
            x0, y_top = self._geo_to_xy((la1 + 1) * b, lo0 * b)
            x1, _     = self._geo_to_xy((la1 + 1) * b, (lo1 + 1) * b)
            _,  y_bot = self._geo_to_xy(la0 * b, lo0 * b)
            c.create_rectangle(x0, y_top, x1, y_bot,
                               fill=color, stipple=stipple, outline="")

    def _draw_bg(self):
        v = self._v
        # Crop+LANCZOS+PhotoImage costs several ms and _redraw runs on every
        # hover-cell change -- reuse the last rendering while neither the view
        # nor the mosaic changed (pan/zoom invalidates the key).
        key = (id(self._bg_mosaic), self._cw, self._ch,
               round(v["lon_min"], 9), round(v["lon_max"], 9),
               round(v["lat_min"], 9), round(v["lat_max"], 9))
        if key == self._bg_draw_key and self._bg_photo is not None:
            self._canvas.create_image(*self._bg_off, image=self._bg_photo,
                                      anchor="nw")
            return
        lon_w_bg, lat_s_bg, lon_e_bg, lat_n_bg = self._bg_bounds
        bw, bh = self._bg_mosaic.size
        my_bg_n = _mercY(lat_n_bg); my_bg_s = _mercY(lat_s_bg)
        my_v_n  = self._merc_n();   my_v_s  = self._merc_s()

        def px_x(lon): return (lon - lon_w_bg) / (lon_e_bg - lon_w_bg) * bw
        def px_y(my):  return (my_bg_n - my)   / (my_bg_n - my_bg_s)   * bh

        cx0 = px_x(v["lon_min"]); cx1 = px_x(v["lon_max"])
        cy0 = px_y(my_v_n);       cy1 = px_y(my_v_s)
        cx0c = max(0, cx0); cy0c = max(0, cy0)
        cx1c = min(bw, cx1); cy1c = min(bh, cy1)
        if cx1c <= cx0c or cy1c <= cy0c: return

        span_x = cx1 - cx0; span_y = cy1 - cy0
        if span_x <= 0 or span_y <= 0: return
        off_x = max(0, int((cx0c - cx0) / span_x * self._cw))
        off_y = max(0, int((cy0c - cy0) / span_y * self._ch))
        dst_w = max(1, int((cx1c - cx0c) / span_x * self._cw))
        dst_h = max(1, int((cy1c - cy0c) / span_y * self._ch))

        crop    = self._bg_mosaic.crop((int(cx0c), int(cy0c), int(cx1c), int(cy1c)))
        resized = crop.resize((dst_w, dst_h), Image.LANCZOS)
        self._bg_photo = ImageTk.PhotoImage(resized)
        self._bg_draw_key = key
        self._bg_off = (off_x, off_y)
        self._canvas.create_image(off_x, off_y, image=self._bg_photo, anchor="nw")

    def _draw_grid(self):
        c, v  = self._canvas, self._v
        cd    = self._cell_deg()
        cd_u  = self._cell_deg_u()
        px_per_deg = self._cw / (v["lon_max"] - v["lon_min"])
        cell_px_w  = cd * px_per_deg
        grid_color = "#cccccc" if self._bg_mosaic else "#5a7a9a"

        la_start_u = (int(math.floor(v["lat_min"] * self._SBASE)) // cd_u) * cd_u
        lo_start_u = (int(math.floor(v["lon_min"] * self._SBASE)) // cd_u) * cd_u

        la_u = la_start_u
        while la_u / self._SBASE < v["lat_max"]:
            lat_sw = la_u / self._SBASE
            lo_u = lo_start_u
            while lo_u / self._SBASE < v["lon_max"]:
                lon_sw = lo_u / self._SBASE
                is_sel = self._coarse_is_sel(la_u, lo_u)
                is_hov = (la_u, lo_u) == self._hov

                # Canvas corners
                x0, y_top = self._geo_to_xy(lat_sw + cd, lon_sw)
                x1         = x0 + cell_px_w
                y_bot      = self._geo_to_xy(lat_sw, lon_sw)[1]

                if is_sel:
                    # The white fill is already drawn per base cell; here only
                    # the border, whose colour says what Start will do: yellow =
                    # resume an unfinished area, green = already complete
                    # (nothing to fetch), white = fresh download.
                    outline = (self.C_TODO if self._coarse_has(la_u, lo_u, "todo")
                               else self.C_DONE if self._coarse_has(la_u, lo_u, "done")
                               else self.C_SEL)
                    c.create_rectangle(x0, y_top, x1, y_bot,
                                       fill="", outline=outline, width=2)
                elif is_hov:
                    c.create_rectangle(x0, y_top, x1, y_bot,
                                       fill="#88bbff", stipple="gray25",
                                       outline="#aaddff", width=1)
                else:
                    c.create_rectangle(x0, y_top, x1, y_bot,
                                       fill="", outline=grid_color, width=1)

                h = y_bot - y_top
                if cell_px_w > 30 and h > 14:
                    s_la = (f"{lat_sw:+.2f}" if cd < 1 else f"{int(lat_sw):+d}")
                    s_lo = (f"{lon_sw:+.3f}" if cd < 1 else f"{int(lon_sw):+04d}")
                    fc   = "#101828" if is_sel else "white"
                    c.create_text(x0 + cell_px_w/2, (y_top+y_bot)/2,
                                  text=(f"{s_la}\n{s_lo}" if cell_px_w > 55
                                        else f"{s_la}{s_lo}"),
                                  fill=fc,
                                  font=("Courier New",
                                        max(6, int(min(cell_px_w, h) * 0.20))))
                lo_u += cd_u
            la_u += cd_u

    def _draw_axes(self):
        c, v = self._canvas, self._v
        lr   = v["lon_max"] - v["lon_min"]
        step = max(1, int(lr // 12))
        for lon in range(int(v["lon_min"]), int(v["lon_max"]) + 1, step):
            x = (lon - v["lon_min"]) / lr * self._cw
            c.create_text(x, 7, text=f"{lon:+d}", fill="#d0e4ff",
                          font=("Segoe UI", 7), anchor="n")
        my_n = self._merc_n(); my_s = self._merc_s()
        step2 = max(1, int((v["lat_max"] - v["lat_min"]) // 8))
        for lat in range(int(v["lat_min"]), int(v["lat_max"]) + 1, step2):
            y = (my_n - _mercY(lat)) / (my_n - my_s) * self._ch
            if 0 < y < self._ch:
                c.create_text(6, y, text=f"{lat:+d}", fill="#d0e4ff",
                              font=("Segoe UI", 7), anchor="w")

    def _draw_cities(self):
        c, v  = self._canvas, self._v
        my_n  = self._merc_n(); my_s = self._merc_s()
        lr    = v["lon_max"] - v["lon_min"]
        for name, lat, lon in CITIES:
            if not (v["lat_min"] <= lat <= v["lat_max"] and
                    v["lon_min"] <= lon <= v["lon_max"]):
                continue
            x = (lon - v["lon_min"]) / lr * self._cw
            y = (my_n - _mercY(lat)) / (my_n - my_s) * self._ch
            key = name == "Aschersleben"
            r   = 5 if key else 3
            c.create_oval(x-r, y-r, x+r, y+r,
                          fill="#ff9900" if key else "#ff4444",
                          outline="white", width=1)
            if lr < 80:
                c.create_text(x+r+3, y, text=name, fill="white",
                              font=("Segoe UI", 7), anchor="w")

    def _draw_bbox(self):
        """Draw dashed bounding box around the entire selection."""
        la_vals = [k[0] for k in self._sel]
        lo_vals = [k[1] for k in self._sel]
        lat_s = min(la_vals) / self._SBASE
        lat_n = (max(la_vals) + 1) / self._SBASE
        lon_w = min(lo_vals) / self._SBASE
        lon_e = (max(lo_vals) + 1) / self._SBASE
        x0, y_top = self._geo_to_xy(lat_n, lon_w)
        x1, y_bot = self._geo_to_xy(lat_s, lon_e)
        self._canvas.create_rectangle(x0, y_top, x1, y_bot,
                                      outline="#ffff00", width=2,
                                      fill="", dash=(5, 3))

    # ------------------------------------------------------------------
    # Canvas events
    # ------------------------------------------------------------------

    def _on_resize(self, e):
        changed = (self._cw, self._ch) != (max(e.width, 100), max(e.height, 100))
        self._cw = max(e.width,  100)
        self._ch = max(e.height, 100)
        if changed:
            self._enforce_aspect()        # new window aspect → un-stretch the view
        self._redraw()
        if changed:
            self._schedule_bg_refresh()   # canvas grew/shrank → match resolution

    def _sync_zoom_to_selection(self):
        """If the current selection overlaps already-downloaded data, snap the
        Zoom + HD controls to what that data was made with, so pressing Start
        continues (rather than silently overwriting at a different resolution).
        Returns the matched (zoom, px) or None."""
        if not self._done_info:
            return None
        cands = [self._done_info[k] for k in self._sel if k in self._done_info]
        if not cands:
            return None
        # If the selection spans areas downloaded at DIFFERENT settings, pick
        # the most common one (set iteration order used to make this random).
        z, px = max(set(cands), key=cands.count)
        if self._zoom_var.get() != z:
            self._zoom_var.set(z)
        if self._hd_var.get() != (px >= 4096):
            self._hd_var.set(px >= 4096)
        return (z, px)

    def _on_lclick(self, e):
        if e.state & 0x4:
            self._pan_last = (e.x, e.y); return
        coarse = self._xy_to_coarse(e.x, e.y)
        self._drag_start = coarse
        self._drag_mode  = "remove" if self._coarse_is_sel(*coarse) else "add"
        self._toggle_coarse(coarse, self._drag_mode)
        if self._drag_mode == "add":
            self._sync_zoom_to_selection()
        self._set_hover(coarse)
        self._update_sel_info(coarse)
        self._redraw()

    def _on_ldrag(self, e):
        if e.state & 0x4: return
        if self._drag_start is None: return
        coarse = self._xy_to_coarse(e.x, e.y)
        cd_u   = self._cell_deg_u()
        la0, lo0 = self._drag_start
        la1, lo1 = coarse
        la_min = min(la0, la1); la_max = max(la0, la1)
        lo_min = min(lo0, lo1); lo_max = max(lo0, lo1)
        la = la_min
        while la <= la_max:
            lo = lo_min
            while lo <= lo_max:
                self._toggle_coarse((la, lo), self._drag_mode)
                lo += cd_u
            la += cd_u
        if self._drag_mode == "add":
            self._sync_zoom_to_selection()
        self._set_hover(coarse)
        self._update_sel_info(coarse)
        self._redraw()

    def _on_lrelease(self, _):
        self._drag_start = None
        self._drag_mode  = None
        self._pan_last   = None

    def _on_motion(self, e):
        if e.state & 0x4: return
        coarse = self._xy_to_coarse(e.x, e.y)
        self._set_hover(coarse)
        self._update_sel_info(coarse)

    _RDRAG_THRESHOLD = 5   # px of movement before a right-press becomes a pan

    def _on_rpress(self, e):
        # Don't know yet if this is a deselect-click or a pan-drag; decide on
        # the first motion past the threshold (see _on_rmotion).
        self._rdrag_origin = (e.x, e.y)
        self._rdrag_is_pan = False
        self._pan_last = (e.x, e.y)

    def _on_rmotion(self, e):
        if self._rdrag_origin is None:
            self._rdrag_origin = (e.x, e.y); self._pan_last = (e.x, e.y); return
        if not self._rdrag_is_pan:
            ox, oy = self._rdrag_origin
            if abs(e.x - ox) + abs(e.y - oy) < self._RDRAG_THRESHOLD:
                return                              # still within click tolerance
            self._rdrag_is_pan = True                # movement confirmed -> pan
        self._on_pan(e)

    def _on_rrelease(self, e):
        if not self._rdrag_is_pan:
            # Never moved past the threshold -> treat as a plain deselect click.
            self._toggle_coarse(self._xy_to_coarse(e.x, e.y), "remove")
            self._update_sel_info(); self._redraw()
        self._rdrag_origin = None
        self._rdrag_is_pan = False
        self._pan_last = None

    def _on_pan_start(self, e):
        self._pan_last = (e.x, e.y)

    def _on_pan(self, e):
        if self._pan_last is None:
            self._pan_last = (e.x, e.y); return
        dx, dy = e.x - self._pan_last[0], e.y - self._pan_last[1]
        self._pan_last = (e.x, e.y)
        v = self._v
        dlon = -dx / self._cw * (v["lon_max"] - v["lon_min"])
        v["lon_min"] += dlon; v["lon_max"] += dlon
        my_n = self._merc_n(); my_s = self._merc_s()
        dmy  = dy / self._ch * (my_n - my_s)
        v["lat_max"] = _mercYInv(my_n + dmy)
        v["lat_min"] = _mercYInv(my_s + dmy)
        self._redraw()
        self._schedule_bg_refresh()

    def _on_scroll(self, e):
        factor = 0.65 if e.delta > 0 else 1.45
        self._zoom_centered(factor, e.x, e.y)

    def _set_hover(self, coarse_key):
        if coarse_key != self._hov:
            self._hov = coarse_key
            self._redraw()

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def _zoom_centered(self, factor, cx, cy):
        v = self._v
        lat_c, lon_c = self._xy_to_geo(cx, cy)   # geo under cursor before zoom
        lon_span = max(0.02, min(350.0, (v["lon_max"] - v["lon_min"]) * factor))
        # latitude span derived from longitude span + canvas aspect => no stretch
        # (x is in degrees of lon, y in mercator radians, hence radians())
        my_span = math.radians(lon_span) * self._ch / self._cw
        fx = cx / self._cw
        fy = cy / self._ch
        v["lon_min"] = lon_c - fx * lon_span
        v["lon_max"] = v["lon_min"] + lon_span
        my_cur = _mercY(max(-85.0, min(85.0, lat_c)))
        top, bot = _mercY(85.0), _mercY(-85.0)
        my_n = min(top, my_cur + fy * my_span)
        my_s = max(bot, my_n - my_span)
        v["lat_max"] = _mercYInv(my_n)
        v["lat_min"] = _mercYInv(my_s)
        self._redraw()
        self._schedule_bg_refresh()

    def _zoom_in(self):
        self._zoom_centered(0.6, self._cw/2, self._ch/2)

    def _zoom_out(self):
        self._zoom_centered(1.5, self._cw/2, self._ch/2)

    def _reset_view(self):
        self._v = dict(self._DV)
        self._enforce_aspect()
        self._redraw()
        self._schedule_bg_refresh()

    def _clear(self):
        self._sel.clear()
        self._active_var.set("(none)")
        self._sel_count.set("0 cells selected")
        self._sel_range.set(""); self._est_var.set("")
        self._redraw()

    # ------------------------------------------------------------------
    # Selection info panel
    # ------------------------------------------------------------------

    def _update_sel_info(self, coarse_key=None):
        cd = self._cell_deg()
        self._cell_size_var.set(f"Cell: {cd}° × {cd}°")
        if hasattr(self, "_mesh_info"):
            self._update_mesh_info()
        self._update_hd_info()

        if coarse_key is not None:
            la_u, lo_u = coarse_key
            lat_sw = la_u / self._SBASE
            lon_sw = lo_u / self._SBASE
            if cd < 1:
                s_la = f"{lat_sw:+.2f} … {lat_sw+cd:+.2f}"
                s_lo = f"{lon_sw:+.3f} … {lon_sw+cd:+.3f}"
            else:
                s_la = f"{int(lat_sw):+d} … {int(lat_sw+cd):+d}"
                s_lo = f"{int(lon_sw):+04d} … {int(lon_sw+cd):+04d}"
            self._active_var.set(f"lat {s_la}\nlon {s_lo}")

        n_base = len(self._sel)
        area   = n_base / (self._SBASE * self._SBASE)   # degrees²
        self._sel_count.set(
            f"{n_base} cells  ({area:.3f} deg²)")

        if self._sel:
            la_vals = [k[0] for k in self._sel]
            lo_vals = [k[1] for k in self._sel]
            lat_s = min(la_vals) / self._SBASE
            lat_n = (max(la_vals) + 1) / self._SBASE
            lon_w = min(lo_vals) / self._SBASE
            lon_e = (max(lo_vals) + 1) / self._SBASE
            self._sel_range.set(
                f"lat {lat_s:.2f} … {lat_n:.2f}  "
                f"lon {lon_w:.2f} … {lon_e:.2f}")
            zoom  = self._zoom_var.get()
            total = self._estimate_tiles(zoom)
            # Cells already complete cost nothing on a resume -- estimating the
            # WHOLE selection made continuing a half-done area look as expensive
            # as starting it over.
            n_done = sum(1 for k in self._sel if k in self._done_base)
            left   = 1.0 - n_done / len(self._sel)
            total  = int(total * left)
            s     = (f"{total/1e6:.1f} M" if total >= 1_000_000
                     else f"{total//1000} K" if total >= 1000
                     else str(total))
            hd    = self._hd_var.get()
            # Measured on this machine: ~60 KB per HD tile / ~18 KB per SD
            # tile; ~15 ms/tile sustained (the old 0.25 s/tile figure was
            # ~10x too pessimistic and scared off perfectly feasible areas).
            # While a run is live, use its real measured rate instead.
            bpt = 60_000 if hd else 18_000
            spt = 0.015 if hd else 0.012          # static fallback, s/tile
            if self._running and self._gen_start:
                el = time.time() - self._gen_start
                run_bytes = BW.read() - getattr(self, "_bw_run_base", 0)
                if el > 10 and run_bytes > 5_000_000:
                    spt = max(0.004, bpt / (run_bytes / el))
            dl_gb   = total * bpt / 1e9
            n_grp   = max(1, total // (GROUP_TILES * GROUP_TILES))
            disk_gb = n_grp * _dxt1_dds_size(4096 if hd else 2048) / 1e9
            secs = total * spt
            h, r = divmod(int(secs), 3600); m = r // 60
            t_str = (f"~{h}h {m:02d}m" if h
                     else f"~{m}m" if m else "<1m")
            self._est_var.set(
                f"~{s} sat-tiles  •  DL ~{dl_gb:.1f} GB  •  "
                f"Disk ~{disk_gb:.1f} GB  •  {t_str}")
        else:
            self._sel_range.set(""); self._est_var.set("")

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self._log_box.config(state=tk.NORMAL)
        self._log_box.insert(tk.END, msg + "\n")
        self._log_box.see(tk.END)
        self._log_box.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _start(self):
        if not self._sel:
            messagebox.showwarning("No selection",
                                   "Click cells on the map to select an area.")
            return
        out_dir = self._out_var.get().strip()
        if not out_dir:
            messagebox.showwarning("No output folder",
                                   "Please specify an output folder.")
            return
        os.makedirs(out_dir, exist_ok=True)

        # DoubleVar.get() raises TclError on garbage input, which would kill
        # this Tk callback silently -- Start would just do nothing.
        try:
            water_thr = float(self._water_var.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning(
                "Water threshold",
                "Ungültiger Wert bei 'Water threshold' (z.B. 0.60 eingeben).")
            return
        zoom  = self._zoom_var.get()

        # Resolution-conflict guard: if the selected area already holds data at
        # a DIFFERENT zoom or HD/SD than the current settings, that data would
        # be re-downloaded and overwritten. Ask before doing so.
        cur_px = 4096 if self._hd_var.get() else 2048
        conflict = None
        for k in self._sel:
            if k in self._done_info:
                z0, px0 = self._done_info[k]
                if z0 != zoom or px0 != cur_px:
                    conflict = (z0, px0); break
        if conflict:
            z0, px0 = conflict
            old = f"Zoom {z0} / {'HD' if px0 >= 4096 else 'SD'}"
            new = f"Zoom {zoom} / {'HD' if cur_px >= 4096 else 'SD'}"
            if not messagebox.askyesno(
                "Andere Auflösung überschreiben?",
                f"In diesem Bereich existieren bereits Texturen in {old}.\n"
                f"Du willst jetzt in {new} erzeugen.\n\n"
                f"Damit werden die vorhandenen Texturen komplett NEU "
                f"heruntergeladen und überschrieben (kein Fortsetzen).\n\n"
                f"Wirklich mit {new} überschreiben?"):
                return

        total = self._estimate_tiles(zoom)
        # What this run still has to FETCH (cells already complete are skipped
        # by the resume check) -- the basis for both warnings below, so
        # continuing a nearly-finished area is not judged like a fresh start.
        n_done = sum(1 for k in self._sel if k in self._done_base)
        new    = int(total * (1.0 - n_done / len(self._sel)))
        resume = n_done > 0

        if new > 300_000:
            if not messagebox.askyesno(
                "Very large area",
                f"~{new:,} satellite tiles at zoom {zoom}.\n"
                "Estimated time: several hours.\n\n"
                "Consider zoom 16 or less for large areas. Continue?"):
                return

        # Disk-space check: a full disk mid-run aborts groups with cryptic
        # errors (and used to bake broken textures). Warn BEFORE starting.
        try:
            n_grp   = max(1, new // (GROUP_TILES * GROUP_TILES))
            needed  = n_grp * _dxt1_dds_size(4096 if self._hd_var.get() else 2048)
            free    = shutil.disk_usage(out_dir).free
            if needed > free * 0.95:
                if not messagebox.askyesno(
                    "Wenig Speicherplatz",
                    f"Noch zu laden: ~{needed/1e9:.0f} GB\n"
                    f"Frei auf dem Ziellaufwerk: {free/1e9:.0f} GB\n\n"
                    f"Die Platte wird sehr wahrscheinlich volllaufen und der "
                    f"Download bricht mittendrin ab (Fortsetzen bleibt "
                    f"möglich).\n\nTrotzdem starten?"):
                    return
        except OSError:
            pass

        self._pb["maximum"] = max(total, 1)
        self._pb["value"]   = 0
        self._pct_var.set("0 %"); self._eta_var.set("")
        self._status_var.set("Computing scope …")
        self._log_box.config(state=tk.NORMAL)
        self._log_box.delete("1.0", tk.END)
        self._log_box.config(state=tk.DISABLED)
        self._start_btn.config(state=tk.DISABLED)
        self._pause_btn.config(state=tk.NORMAL)
        if resume:
            self._log(f"[i] Fortsetzen: {n_done} von {len(self._sel)} Zellen sind "
                      f"fertig, ~{new:,} Kacheln fehlen noch. Vorhandene Gruppen "
                      f"werden geprüft und übersprungen.")

        self._stop_event = threading.Event()
        self._gen_start  = time.time()
        self._running    = True
        self._bw_run_base = BW.read()   # live-rate estimate: this run's bytes only
        # Freeze what this run covers, so a pause/crash/close can be resumed
        # exactly -- including tiles it never got to, and regardless of any
        # map clicks made while it runs.
        self._run_snap = self._run_settings()
        self._session_save("running", self._run_snap)

        cfg = dict(
            sel_base_cells  = frozenset(self._sel),
            zoom            = zoom,
            water_threshold = water_thr,
            skip_water      = self._skip_var.get(),
            max_retries     = int(self._retry_var.get()),
            output_dir      = out_dir,
            base_mesh       = self._mesh_var.get(),
            mesh_spacing_m  = int(self._spacing_var.get()),
            group_workers   = int(self._pargrp_var.get()),
            hd              = self._hd_var.get(),
        )

        def worker():
            # try/finally: an unexpected error must NEVER leave the GUI stuck
            # with a dead Start button -- log it and re-enable the controls.
            ok = False
            try:
                ok = bool(run_generation(cfg,
                                         on_progress=self._cb_progress,
                                         on_log=self._cb_log,
                                         stop_event=self._stop_event))
            except Exception:
                import traceback
                self._cb_log("[!] Unerwarteter Fehler — Lauf abgebrochen "
                             "(bereits Geladenes bleibt gültig):\n"
                             + traceback.format_exc())
            finally:
                try:
                    self.root.after(0, lambda: self._on_done(failed=not ok))
                except Exception:
                    pass          # window already closed

        threading.Thread(target=worker, daemon=True).start()

    def _pause(self):
        """Clean, resumable stop: signal the worker to stop after the current
        in-flight groups. Everything downloaded stays valid on disk; finished
        cells go green, the rest of the paused area stays yellow -- and the
        selection is saved, so closing the app keeps that state."""
        if self._stop_event:
            self._stop_event.set()
            self._session_save("paused", getattr(self, "_run_snap", None))
            self._status_var.set("Pausiere … (bereits Geladenes bleibt erhalten)")
            self._pause_btn.config(state=tk.DISABLED)

    def _cb_progress(self, done, total, msg):
        # after() raises once the window is destroyed (user closed mid-run);
        # the worker thread must not die over a progress update.
        try:
            self.root.after(0, lambda: self._update_progress(done, total, msg))
        except Exception:
            pass

    def _cb_log(self, msg):
        try:
            self.root.after(0, lambda m=msg: self._log(m))
        except Exception:
            pass

    def _update_progress(self, done, total, msg):
        # done < 0 = post-download finalize phase (DSF assemble / DEM / compile):
        # no group count to show, so pulse the bar and just show the status so
        # the user sees it's still working, not frozen at 100%.
        if done < 0:
            if not self._pb_pulsing:
                self._pb.config(mode="indeterminate")
                self._pb.start(60)
                self._pb_pulsing = True
            self._pct_var.set("…")
            self._status_var.set(msg[:55])
            self._eta_var.set("arbeitet …")
            return
        if self._pb_pulsing:
            self._pb.stop()
            self._pb.config(mode="determinate")
            self._pb_pulsing = False
        pct = done / total * 100 if total else 0
        self._pb["value"]   = done
        self._pb["maximum"] = total
        self._pct_var.set(f"{pct:.1f} %")
        self._status_var.set(msg[:55])
        elapsed = time.time() - (self._gen_start or time.time())
        if done > 0 and elapsed > 1:
            eta = elapsed / done * (total - done)
            h, r = divmod(int(eta), 3600); m, s = divmod(r, 60)
            self._eta_var.set(f"ETA {h}h {m:02d}m" if h
                              else f"ETA {m}m {s:02d}s")

    def _on_done(self, failed=False):
        self._running = False
        if self._pb_pulsing:
            self._pb.stop()
            self._pb.config(mode="determinate")
            self._pb_pulsing = False
        self._start_btn.config(state=tk.NORMAL)
        self._pause_btn.config(state=tk.DISABLED)
        paused = bool(self._stop_event and self._stop_event.is_set())
        # A clean finish retires the session; a pause/failure keeps it so the
        # next launch can mark and continue the unfinished area.
        self._session_save("paused" if (paused or failed) else "done",
                           getattr(self, "_run_snap", None))
        # Re-derive the green/yellow marking from what is actually on disk.
        self._scan_existing()
        # Pause outranks "failed": cancelling sets both, and the user asked
        # for the pause.
        if paused:
            self._status_var.set("Pausiert — gelb = fehlt noch. Start setzt fort.")
            self._pct_var.set(""); self._eta_var.set("")
        elif failed:
            self._status_var.set("Fehler — siehe Log. Start setzt fort.")
            self._pct_var.set(""); self._eta_var.set("")
        else:
            self._pb["value"] = self._pb["maximum"]
            self._status_var.set("Done!")
            self._pct_var.set("100 %"); self._eta_var.set("")
            messagebox.showinfo(
                "Generation complete",
                "All scenery tiles generated.\n\n"
                "Copy the zOrtho4XP_… folder to:\n"
                "  X-Plane 12 / Custom Scenery /\n\n"
                "Then start X-Plane (or Reload Scenery).")


# ===========================================================================
#  Entry point
# ===========================================================================
if __name__ == "__main__":
    root = tk.Tk()
    OrthoApp(root)
    root.mainloop()
