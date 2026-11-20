# X-Plane Ortho Scenery Generator

A standalone, Ortho4XP-compatible satellite-imagery scenery generator for
**X-Plane 12**. Pick an area on an interactive map, choose a resolution, and it
downloads Apple Maps satellite tiles, compresses them to BC1/DXT1 DDS textures,
and writes the `.ter`/`.pol` + DSF files X-Plane needs — either as a **draped
overlay** over X-Plane's own terrain (safe for any partial selection) or as a
**base mesh** with real DEM elevation (full 1°×1° tiles only).

Run it:

```bash
python ortho_scenery_generator.py
```

The GUI opens automatically with a real satellite background, an adaptive
selection grid, a live progress bar/ETA and a bandwidth readout.

---

## Requirements

- **Python 3.8+**
- `requests`, `pillow` (required), `numpy` (strongly recommended — see below)
- **texconv** (optional but recommended) — Microsoft's DirectXTex BC1 compressor.
  Put `texconv.exe` in `~/Downloads`, `~/Desktop`, `C:\tools`, or anywhere on
  `PATH`. Download: https://github.com/microsoft/DirectXTex/releases
- **DSFTool** (optional) — compiles the text DSF to binary. Part of the X-Plane
  scenery tools; the generator looks in Downloads/Desktop and the X-Plane
  install.

```bash
pip install -r requirements.txt
```

### Texture compression — what actually happens

Textures are stored as **BC1/DXT1** — the most compact format X-Plane supports
(4 bits/pixel, 8× smaller than uncompressed RGBA) with a full mip chain.

- **With texconv:** best-quality BC1 (used whenever texconv is found, on any OS).
- **Without texconv but with numpy:** a built-in pure-Python BC1 encoder produces
  the *same compact format* (~1/8 the size of uncompressed, and a texture
  X-Plane loads reliably). Slightly lower quality than texconv.
- **Without texconv and without numpy:** falls back to uncompressed DDS (~8×
  larger, may not load). Install numpy or texconv to avoid this.

---

## Choosing resolution and zoom (storage & download)

Two independent knobs control texture detail: **Zoom level** and the **HD/SD**
switch. They interact in one simple way:

> **HD zoom N  ≡  SD zoom N+1**  — identical effective ground resolution.
> HD (`scale=2`) doubles the pixel density, which is exactly what going up one
> zoom level does. The **HD variant is always the better route**: same look,
> ~15–20 % less download, ~4× fewer files (lighter DSF, faster X-Plane load),
> same disk.

Each zoom step = **2× resolution = 4× data/time/disk**. The HD↔SD swap is the
only "free lunch"; choosing a zoom level trades real detail against cost.

Per **1°×1° tile** at ~51° N (HD; SD is exactly ¼ the disk):

| Zoom | cm/px (HD) | HD disk | SD disk | HD download |
|---|---|---|---|---|
| 14 | ~301 | 0.8 GB | 0.2 GB | ~0.3 GB |
| 15 | ~150 | 2.7 GB | 0.7 GB | ~1 GB |
| 16 | ~75 | 10.2 GB | 2.6 GB | ~4 GB |
| 17 | ~38 | 38.9 GB | 9.7 GB | ~13 GB |
| 18 | ~19 | 151.3 GB | 37.8 GB | ~52 GB |

**The sweet spot is ~37 cm/px (HD 17 / SD 18).** Apple's imagery is natively
only ~30–50 cm/px, so:

- **Zoom 17 HD** extracts almost all the *real* detail Apple has.
- **Zoom 18+** (~19 cm/px) is mostly interpolation — 4× the cost for little real
  gain. If you want ~19 cm/px, use **HD 18**, never SD 19.
- **Zoom 16 and below** drops below Apple's native detail: smaller/faster, but
  genuinely softer up close.

Rule of thumb by use case:

- **Airliners / high cruise:** zoom 16 is plenty — take the 4× saving.
- **GA, helicopters, airport surroundings, low flying:** zoom 17 HD.

---

## Saving storage without losing quality

- **HD → SD** (or one zoom lower): the big levers above.
- **Skip pure-water tiles** (checkbox): over open ocean, a group is replaced by a
  ~1 KB solid-colour placeholder instead of an 11 MB / 2.8 MB texture, *and* only
  1 preview tile is downloaded instead of 64. Quality-safe: a group is only
  skipped when it is **≥ 97 % water**, so coastlines and small islands are never
  flattened. Big win for coastal regions.
- **Select only what you need:** the map lets you select down to 0.05° cells, so
  you never have to grab a whole 1° tile for a single airport.

## Speeding up the download

The download gets slower from, in order: too few parallel connections, HTTP/1.1
per-request overhead, Apple rate-limiting (429), and compression back-pressure.
Practical levers:

- **Raise "Parallel groups"** (up to 24) on a fast connection — it scales the
  shared connection pool. Watch the **Mbit/s** readout: at your line's ceiling,
  more connections won't help; well below it, turn it up.
- **Skip pure-water tiles** for coastal areas (above).
- **Lower zoom / SD** — fewer/smaller tiles (see the resolution section).
- **Generate onto an SSD** — on a slow disk, per-group compression (one texconv
  process + a staging image per group) back-pressures and stalls the download.

---

## Resume & safety

- **Resumable:** closing or pausing mid-run is safe. Every finished texture is
  byte-validated on disk; a partial/corrupt file is re-downloaded. Green cells on
  the map are complete, yellow are started-but-incomplete; **Start/Resume**
  fetches only what's missing.
- **Overlay mode never leaves holes** — imagery drapes over X-Plane's existing
  terrain, so any partial selection is valid.
- **Base-mesh mode** replaces terrain and therefore requires *full* 1°×1° tiles
  (the generator refuses a partial selection in that mode).

## Install the output

Copy the generated `zOrtho4XP_…` folder into:

```
X-Plane 12/Custom Scenery/
```

then restart X-Plane (or Reload Scenery).

---

## Notes

- Imagery is fetched from Apple Maps via a public MapKit token flow; it is meant
  for personal use. Respect Apple's terms.
- The generator is a single file (`ortho_scenery_generator.py`); the engine
  (download, compression, DDS/DSF writers) is importable for testing — see
  `tests/`.
