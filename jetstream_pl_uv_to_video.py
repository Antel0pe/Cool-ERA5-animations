# jetstream_pl_uv_to_video_efficient.py
# Reads 4 ERA5 pressure-level NetCDFs (200/225/250/300 hPa) containing u/v,
# computes wind speed per level, picks the dominant level per pixel (streaming, no (L,H,W) stack),
# then encodes:
#   Hue  = dominant level (fixed palette)
#   Value/Brightness = dominant speed mapped from MIN_SPEED..MAX_SPEED
#   Saturation = fixed
#
# Perf/RAM changes vs original:
# - No np.stack((L,H,W)) per frame: streaming argmax across levels
# - open_dataset with chunks={"time": 1}, cache=False, explicit engine
# - keep only u/v variables (drop everything else)
# - minimize copies: float32, astype(copy=False), in-place sqrt
#
# Output:
#   (A) write one RGBA frame per hour to ./jetstream_pl_images
#   (B) stream frames directly into an .mp4 via ffmpeg (no PNGs written)

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict

import numpy as np
import xarray as xr
from PIL import Image
from dotenv import load_dotenv  # optional


# ----------------------------
# Constants you edit
# ----------------------------

DATA_DIR_ENV = "ERA5_CACHE_DIR"

FILES_BY_LEVEL_HPA: Dict[int, str] = {
    200: "era5_uv_pl_2020_H1_200hPa_0p25.nc",
    225: "era5_uv_pl_2020_H1_225hPa_0p25.nc",
    250: "250hpa_uv_2020.nc",
    300: "era5_uv_pl_2020_H1_300hPa_0p25.nc",
}

U_NAME = "u"
V_NAME = "v"

START_TIME = "2020-01-01T00:00:00"
END_TIME   = "2020-01-10T23:00:00"

STREAM_TO_VIDEO = True

FPS = 30
OUT_VIDEO = Path(f"./test_outputs/jetstream_pl_dominantLevel_{START_TIME[:10]}_{END_TIME[:10]}_{FPS}fps.mp4")

IMG_EXT = "png"

MIN_SPEED = 40.0
MAX_SPEED = 70.0

FIXED_SATURATION = 0.95
ALPHA_BELOW_MIN = True

LOG_EVERY_N = 168  # ~weekly


# ----------------------------
# Helpers
# ----------------------------

def resolve_input_paths() -> Dict[int, Path]:
    load_dotenv()
    base = os.environ.get(DATA_DIR_ENV)
    if not base:
        raise EnvironmentError(f"Env var {DATA_DIR_ENV} is not set.")
    basep = Path(base)

    out: Dict[int, Path] = {}
    for lvl, fn in FILES_BY_LEVEL_HPA.items():
        p = basep / fn
        if not p.exists():
            raise FileNotFoundError(f"Input NetCDF not found for {lvl} hPa: {p}")
        out[lvl] = p
    return out


def find_coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.variables:
            return name
    raise KeyError(
        f"Could not find any of these coordinates: {candidates}. "
        f"Available coords: {list(ds.coords)} / vars: {list(ds.variables)}"
    )


def find_time_name(ds: xr.Dataset) -> str:
    for name in ("time", "valid_time"):
        if name in ds.coords or name in ds.variables:
            return name
    raise KeyError(
        f"No time coordinate found. Available coords: {list(ds.coords)} "
        f"vars: {list(ds.variables)}"
    )


def find_level_name(ds: xr.Dataset) -> str | None:
    for name in ("pressure_level", "plev", "level", "isobaricInhPa"):
        if name in ds.dims or name in ds.coords:
            return name
    return None


def start_ffmpeg_writer(width: int, height: int, fps: int, out_path: Path) -> subprocess.Popen:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(out_path),
    ]

    print("\nRunning ffmpeg:")
    print("  " + " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def even_crop_rgba(rgba: np.ndarray) -> np.ndarray:
    h, w, _ = rgba.shape
    if (h % 2) == 1:
        rgba = rgba[:-1, :, :]
    if (w % 2) == 1:
        rgba = rgba[:, :-1, :]
    return rgba


def hsv_to_rgba(h: np.ndarray, s: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Vectorized HSV (0..1) -> RGBA uint8.
    h,s,v,a: (H,W) float32 in [0,1]
    """
    h = np.mod(h, 1.0)
    s = np.clip(s, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    a = np.clip(a, 0.0, 1.0)

    c = v * s
    hp = h * 6.0
    x = c * (1.0 - np.abs(np.mod(hp, 2.0) - 1.0))
    m = v - c

    zeros = np.zeros_like(h, dtype=np.float32)

    r1 = np.empty_like(h, dtype=np.float32)
    g1 = np.empty_like(h, dtype=np.float32)
    b1 = np.empty_like(h, dtype=np.float32)

    m0 = (0.0 <= hp) & (hp < 1.0)
    m1_ = (1.0 <= hp) & (hp < 2.0)
    m2 = (2.0 <= hp) & (hp < 3.0)
    m3 = (3.0 <= hp) & (hp < 4.0)
    m4 = (4.0 <= hp) & (hp < 5.0)
    m5 = (5.0 <= hp) & (hp < 6.0)

    r1[m0], g1[m0], b1[m0] = c[m0], x[m0], zeros[m0]
    r1[m1_], g1[m1_], b1[m1_] = x[m1_], c[m1_], zeros[m1_]
    r1[m2], g1[m2], b1[m2] = zeros[m2], c[m2], x[m2]
    r1[m3], g1[m3], b1[m3] = zeros[m3], x[m3], c[m3]
    r1[m4], g1[m4], b1[m4] = x[m4], zeros[m4], c[m4]
    r1[m5], g1[m5], b1[m5] = c[m5], zeros[m5], x[m5]

    other = ~(m0 | m1_ | m2 | m3 | m4 | m5)
    if np.any(other):
        r1[other], g1[other], b1[other] = zeros[other], zeros[other], zeros[other]

    r = (r1 + m)
    g = (g1 + m)
    b = (b1 + m)

    rgba = np.stack([r, g, b, a], axis=-1)
    return (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)


def level_to_hue(level_hpa: int) -> float:
    mapping = {
        200: 0.83,  # magenta
        225: 0.05,  # orange/red
        250: 0.33,  # green
        300: 0.55,  # cyan/blue
    }
    if level_hpa not in mapping:
        raise KeyError(f"No hue mapping for level {level_hpa} hPa.")
    return mapping[level_hpa]


def normalize_speed_to_value(speed: np.ndarray, min_speed: float, max_speed: float) -> np.ndarray:
    s = speed.astype(np.float32, copy=False)
    t = (s - float(min_speed)) / (float(max_speed) - float(min_speed))
    return np.clip(t, 0.0, 1.0)


def load_level_dataset(path: Path) -> xr.Dataset:
    """
    Perf/RAM:
    - chunks={'time': 1} to read one timestep at a time lazily
    - cache=False to avoid holding decoded arrays in memory
    - select only u/v to drop other vars immediately
    """
    ds = xr.open_dataset(
        path,
        engine="netcdf4",      # try "h5netcdf" if it benchmarks faster for your files
        decode_times=True,
        chunks={"time": 1},
        cache=False,
    )

    missing = [vn for vn in (U_NAME, V_NAME) if vn not in ds.variables]
    if missing:
        raise KeyError(f"{path.name}: missing variables {missing}. Vars: {list(ds.variables)}")

    # Drop everything except u/v ASAP
    ds = ds[[U_NAME, V_NAME]]
    return ds


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    paths = resolve_input_paths()
    out_dir = Path("./jetstream_pl_images")

    levels = sorted(paths.keys())

    # Open datasets (lazy, chunked, minimal vars)
    dsets: Dict[int, xr.Dataset] = {lvl: load_level_dataset(paths[lvl]) for lvl in levels}

    # Determine lat orientation using one dataset
    ds0 = dsets[levels[0]]
    lat_name = find_coord_name(ds0, ("latitude", "lat"))
    lat_vals = ds0[lat_name].values
    lat_ascending = bool(lat_vals[0] < lat_vals[-1])

    # Slice time & normalize dims/names per level, keep as dask-backed arrays
    u_by_level: Dict[int, xr.DataArray] = {}
    v_by_level: Dict[int, xr.DataArray] = {}

    for lvl in levels:
        ds = dsets[lvl]
        time_name = find_time_name(ds)

        u = ds[U_NAME].sel(**{time_name: slice(START_TIME, END_TIME)})
        v = ds[V_NAME].sel(**{time_name: slice(START_TIME, END_TIME)})

        # normalize time coordinate name
        if time_name != "time":
            u = u.rename({time_name: "time"})
            v = v.rename({time_name: "time"})

        # squeeze level dim if present and size==1
        level_name = find_level_name(ds)
        if level_name is not None and level_name in u.dims and u.sizes.get(level_name, 0) == 1:
            u = u.isel(**{level_name: 0})
            v = v.isel(**{level_name: 0})

        u_by_level[lvl] = u
        v_by_level[lvl] = v

    # Reference times
    ref_lvl = levels[0]
    times = u_by_level[ref_lvl]["time"].values
    if len(times) == 0:
        raise ValueError("No timesteps found in the requested time range.")

    # Ensure time alignment across levels
    for lvl in levels:
        u = u_by_level[lvl]
        v = v_by_level[lvl]
        if u.sizes.get("time") != v.sizes.get("time"):
            raise ValueError(f"{lvl} hPa: u and v time lengths differ.")
        if not np.array_equal(u["time"].values, v["time"].values):
            raise ValueError(f"{lvl} hPa: u and v time coords differ.")
        if not np.array_equal(u["time"].values, times):
            raise ValueError(f"{lvl} hPa: time coords differ from reference {ref_lvl} hPa.")

    # Initialize IO targets and per-frame working buffers
    ffmpeg_proc = None
    ffmpeg_stdin = None

    # Determine shape from one timestep (forces only one chunk read)
    u0 = u_by_level[ref_lvl].isel(time=0).values
    v0 = v_by_level[ref_lvl].isel(time=0).values
    if u0.ndim != 2 or v0.ndim != 2:
        raise ValueError(f"Expected 2D (lat,lon); got {u0.shape} and {v0.shape}")
    h, w = u0.shape

    # Precompute hue list
    hues = np.array([level_to_hue(lvl) for lvl in levels], dtype=np.float32)

    # Precompute constant saturation field once
    s_field = np.full((h, w), float(FIXED_SATURATION), dtype=np.float32)

    # Pre-allocate streaming argmax buffers once
    best_speed = np.empty((h, w), dtype=np.float32)
    best_idx = np.empty((h, w), dtype=np.uint8)

    if STREAM_TO_VIDEO:
        h2 = h - (h % 2)
        w2 = w - (w % 2)
        ffmpeg_proc = start_ffmpeg_writer(width=w2, height=h2, fps=FPS, out_path=OUT_VIDEO)
        if ffmpeg_proc.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin.")
        ffmpeg_stdin = ffmpeg_proc.stdin
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, t in enumerate(times):
        # Streaming dominant-level selection: no (L,H,W) stack
        best_speed.fill(-np.inf)
        best_idx.fill(0)

        for li, lvl in enumerate(levels):
            # Pull one time slice; keep float32 with minimal copying
            uu = u_by_level[lvl].isel(time=i).values.astype(np.float32, copy=False)
            vv = v_by_level[lvl].isel(time=i).values.astype(np.float32, copy=False)

            # speed = sqrt(u^2 + v^2) with minimal temporaries
            sp = uu * uu
            sp += vv * vv
            np.sqrt(sp, out=sp)  # reuse sp as speed

            m = sp > best_speed
            if np.any(m):
                best_speed[m] = sp[m]
                best_idx[m] = li

        dom_speed = best_speed  # (H,W) float32
        dom_idx = best_idx.astype(np.int32, copy=False)

        h_field = hues[dom_idx]  # (H,W) float32

        v_field = normalize_speed_to_value(dom_speed, MIN_SPEED, MAX_SPEED)

        if ALPHA_BELOW_MIN:
            a_field = (dom_speed >= MIN_SPEED).astype(np.float32, copy=False)
        else:
            a_field = np.ones_like(v_field, dtype=np.float32)

        # NaNs -> transparent
        nan_mask = ~np.isfinite(dom_speed)
        if np.any(nan_mask):
            a_field = a_field.copy()  # only copy if we need to edit
            v_field = v_field.copy()
            a_field[nan_mask] = 0.0
            v_field[nan_mask] = 0.0

        rgba = hsv_to_rgba(h_field, s_field, v_field, a_field)

        # Flip to make north-up if latitude ascending (-90..90)
        if lat_ascending:
            rgba = np.flipud(rgba)

        if STREAM_TO_VIDEO:
            rgba = even_crop_rgba(rgba)
            ffmpeg_stdin.write(rgba.tobytes())
        else:
            img = Image.fromarray(rgba, mode="RGBA")
            ts = np.datetime_as_string(t, unit="s").replace(":", "").replace("-", "")
            out_path = out_dir / f"frame_{i:05d}_{ts}.{IMG_EXT}"
            img.save(out_path)

        if (i % LOG_EVERY_N) == 0:
            if STREAM_TO_VIDEO:
                print(f"[{i:05d}/{len(times)-1:05d}] streamed frame {i:05d}")
            else:
                print(f"[{i:05d}/{len(times)-1:05d}] wrote {out_path.name}")

    # Close ffmpeg / finalize
    if STREAM_TO_VIDEO:
        assert ffmpeg_proc is not None and ffmpeg_stdin is not None
        ffmpeg_stdin.close()
        ret = ffmpeg_proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with code {ret}")
        print(f"\nWrote video: {OUT_VIDEO.resolve()}")
    else:
        print(f"Done. Wrote {len(times)} frames to: {out_dir.resolve()}")

    # Close datasets explicitly (avoid lingering file handles)
    for ds in dsets.values():
        ds.close()


if __name__ == "__main__":
    main()
