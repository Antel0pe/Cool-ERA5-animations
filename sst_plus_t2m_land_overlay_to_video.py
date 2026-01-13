# sst_plus_t2m_land_overlay_to_video.py
# Reads SST + 2m temp NetCDFs, composites them:
#   - use SST over ocean (where SST is finite / defined)
#   - use 2m temp over land (where SST is NaN / not defined)
# Then either:
#   (A) writes one RGBA frame per hour to ./<OUT_PREFIX>_images
#   (B) streams frames directly into an .mp4 via ffmpeg (no PNGs written)
#
# Matches the conventions/structure of your two scripts.

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image
from dotenv import load_dotenv  # optional


# ----------------------------
# Constants you edit
# ----------------------------

# Env var that points to the directory containing the NetCDFs
DATA_DIR_ENV = "ERA5_CACHE_DIR"

# Input NetCDF filenames (inside DATA_DIR_ENV)
SST_NC_FILENAME = "sst_2020.nc"
T2M_NC_FILENAME = "t2m_2020.nc"

# Variable names inside each file
SST_VAR_NAME = "sst"   # common candidates: "sst"
T2M_VAR_NAME = "2t"    # common candidates: "2t"

# Time range (inclusive). Use ISO strings.
START_TIME = "2020-01-01T00:00:00"
END_TIME   = "2020-01-10T23:00:00"

# Mode:
#   False -> write PNGs to ./<OUT_PREFIX>_images
#   True  -> stream frames straight to MP4 via ffmpeg (no PNGs)
STREAM_TO_VIDEO = True

# Output naming
OUT_PREFIX = "sst_ocean_t2m_land"

# Video settings (used when STREAM_TO_VIDEO = True)
FPS = 30
OUT_VIDEO = Path(f"./test_outputs/{OUT_PREFIX}_{START_TIME[:10]}_{END_TIME[:10]}_{FPS}fps.mp4")

# Output image format (used when STREAM_TO_VIDEO = False)
IMG_EXT = "png"

# Color scales (assume Kelvin like ERA5)
# SST: ocean temps
# SST_MIN_C = -2.0
# SST_MAX_C = 35.0
SST_MIN_C = -40.0
SST_MAX_C = 40.0
# 2m temp: air temps
T2M_MIN_C = -40.0
T2M_MAX_C = 40.0

# Progress logging
LOG_EVERY_N = 168  # ~weekly at hourly cadence


# ----------------------------
# Helpers
# ----------------------------

def resolve_input_paths() -> tuple[Path, Path]:
    load_dotenv()

    base = os.environ.get(DATA_DIR_ENV)
    if not base:
        raise EnvironmentError(f"Env var {DATA_DIR_ENV} is not set.")
    base_dir = Path(base)

    return base_dir / SST_NC_FILENAME, base_dir / T2M_NC_FILENAME


def find_coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.variables:
            return name
    raise KeyError(f"Could not find any of these coordinates: {candidates}. Available: {list(ds.coords)}")


def temp_to_rgba(temp_k: np.ndarray, min_c: float, max_c: float) -> np.ndarray:
    """
    Map temperature to RGB with a perceptually sane scheme:
      cold -> light blue
      midpoint -> white
      hot -> soft red / orange

    Returns uint8 RGBA array (H, W, 4).
    """
    temp_c = temp_k.astype(np.float32) - 273.15

    # Normalize to [0, 1]
    t = (temp_c - float(min_c)) / (float(max_c) - float(min_c))
    t = np.clip(t, 0.0, 1.0)

    # Define endpoints (tuned for luminance balance)
    cold = np.array([80, 140, 255], dtype=np.float32)   # light blue / cyan
    mid  = np.array([255, 255, 255], dtype=np.float32)  # white
    hot  = np.array([255, 120, 80], dtype=np.float32)   # soft red / orange

    rgb = np.empty(t.shape + (3,), dtype=np.float32)

    # Below midpoint: cold -> white
    mask_cold = t <= 0.5
    tc = t[mask_cold] / 0.5
    rgb[mask_cold] = cold * (1.0 - tc[..., None]) + mid * tc[..., None]

    # Above midpoint: white -> hot
    mask_hot = ~mask_cold
    th = (t[mask_hot] - 0.5) / 0.5
    rgb[mask_hot] = mid * (1.0 - th[..., None]) + hot * th[..., None]

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    a = np.full(t.shape, 255, dtype=np.uint8)

    # Handle NaNs: transparent
    nan_mask = ~np.isfinite(temp_c)
    if np.any(nan_mask):
        rgb[nan_mask] = 0
        a[nan_mask] = 0

    return np.dstack([rgb, a])


def even_crop_rgba(rgba: np.ndarray) -> np.ndarray:
    """
    Ensure width/height are even for H.264/yuv420p by dropping the last
    row/col if needed.
    rgba: (H, W, 4)
    """
    h, w, _ = rgba.shape
    if (h % 2) == 1:
        rgba = rgba[:-1, :, :]
    if (w % 2) == 1:
        rgba = rgba[:, :-1, :]
    return rgba


def start_ffmpeg_writer(width: int, height: int, fps: int, out_path: Path) -> subprocess.Popen:
    """
    Launch ffmpeg and feed raw RGBA frames to stdin.
    """
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


def main() -> None:
    sst_path, t2m_path = resolve_input_paths()
    if not sst_path.exists():
        raise FileNotFoundError(f"SST NetCDF not found: {sst_path}")
    if not t2m_path.exists():
        raise FileNotFoundError(f"2m temp NetCDF not found: {t2m_path}")

    out_dir = Path(f"./{OUT_PREFIX}_images")

    ds_sst = xr.open_dataset(sst_path, decode_times=True)
    ds_t2m = xr.open_dataset(t2m_path, decode_times=True)

    if SST_VAR_NAME not in ds_sst.variables:
        raise KeyError(f"SST var '{SST_VAR_NAME}' not found. Variables: {list(ds_sst.variables)}")
    if T2M_VAR_NAME not in ds_t2m.variables:
        raise KeyError(f"2m temp var '{T2M_VAR_NAME}' not found. Variables: {list(ds_t2m.variables)}")

    # Identify latitude coord and decide if we need to flip north-up.
    lat_name_sst = find_coord_name(ds_sst, ("latitude", "lat"))
    lat_name_t2m = find_coord_name(ds_t2m, ("latitude", "lat"))

    lat_vals = ds_sst[lat_name_sst].values
    lat_ascending = bool(lat_vals[0] < lat_vals[-1])

    # Time slices
    da_sst = ds_sst[SST_VAR_NAME].sel(time=slice(START_TIME, END_TIME))
    da_t2m = ds_t2m[T2M_VAR_NAME].sel(time=slice(START_TIME, END_TIME))

    times_sst = da_sst["time"].values
    times_t2m = da_t2m["time"].values

    if len(times_sst) == 0:
        raise ValueError("No SST timesteps found in the requested time range.")
    if len(times_t2m) == 0:
        raise ValueError("No 2m temp timesteps found in the requested time range.")

    # Sanity: require same number of timesteps + aligned times (common for ERA5 hourly)
    if len(times_sst) != len(times_t2m):
        raise ValueError(f"Time length mismatch: SST={len(times_sst)} vs T2M={len(times_t2m)}")
    if not np.array_equal(times_sst, times_t2m):
        # Don’t try to get clever; fail loudly to avoid silent misalignment.
        raise ValueError("SST and T2M time coordinates are not identical. Ensure both are the same hourly grid.")

    # Sanity: require identical spatial shape
    first_sst = da_sst.isel(time=0).values
    first_t2m = da_t2m.isel(time=0).values
    if first_sst.ndim != 2 or first_t2m.ndim != 2:
        raise ValueError(f"Expected 2D (lat,lon) per timestep; got SST={first_sst.shape}, T2M={first_t2m.shape}")
    if first_sst.shape != first_t2m.shape:
        raise ValueError(f"Spatial shape mismatch: SST={first_sst.shape} vs T2M={first_t2m.shape}")

    ffmpeg_proc = None
    ffmpeg_stdin = None

    if STREAM_TO_VIDEO:
        h, w = first_sst.shape
        h2 = h - (h % 2)
        w2 = w - (w % 2)
        ffmpeg_proc = start_ffmpeg_writer(width=w2, height=h2, fps=FPS, out_path=OUT_VIDEO)
        if ffmpeg_proc.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin.")
        ffmpeg_stdin = ffmpeg_proc.stdin
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Main loop
    for i, t in enumerate(times_sst):
        sst = da_sst.sel(time=t).values
        t2m = da_t2m.sel(time=t).values

        # Colorize separately
        rgba_sst = temp_to_rgba(sst, SST_MIN_C, SST_MAX_C)
        rgba_t2m = temp_to_rgba(t2m, T2M_MIN_C, T2M_MAX_C)

        # Composite:
        # - SST is defined on ocean; land is NaN (or masked). Use SST where finite, else use T2M.
        # IMPORTANT: use *the original SST field* to decide the mask (not rgba alpha),
        # because your temp_to_rgba makes NaNs transparent.
        ocean_mask = np.isfinite(sst)  # True where SST exists

        rgba = rgba_t2m.copy()
        rgba[ocean_mask] = rgba_sst[ocean_mask]

        # Flip so north is up if latitude is ascending.
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
                print(f"[{i:05d}/{len(times_sst)-1:05d}] streamed frame {i:05d}")
            else:
                print(f"[{i:05d}/{len(times_sst)-1:05d}] wrote {out_path.name}")

    if STREAM_TO_VIDEO:
        assert ffmpeg_proc is not None and ffmpeg_stdin is not None
        ffmpeg_stdin.close()
        ret = ffmpeg_proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with code {ret}")
        print(f"\nWrote video: {OUT_VIDEO.resolve()}")
    else:
        print(f"Done. Wrote {len(times_sst)} frames to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
