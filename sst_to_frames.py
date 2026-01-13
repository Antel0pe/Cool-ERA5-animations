# sst_to_frames.py
# Reads sst_2020.nc (or whatever you set), extracts SST, and either:
#   (A) writes one RGB frame per hour to ./<VAR_NAME>_images
#   (B) streams frames directly into an .mp4 via ffmpeg (no PNGs written)

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

# Env var that points to the directory containing the NetCDF
DATA_DIR_ENV = "ERA5_CACHE_DIR"

# Input NetCDF file name (inside DATA_DIR_ENV)
NC_FILENAME = "sst_2020.nc"     # <--- set this to your SST netcdf filename

# Variable name inside the file (common candidates: "sst", "SSTK", etc.)
VAR_NAME = "sst"               # <--- set to the variable name in your file

# Time range (inclusive). Use ISO strings.
START_TIME = "2020-01-01T00:00:00"
END_TIME   = "2020-06-10T23:00:00"

# Mode:
#   False -> write PNGs to ./<VAR_NAME>_images
#   True  -> stream frames straight to MP4 via ffmpeg (no PNGs)
STREAM_TO_VIDEO = True

# Video settings (used when STREAM_TO_VIDEO = True)
FPS = 30
OUT_VIDEO = Path(f"./test_outputs/{VAR_NAME}_{START_TIME[:10]}_{END_TIME[:10]}_{FPS}fps.mp4")

# Output image format (used when STREAM_TO_VIDEO = False)
IMG_EXT = "png"

# Color scale (assumes SST is Kelvin, like ERA5).
# Tune for ocean temps (°C). This is a reasonable global range.
MIN_C = -2.0
MAX_C =  35.0

# Progress logging
LOG_EVERY_N = 168  # ~weekly at hourly cadence


# ----------------------------
# Helpers
# ----------------------------

def resolve_input_path() -> Path:
    load_dotenv()

    base = os.environ.get(DATA_DIR_ENV)
    if not base:
        raise EnvironmentError(f"Env var {DATA_DIR_ENV} is not set.")
    base_dir = Path(base)

    return base_dir / NC_FILENAME


def find_coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.variables:
            return name
    raise KeyError(f"Could not find any of these coordinates: {candidates}. Available: {list(ds.coords)}")


# def temp_to_rgba(temp_k: np.ndarray, min_c: float, max_c: float) -> np.ndarray:
#     """
#     Map temperature to RGB:
#       cold -> blue, hot -> red
#     Returns uint8 RGBA array (H, W, 4).
#     """
#     temp_c = temp_k.astype(np.float32) - 273.15

#     # Normalize to [0, 1]
#     t = (temp_c - float(min_c)) / (float(max_c) - float(min_c))
#     t = np.clip(t, 0.0, 1.0)

#     r = (t * 255.0).astype(np.uint8)
#     g = np.zeros_like(r, dtype=np.uint8)
#     b = ((1.0 - t) * 255.0).astype(np.uint8)
#     a = np.full_like(r, 255, dtype=np.uint8)

#     # Handle NaNs: make them transparent black
#     nan_mask = ~np.isfinite(temp_c)
#     if np.any(nan_mask):
#         r[nan_mask] = 0
#         g[nan_mask] = 0
#         b[nan_mask] = 0
#         a[nan_mask] = 0

#     return np.stack([r, g, b, a], axis=-1)

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
    mid  = np.array([255, 255, 255], dtype=np.float32) # white
    hot  = np.array([255, 120, 80], dtype=np.float32)  # soft red / orange

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
    in_path = resolve_input_path()
    if not in_path.exists():
        raise FileNotFoundError(f"Input NetCDF not found: {in_path}")

    out_dir = Path(f"./{VAR_NAME}_images")

    ds = xr.open_dataset(in_path, decode_times=True)

    if VAR_NAME not in ds.variables:
        raise KeyError(f"Variable '{VAR_NAME}' not found. Variables: {list(ds.variables)}")

    # Identify latitude coord and decide if we need to flip north-up.
    lat_name = find_coord_name(ds, ("latitude", "lat"))
    lat_vals = ds[lat_name].values
    lat_ascending = bool(lat_vals[0] < lat_vals[-1])

    da = ds[VAR_NAME].sel(time=slice(START_TIME, END_TIME))
    times = da["time"].values
    if len(times) == 0:
        raise ValueError("No timesteps found in the requested time range.")

    ffmpeg_proc = None
    ffmpeg_stdin = None

    if STREAM_TO_VIDEO:
        first = da.isel(time=0).values
        if first.ndim != 2:
            raise ValueError(f"Expected 2D (lat,lon) per timestep; got shape {first.shape}")

        h, w = first.shape
        h2 = h - (h % 2)
        w2 = w - (w % 2)

        ffmpeg_proc = start_ffmpeg_writer(width=w2, height=h2, fps=FPS, out_path=OUT_VIDEO)
        if ffmpeg_proc.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin.")
        ffmpeg_stdin = ffmpeg_proc.stdin
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, t in enumerate(times):
        frame = da.sel(time=t).values

        rgba = temp_to_rgba(frame, MIN_C, MAX_C)

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
                print(f"[{i:05d}/{len(times)-1:05d}] streamed frame {i:05d}")
            else:
                print(f"[{i:05d}/{len(times)-1:05d}] wrote {out_path.name}")

    if STREAM_TO_VIDEO:
        assert ffmpeg_proc is not None and ffmpeg_stdin is not None
        ffmpeg_stdin.close()
        ret = ffmpeg_proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with code {ret}")
        print(f"\nWrote video: {OUT_VIDEO.resolve()}")
    else:
        print(f"Done. Wrote {len(times)} frames to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
