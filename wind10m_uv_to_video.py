# wind10m_uv_to_video.py
# Reads wind10mUV_2020.nc, extracts 10u/10v, computes wind speed, and either:
#   (A) writes one RGBA frame per hour to ./wind10m_speed_images
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
NC_FILENAME = "wind10mUV_2020.nc"

# Variable names inside the file
U_NAME = "10u"
V_NAME = "10v"

# Time range (inclusive). Use ISO strings.
START_TIME = "2020-01-01T00:00:00"
END_TIME   = "2020-01-10T23:00:00"

# Mode:
#   False -> write PNGs to ./wind10m_speed_images
#   True  -> stream frames straight to MP4 via ffmpeg (no PNGs)
STREAM_TO_VIDEO = True

# Video settings (used when STREAM_TO_VIDEO = True)
FPS = 30
OUT_VIDEO = Path(f"./test_outputs/wind10m_speed_{START_TIME[:10]}_{END_TIME[:10]}_{FPS}fps.mp4")

# Output image format (used when STREAM_TO_VIDEO = False)
IMG_EXT = "png"

# Speed scale (m/s). 0 -> white, MAX -> blue.
# Good starting points:
#   typical global 10m winds: 0..20 m/s (storms can exceed that)
MIN_SPEED = 0.0
MAX_SPEED = 25.0

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
    return Path(base) / NC_FILENAME


def find_coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.variables:
            return name
    raise KeyError(f"Could not find any of these coordinates: {candidates}. Available: {list(ds.coords)}")


def speed_to_rgba(speed: np.ndarray, min_speed: float, max_speed: float) -> np.ndarray:
    """
    Map wind speed (m/s) to RGBA:
      0 m/s -> white
      higher -> increasingly blue (white -> blue)

    Returns uint8 RGBA array (H, W, 4).
    """
    s = speed.astype(np.float32)

    # Normalize to [0, 1]
    t = (s - float(min_speed)) / (float(max_speed) - float(min_speed))
    t = np.clip(t, 0.0, 1.0)

    # White -> Blue ramp
    # r,g drop from 255 to 0 as t increases; b stays 255.
    rg = ((1.0 - t) * 255.0).astype(np.uint8)
    b = np.full_like(rg, 255, dtype=np.uint8)
    a = np.full_like(rg, 255, dtype=np.uint8)

    # Handle NaNs: transparent
    nan_mask = ~np.isfinite(s)
    if np.any(nan_mask):
        rg[nan_mask] = 0
        b[nan_mask] = 0
        a[nan_mask] = 0

    r = rg
    g = rg
    return np.stack([r, g, b, a], axis=-1)


def even_crop_rgba(rgba: np.ndarray) -> np.ndarray:
    """
    Ensure width/height are even for H.264/yuv420p by dropping the last row/col if needed.
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

    out_dir = Path("./wind10m_speed_images")

    ds = xr.open_dataset(in_path, decode_times=True)

    for vn in (U_NAME, V_NAME):
        if vn not in ds.variables:
            raise KeyError(f"Variable '{vn}' not found. Variables: {list(ds.variables)}")

    lat_name = find_coord_name(ds, ("latitude", "lat"))
    lat_vals = ds[lat_name].values
    lat_ascending = bool(lat_vals[0] < lat_vals[-1])

    u = ds[U_NAME].sel(time=slice(START_TIME, END_TIME))
    v = ds[V_NAME].sel(time=slice(START_TIME, END_TIME))

    times = u["time"].values
    if len(times) == 0:
        raise ValueError("No timesteps found in the requested time range.")

    # Basic alignment check
    if u.sizes.get("time") != v.sizes.get("time"):
        raise ValueError("10u and 10v have different time lengths after slicing.")
    if not np.array_equal(u["time"].values, v["time"].values):
        raise ValueError("10u and 10v time coordinates do not match after slicing.")

    ffmpeg_proc = None
    ffmpeg_stdin = None

    if STREAM_TO_VIDEO:
        # Determine frame size from first slice
        u0 = u.isel(time=0).values
        v0 = v.isel(time=0).values
        if u0.ndim != 2 or v0.ndim != 2:
            raise ValueError(f"Expected 2D (lat,lon) per timestep; got shapes {u0.shape} and {v0.shape}")

        h, w = u0.shape
        h2 = h - (h % 2)
        w2 = w - (w % 2)

        ffmpeg_proc = start_ffmpeg_writer(width=w2, height=h2, fps=FPS, out_path=OUT_VIDEO)
        if ffmpeg_proc.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin.")
        ffmpeg_stdin = ffmpeg_proc.stdin
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, t in enumerate(times):
        uu = u.sel(time=t).values
        vv = v.sel(time=t).values

        speed = np.sqrt(uu.astype(np.float32) ** 2 + vv.astype(np.float32) ** 2)
        rgba = speed_to_rgba(speed, MIN_SPEED, MAX_SPEED)

        # Flip to make north-up if latitude is ascending (-90..90)
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
