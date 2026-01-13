#!/usr/bin/env python3
"""
Read an ERA5-style GRIB file, extract hourly 2m temperature for all of 2020,
colorize each hour (cold->blue, hot->red), write PNG frames to ./images,
then stitch into a 30fps MP4 using ffmpeg.

Assumptions:
- Your GRIB contains 2m temperature (often shortName "2t") on a lat/lon grid.
- Time dimension is hourly and includes (at least) 2020-01-01..2020-12-31 23:00.
- You have: pip install xarray cfgrib numpy pillow python-dotenv
- You have ffmpeg installed and on PATH.
"""

from __future__ import annotations

import os
import sys
import math
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr
from PIL import Image
from dotenv import load_dotenv

# ----------------------------
# Constants you change up top
# ----------------------------
GRIB_FILENAME = "2020_hourly_10mWind_2mTemp_sst.grib"  # file name under DATA_DIR_BASE
ENV_PATH = ".env"                      # will be loaded if present
DATA_DIR_ENVVAR = "ERA5_CACHE_DIR"      # env var name for base path holding the GRIB
OUT_DIR = Path("./images")
VIDEO_PATH = Path("./test_outputs/t2m_2020_30fps.mp4")
FPS = 30

# Temperature-to-color scale (Kelvin). Adjust if you want a different range.
# Roughly: -40C .. +45C
TMIN_K = 233.15
TMAX_K = 318.15

# Output frame sizing:
# - If your grid is huge, writing full-res frames will be heavy.
# - Set MAX_DIM to a smaller number (e.g. 1024) to downscale the longer edge.
MAX_DIM: int | None = None  # None = no resize


def resolve_paths() -> Tuple[Path, Path]:
    load_dotenv(ENV_PATH)

    base = os.environ.get(DATA_DIR_ENVVAR)
    if not base:
        raise RuntimeError(
            f"Missing env var {DATA_DIR_ENVVAR}. Put it in {ENV_PATH} like:\n"
            f'{DATA_DIR_ENVVAR}="/path/to/data"\n'
        )

    grib_path = Path(base) / GRIB_FILENAME
    if not grib_path.exists():
        raise FileNotFoundError(f"GRIB not found: {grib_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return grib_path, OUT_DIR


def temp_to_rgb(temp_k: np.ndarray, tmin_k: float = TMIN_K, tmax_k: float = TMAX_K) -> np.ndarray:
    """
    Map temperature to RGB:
      cold -> blue, hot -> red, mid -> purple-ish
    Returns uint8 RGB image (H, W, 3).

    This is intentionally simple and fast:
      x = clip((T - tmin)/(tmax-tmin), 0..1)
      R = x, G = 0, B = 1-x
    """
    x = (temp_k.astype(np.float32) - float(tmin_k)) / float(tmax_k - tmin_k)
    x = np.clip(x, 0.0, 1.0)

    r = (255.0 * x).astype(np.uint8)
    g = np.zeros_like(r, dtype=np.uint8)
    b = (255.0 * (1.0 - x)).astype(np.uint8)

    return np.dstack([r, g, b])


def maybe_flip_lat(rgb: np.ndarray, lat: np.ndarray | None) -> np.ndarray:
    """
    If latitude is ascending (-90..90), images will appear upside-down vs common map convention.
    Flip vertically to make north up.
    """
    if lat is None:
        return rgb
    if lat.ndim == 1 and lat.size >= 2 and (lat[0] < lat[-1]):
        return np.flipud(rgb)
    return rgb


def maybe_resize(rgb: np.ndarray, max_dim: int | None) -> np.ndarray:
    if max_dim is None:
        return rgb
    h, w = rgb.shape[0], rgb.shape[1]
    if max(h, w) <= max_dim:
        return rgb
    scale = max_dim / float(max(h, w))
    new_w = max(1, int(math.floor(w * scale)))
    new_h = max(1, int(math.floor(h * scale)))
    img = Image.fromarray(rgb, mode="RGB").resize((new_w, new_h), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def open_t2m_dataset(grib_path: Path) -> xr.DataArray:
    """
    Open GRIB and return a DataArray (time, lat, lon) for 2m temperature.
    Uses cfgrib (ecCodes under the hood).
    """
    # cfgrib can sometimes split variables into multiple "datasets".
    # We open the file and try common ways to locate 2m temperature.
    ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": str(grib_path) + ".idx"})

    # Try common variable names for ERA5 GRIB.
    for name in ("t2m", "2t", "t"):
        if name in ds.data_vars:
            da = ds[name]
            break
    else:
        # Fallback: look for a variable with GRIB_shortName == "2t"
        for vname, v in ds.data_vars.items():
            short = v.attrs.get("GRIB_shortName")
            if short == "2t":
                da = v
                break
        else:
            raise KeyError(
                f"Could not find 2m temperature in GRIB. Variables present: {list(ds.data_vars.keys())}"
            )

    # Normalize dimension names
    # cfgrib usually uses "time", "latitude", "longitude"
    # but handle a couple variants.
    lat_name = "latitude" if "latitude" in da.dims else ("lat" if "lat" in da.dims else None)
    lon_name = "longitude" if "longitude" in da.dims else ("lon" if "lon" in da.dims else None)
    time_name = "time" if "time" in da.dims else None

    if time_name is None or lat_name is None or lon_name is None:
        raise ValueError(f"Unexpected dims for t2m: {da.dims}")

    # Ensure time is decoded as datetime64 (cfgrib usually does)
    # Filter to 2020 exactly (safe even if file contains extra).
    da_2020 = da.sel(time=slice("2020-01-01T00:00:00", "2020-12-31T23:00:00"))

    return da_2020


def write_frames(t2m: xr.DataArray, out_dir: Path) -> None:
    lat = None
    if "latitude" in t2m.coords:
        lat = t2m["latitude"].values
    elif "lat" in t2m.coords:
        lat = t2m["lat"].values

    times = t2m["time"].values
    n = times.shape[0]
    if n == 0:
        raise RuntimeError("No timesteps selected for 2020. Check your GRIB contents and time coordinate.")

    print(f"Writing {n} frames to {out_dir.resolve()} ...")

    # Iterate one hour at a time to avoid holding the entire year in memory.
    for i in range(n):
        # Extract one timestep -> (lat, lon) array
        frame = t2m.isel(time=i).values  # numpy array (K)

        rgb = temp_to_rgb(frame, TMIN_K, TMAX_K)
        rgb = maybe_flip_lat(rgb, lat)
        rgb = maybe_resize(rgb, MAX_DIM)

        # Build a stable, ffmpeg-friendly filename
        # Use sequential numbering to guarantee ordering.
        out_path = out_dir / f"frame_{i:05d}.png"

        Image.fromarray(rgb, mode="RGB").save(out_path, format="PNG")

        if (i + 1) % 200 == 0 or (i + 1) == n:
            ts = np.datetime_as_string(times[i], unit="h")
            print(f"  [{i+1:5d}/{n}] wrote {out_path.name} (time={ts})")


def stitch_video_ffmpeg(frames_dir: Path, fps: int, out_video: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg and try again.")

    # Input pattern must match what we wrote above.
    # -framerate controls how quickly frames are read.
    # -r controls output fps.
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(out_video),
    ]

    print("\nRunning ffmpeg:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nWrote video: {out_video.resolve()}")


def main() -> int:
    try:
        grib_path, out_dir = resolve_paths()

        t2m = open_t2m_dataset(grib_path)
        print(f"Opened: {grib_path}")
        print(f"Selected time range: {str(t2m.time.values[0])} .. {str(t2m.time.values[-1])}")
        print(f"Grid shape: {t2m.sizes}")

        write_frames(t2m, out_dir)
        stitch_video_ffmpeg(out_dir, FPS, VIDEO_PATH)

        return 0
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
