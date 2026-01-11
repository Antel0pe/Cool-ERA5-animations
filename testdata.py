# makePngs.py
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr
from dotenv import load_dotenv
from dask.diagnostics import ProgressBar
from PIL import Image
import dask


# ---------------------------
# Knobs you change while iterating
# ---------------------------
ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

VAR = "u_component_of_wind"
LEVEL_HPA = 850
START = "2010-01-01T00:00:00"
END   = "2010-01-15T00:00:00"   # end exclusive

OUT_DIR_NAME = "images"

# Color scale (m/s)
VMIN = -30.0
VMAX =  30.0

# Batch size in hours (this is the speed knob)
BATCH_HOURS = 168
# ---------------------------


def get_out_dir() -> Path:
    load_dotenv()
    base = os.getenv("OUT_DIR", ".")
    out_dir = Path(base).expanduser().resolve() / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def open_arco_zarr(zarr_url: str) -> xr.Dataset:
    return xr.open_zarr(
        zarr_url,
        consolidated=True,
        chunks=None,  # fast open
        storage_options={"token": "anon"},
    )


def infer_coord_names(ds: xr.Dataset) -> tuple[str, str]:
    time_name = "time" if "time" in ds.coords else ("valid_time" if "valid_time" in ds.coords else None)
    level_name = "level" if "level" in ds.coords else ("pressure_level" if "pressure_level" in ds.coords else None)
    if time_name is None:
        raise KeyError(f"Couldn't find time coord. Coords: {list(ds.coords)}")
    if level_name is None:
        raise KeyError(f"Couldn't find level coord. Coords: {list(ds.coords)}")
    return time_name, level_name


def simple_blue_red_rgba(field: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    f = field.astype(np.float32, copy=False)

    m = max(abs(vmin), abs(vmax))
    if m == 0:
        m = 1.0
    a = np.clip(f / m, -1.0, 1.0)  # [-1, 1]

    r = np.where(a > 0, a, 0.0)
    b = np.where(a < 0, -a, 0.0)
    g = 1.0 - (np.abs(a) * 0.75)

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)

    rgba = (rgb * 255.0 + 0.5).astype(np.uint8)
    alpha = np.full((*rgba.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate([rgba, alpha], axis=-1)


def time_to_fname(t: np.datetime64) -> str:
    s = np.datetime_as_string(t, unit="m")  # "YYYY-MM-DDTHH:MM"
    date, hm = s.split("T")
    ymd = date.replace("-", "")
    hhmm = hm.replace(":", "")
    return f"{ymd}_{hhmm}"


def main() -> None:
    # Good default for IO-bound chunk fetching
    dask.config.set(scheduler="threads")

    out_dir = get_out_dir()
    print("Output dir:", out_dir)

    ds = open_arco_zarr(ZARR_URL)
    time_name, level_name = infer_coord_names(ds)

    if VAR not in ds.data_vars:
        raise KeyError(f"Variable '{VAR}' not found. Available vars include: {list(ds.data_vars)[:20]} ...")

    da = (
        ds[VAR]
        .sel({level_name: LEVEL_HPA})
        .sel({time_name: slice(START, END)})
        .chunk({time_name: BATCH_HOURS})  # <-- batching happens here
    )

    times = da[time_name].values
    n = len(times)
    print(f"Frames to write: {n}")
    print(f"Batch hours: {BATCH_HOURS}  (≈ {int(np.ceil(n / BATCH_HOURS))} remote reads)")

    # Process batches
    with ProgressBar():
        for i0 in range(0, n, BATCH_HOURS):
            i1 = min(i0 + BATCH_HOURS, n)

            # One remote compute per batch
            block = da.isel({time_name: slice(i0, i1)}).compute().values  # (tb, lat, lon)

            # Write PNGs for each hour in the batch
            for j in range(i1 - i0):
                t = times[i0 + j]
                frame = block[j]  # (lat, lon)
                rgba = simple_blue_red_rgba(frame, VMIN, VMAX)

                fname = f"{VAR}_hPa{LEVEL_HPA}_{time_to_fname(t)}.png"
                Image.fromarray(rgba, mode="RGBA").save(out_dir / fname)

            print(f"Wrote {i1}/{n}", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
