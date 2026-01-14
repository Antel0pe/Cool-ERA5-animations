# lsm_to_pngs.py
# Reads an ERA5 land-sea mask NetCDF (static field), writes:
#   (1) landsea_mask.png    : 1 if land, 0 if ocean  (matches dataset semantics)
#   (2) coastlines.png      : 1 on coastline pixels, 0 elsewhere
#
# Coastline definition here (4-neighbor / Von Neumann):
#   A land pixel is coastline if ANY of its up/down/left/right neighbors is sea.
#
# NOTE ON DIAGONALS (important):
# - Your 4-neighbor rule is totally reasonable and common.
# - If you care about “catch every thin diagonal shoreline,” then you SHOULD include diagonals too
#   (8-neighbor / Moore), because some coast segments can touch sea only diagonally at this resolution.
# - This script exposes a toggle USE_DIAGONALS so you can choose.

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image
from dotenv import load_dotenv  # optional


# ----------------------------
# Constants you edit
# ----------------------------

DATA_DIR_ENV = "ERA5_CACHE_DIR"

# The NetCDF you downloaded (inside ERA5_CACHE_DIR)
NC_FILENAME = "era5_land_sea_mask_0p25.nc"

# Variable name inside the file (ERA5 is usually "lsm")
VAR_NAME = "lsm"

# Output dir (created if needed)
OUT_DIR = Path("./mask_outputs")

# Output filenames
OUT_LANDSEA = OUT_DIR / "landsea_mask.png"
OUT_COAST   = OUT_DIR / "coastlines.png"

# Coastline neighborhood rule:
# False -> 4-neighbor (up/down/left/right)
# True  -> 8-neighbor (also diagonals)  [more “complete” coast detection]
USE_DIAGONALS = False


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
    raise KeyError(
        f"Could not find any of these coordinates: {candidates}. "
        f"Available coords: {list(ds.coords)}"
    )


def to_uint8_png(mask01: np.ndarray) -> Image.Image:
    """
    Convert a 0/1 mask to an 8-bit grayscale PNG where:
      0 -> black
      1 -> white
    """
    if mask01.dtype != np.uint8:
        mask01 = mask01.astype(np.uint8)
    img = (mask01 * 255).astype(np.uint8)
    return Image.fromarray(img, mode="L")


def compute_coastline(land01: np.ndarray, use_diagonals: bool) -> np.ndarray:
    """
    land01: uint8 array (H,W) with 1=land, 0=sea
    Returns coastline01: uint8 array (H,W) with 1=coastline pixels, 0 otherwise

    Coastline pixel definition:
      land pixel AND has at least one neighboring sea pixel (per chosen neighborhood).
    """
    land = land01.astype(bool)

    # We treat out-of-bounds as sea by padding with sea (False).
    pad = 1
    p = np.pad(land, pad_width=pad, mode="constant", constant_values=False)

    # 4-neighbors
    up    = p[0:-2, 1:-1]
    down  = p[2:  , 1:-1]
    left  = p[1:-1, 0:-2]
    right = p[1:-1, 2:  ]

    # A neighbor is sea if it's False
    sea_adjacent = (~up) | (~down) | (~left) | (~right)

    if use_diagonals:
        ul = p[0:-2, 0:-2]
        ur = p[0:-2, 2:  ]
        dl = p[2:  , 0:-2]
        dr = p[2:  , 2:  ]
        sea_adjacent |= (~ul) | (~ur) | (~dl) | (~dr)

    coast = land & sea_adjacent
    return coast.astype(np.uint8)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    in_path = resolve_input_path()
    if not in_path.exists():
        raise FileNotFoundError(f"Input NetCDF not found: {in_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(in_path, decode_times=True)

    if VAR_NAME not in ds.variables:
        raise KeyError(f"Variable '{VAR_NAME}' not found. Variables: {list(ds.variables)}")

    # Pull the mask array. ERA5 lsm is typically (lat, lon) with float values in [0,1].
    da = ds[VAR_NAME]

    # If it has extra singleton dims, squeeze them out.
    da2 = da.squeeze(drop=True)

    arr = da2.values
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask (lat,lon); got shape {arr.shape}")

    # Figure out if we should flip north-up.
    lat_name = find_coord_name(ds, ("latitude", "lat"))
    lat_vals = ds[lat_name].values
    lat_ascending = bool(lat_vals[0] < lat_vals[-1])

    # Convert to 0/1 like the dataset semantics (1 land, 0 sea).
    # Some files store exact 0/1; others store fractions near coasts.
    # We threshold at 0.5 to force a crisp binary land/sea mask.
    land01 = (arr >= 0.5).astype(np.uint8)

    # Flip to north-up if needed
    if lat_ascending:
        land01 = np.flipud(land01)

    coast01 = compute_coastline(land01, use_diagonals=USE_DIAGONALS)

    # Write PNGs
    to_uint8_png(land01).save(OUT_LANDSEA)
    to_uint8_png(coast01).save(OUT_COAST)

    print(f"Wrote: {OUT_LANDSEA.resolve()}")
    print(f"Wrote: {OUT_COAST.resolve()}")
    print(f"Coastline neighborhood: {'8-neighbor (with diagonals)' if USE_DIAGONALS else '4-neighbor (no diagonals)'}")


if __name__ == "__main__":
    main()
