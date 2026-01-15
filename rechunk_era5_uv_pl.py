import os
from pathlib import Path
import xarray as xr
from dotenv import load_dotenv
from dask.diagnostics import ProgressBar

DATA_DIR_ENV = "ERA5_CACHE_DIR"


def get_base_dir() -> Path:
    load_dotenv()
    base = os.environ.get(DATA_DIR_ENV)
    if not base:
        raise RuntimeError(f"Env var {DATA_DIR_ENV} is not set.")
    return Path(base)


base_dir = get_base_dir()

in_path  = base_dir / "era5_uv_pl_2020_H1_300hPa_0p25.nc"
out_path = base_dir / "era5_uv_pl_2020_H1_300hPa_0p25_time1.nc"

# open dataset
ds = xr.open_dataset(in_path, engine="netcdf4")

# rechunk: valid_time = 1
ds_time1 = ds.chunk({"valid_time": 1})

# write with progress bar
with ProgressBar():
    ds_time1.to_netcdf(
        out_path,
        engine="netcdf4",
        format="NETCDF4",
    )

ds.close()
