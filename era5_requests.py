# download_era5_uv_levels_2020H1.py
# Submits 4 separate CDS requests in parallel:
#   ERA5 pressure-level u/v wind at 200, 225, 275, 300 hPa
#   Time range: 2020-01-01 00:00 through 2020-06-30 23:00 (all hours)
#   Output: NetCDF files saved into ERA5_CACHE_DIR
#
# Notes:
# - "Parallel" here means we start all downloads concurrently (separate processes).
# - CDS may still queue jobs server-side depending on load / your account limits.

from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import load_dotenv


# ----------------------------
# Constants you edit
# ----------------------------

DATA_DIR_ENV = "ERA5_CACHE_DIR"

YEAR = "2020"
MONTHS = [f"{m:02d}" for m in range(1, 7)]           # Jan..Jun
DAYS = [f"{d:02d}" for d in range(1, 32)]            # 01..31 (CDS handles month lengths)
HOURS = [f"{h:02d}:00" for h in range(24)]           # 00:00..23:00

PRESSURE_LEVELS_HPA = ["200", "225", "275", "300"]   # each as a separate request

# ERA5 pressure-level wind components
VARIABLES = ["u_component_of_wind", "v_component_of_wind"]

DATASET = "reanalysis-era5-pressure-levels"


# ----------------------------
# Helpers
# ----------------------------

def resolve_out_dir() -> Path:
    load_dotenv()
    base = os.environ.get(DATA_DIR_ENV)
    if not base:
        raise EnvironmentError(f"Env var {DATA_DIR_ENV} is not set.")
    out_dir = Path(base)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _download_one_level(level_hpa: str, out_path_str: str) -> str:
    """
    Worker runs in its own process so each request can run concurrently.
    """
    import cdsapi  # import inside worker for clean multiprocessing

    client = cdsapi.Client()

    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [YEAR],
        "month": MONTHS,
        "day": DAYS,
        "time": HOURS,
        "pressure_level": [level_hpa],
        "data_format": "netcdf",
    }

    target = str(out_path_str)
    client.retrieve(DATASET, request, target)
    return target


def main() -> None:
    out_dir = resolve_out_dir()

    jobs = []
    with ProcessPoolExecutor(max_workers=min(4, len(PRESSURE_LEVELS_HPA))) as ex:
        for lvl in PRESSURE_LEVELS_HPA:
            out_name = f"era5_uv_pl_{YEAR}_H1_{lvl}hPa_0p25.nc"
            out_path = out_dir / out_name
            jobs.append(ex.submit(_download_one_level, lvl, str(out_path)))

        for fut in as_completed(jobs):
            try:
                p = fut.result()
                print(f"✅ Finished: {p}")
            except Exception as e:
                print(f"❌ Failed: {e}")


if __name__ == "__main__":
    main()
