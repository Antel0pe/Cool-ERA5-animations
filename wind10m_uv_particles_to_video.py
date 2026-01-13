# wind10m_uv_particles_to_video.py
# Reads wind10mUV_2020.nc, extracts 10u/10v, computes wind speed background,
# and overlays advected wind particles (one dot per particle) before encoding video.

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

DATA_DIR_ENV = "ERA5_CACHE_DIR"
NC_FILENAME = "wind10mUV_2020.nc"

U_NAME = "10u"
V_NAME = "10v"

START_TIME = "2020-01-01T00:00:00"
END_TIME   = "2020-01-07T23:00:00"

STREAM_TO_VIDEO = True

FPS = 30
OUT_VIDEO = Path(f"./test_outputs/wind10m_particles_{START_TIME[:10]}_{END_TIME[:10]}_{FPS}fps.mp4")

IMG_EXT = "png"

MIN_SPEED = 0.0
MAX_SPEED = 25.0

LOG_EVERY_N = 168  # ~weekly at hourly cadence

# ---- particle integration ----
R_EARTH_M = 6371000.0
DT_SECONDS = 3600.0

# We'll remove one latitude row to go 721 -> 720 (you said you do this manually later).
# This chooses the LAST row (common choice). If you want to drop the FIRST row instead,
# set DROP_LAT_ROW = 0.
DROP_LAT_ROW = -1

# pole handling
POLE_EPS = 1e-6  # avoid exactly +/-90

PARTICLE_STRIDE = 16

# reseed controls (based on per-step displacement)
RESEED_DIST_M = 250_000.0
RESEED_P_MAX  = 0.25
RESEED_RAMP_M = 250_000.0

STILL_M_PER_STEP = 500.0     # below this is "barely moving" (meters per hour-step)
STILL_P_MAX = 0.8            # max chance to die per frame when totally still
STILL_RAMP_M = 1500.0        # ramp from STILL_P_MAX at 0 up to ~0 near STILL_M_PER_STEP

DENSITY_TILE = 8            # pixels per tile (try 4, 6, 8)
MAX_PARTICLES_PER_TILE = 2    # keep 1–3

TRAIL_DECAY = 0.96        # closer to 1 = longer trails (try 0.94..0.985)
TRAIL_STRENGTH = 0.25     # how much a particle “writes” per frame (0..1)
TRAIL_GAMMA = 1.0         # >1 makes faint trails fainter; <1 makes them pop
TRAIL_ALPHA_MAX = 0.85    # cap opacity of trails when compositing
TRAIL_COLOR = np.array([0, 0, 0], dtype=np.float32)  # black trails
HEAD_ALPHA = 0.35 

# dot drawing
DOT_RGBA = np.array([0, 0, 0, 255], dtype=np.uint8)  # black


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
    s = speed.astype(np.float32)
    t = (s - float(min_speed)) / (float(max_speed) - float(min_speed))
    t = np.clip(t, 0.0, 1.0)

    rg = ((1.0 - t) * 255.0).astype(np.uint8)
    b = np.full_like(rg, 255, dtype=np.uint8)
    a = np.full_like(rg, 255, dtype=np.uint8)

    nan_mask = ~np.isfinite(s)
    if np.any(nan_mask):
        rg[nan_mask] = 0
        b[nan_mask] = 0
        a[nan_mask] = 0

    return np.stack([rg, rg, b, a], axis=-1)


def even_crop_rgba(rgba: np.ndarray) -> np.ndarray:
    h, w, _ = rgba.shape
    if (h % 2) == 1:
        rgba = rgba[:-1, :, :]
    if (w % 2) == 1:
        rgba = rgba[:, :-1, :]
    return rgba


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


# ---- particle math ----

def wrap_lon(lon_deg: np.ndarray) -> np.ndarray:
    # [-180, 180)
    return ((lon_deg + 180.0) % 360.0) - 180.0


def reflect_poles(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat = lat_deg.copy()
    lon = lon_deg.copy()

    over = lat > (90.0 - POLE_EPS)
    if np.any(over):
        lat[over] = (180.0 - lat[over])
        lon[over] = lon[over] + 180.0

    under = lat < (-90.0 + POLE_EPS)
    if np.any(under):
        lat[under] = (-180.0 - lat[under])
        lon[under] = lon[under] + 180.0

    lon = wrap_lon(lon)
    lat = np.clip(lat, -90.0 + POLE_EPS, 90.0 - POLE_EPS)
    return lat, lon


def uv_to_dlat_dlon_deg_per_s(uu: np.ndarray, vv: np.ndarray, lat_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.deg2rad(lat_deg)
    coslat = np.cos(lat_rad)
    coslat = np.clip(coslat, 1e-3, None)

    dlat = (vv / R_EARTH_M) * (180.0 / np.pi)
    dlon = (uu / (R_EARTH_M * coslat)) * (180.0 / np.pi)
    return dlat, dlon


def nearest_uv(
    u2d: np.ndarray,
    v2d: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    plat: np.ndarray,
    plon: np.ndarray,
    lon_is_0_360: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized nearest-neighbor sampling on a regular lat/lon grid.
    plat/plon same shape as output.
    """
    H = lat_vals.shape[0]
    W = lon_vals.shape[0]

    lat0 = float(lat_vals[0])
    lat1 = float(lat_vals[-1])
    lon0 = float(lon_vals[0])
    lon1 = float(lon_vals[-1])

    if lon_is_0_360:
        plon2 = plon % 360.0
    else:
        plon2 = wrap_lon(plon)

    # lat can be ascending or descending
    if lat0 > lat1:  # descending
        fi = (lat0 - plat) * (H - 1) / (lat0 - lat1)
    else:
        fi = (plat - lat0) * (H - 1) / (lat1 - lat0)

    # lon assumed ascending
    fj = (plon2 - lon0) * (W - 1) / (lon1 - lon0)

    ii = np.clip(np.rint(fi).astype(np.int32), 0, H - 1)
    jj = np.clip(np.rint(fj).astype(np.int32), 0, W - 1)

    return u2d[ii, jj], v2d[ii, jj]


def rk2_step_midpoint(
    plat: np.ndarray,
    plon: np.ndarray,
    dt: float,
    u2d: np.ndarray,
    v2d: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    lon_is_0_360: bool,
) -> tuple[np.ndarray, np.ndarray]:
    # k1
    u1, v1 = nearest_uv(u2d, v2d, lat_vals, lon_vals, plat, plon, lon_is_0_360)
    dlat1, dlon1 = uv_to_dlat_dlon_deg_per_s(u1, v1, plat)

    # midpoint
    lat_mid = plat + 0.5 * dt * dlat1
    lon_mid = plon + 0.5 * dt * dlon1
    lon_mid = (lon_mid % 360.0) if lon_is_0_360 else wrap_lon(lon_mid)
    lat_mid, lon_mid = reflect_poles(lat_mid, lon_mid)

    # k2 at midpoint
    u2, v2 = nearest_uv(u2d, v2d, lat_vals, lon_vals, lat_mid, lon_mid, lon_is_0_360)
    dlat2, dlon2 = uv_to_dlat_dlon_deg_per_s(u2, v2, lat_mid)

    # full step
    lat_new = plat + dt * dlat2
    lon_new = plon + dt * dlon2
    lon_new = (lon_new % 360.0) if lon_is_0_360 else wrap_lon(lon_new)
    lat_new, lon_new = reflect_poles(lat_new, lon_new)

    return lat_new, lon_new


def step_distance_m(lat0: np.ndarray, lon0: np.ndarray, lat1: np.ndarray, lon1: np.ndarray) -> np.ndarray:
    lat0r = np.deg2rad(lat0.astype(np.float64))
    lat1r = np.deg2rad(lat1.astype(np.float64))
    lon0r = np.deg2rad(lon0.astype(np.float64))
    lon1r = np.deg2rad(lon1.astype(np.float64))

    dlat = lat1r - lat0r
    dlon = lon1r - lon0r
    dlon = (dlon + np.pi) % (2*np.pi) - np.pi  # wrap shortest way

    latm = 0.5 * (lat0r + lat1r)
    x = dlon * np.cos(latm)
    y = dlat
    return (R_EARTH_M * np.sqrt(x*x + y*y)).astype(np.float32)


def rand01_from_id_and_frame(pid: np.ndarray, frame_i: int) -> np.ndarray:
    x = pid.astype(np.uint32)

    # force uint32 arithmetic BEFORE multiply
    frame = np.uint32(frame_i)
    x ^= frame * np.uint32(747796405)

    x ^= (x >> 16)
    x *= np.uint32(2246822519)
    x ^= (x >> 13)
    x *= np.uint32(3266489917)
    x ^= (x >> 16)

    return (x >> 8).astype(np.float32) / np.float32(1 << 24)

def particles_to_pixel_indices(
    plat: np.ndarray,
    plon: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    H: int,
    W: int,
    lat_ascending: bool,
    lon_is_0_360: bool,
) -> tuple[np.ndarray, np.ndarray]:
    lat0 = float(lat_vals[0])
    lat1 = float(lat_vals[-1])
    lon0 = float(lon_vals[0])
    lon1 = float(lon_vals[-1])

    if lon_is_0_360:
        plon2 = plon % 360.0
    else:
        plon2 = wrap_lon(plon)

    if lat0 > lat1:  # descending
        fi = (lat0 - plat) * (H - 1) / (lat0 - lat1)
    else:
        fi = (plat - lat0) * (H - 1) / (lat1 - lat0)

    fj = (plon2 - lon0) * (W - 1) / (lon1 - lon0)

    ii = np.clip(np.rint(fi).astype(np.int32), 0, H - 1)
    jj = np.clip(np.rint(fj).astype(np.int32), 0, W - 1)

    # If the rendered image was flipud'd, flip the row indices to match
    if lat_ascending:
        ii = (H - 1) - ii

    return ii, jj


# ----------------------------
# Main
# ----------------------------

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
    lon_name = find_coord_name(ds, ("longitude", "lon"))

    lat_vals_full = ds[lat_name].values
    lon_vals = ds[lon_name].values

    lat_ascending = bool(lat_vals_full[0] < lat_vals_full[-1])

    # Decide lon convention in-file
    lon_is_0_360 = bool(float(lon_vals.min()) >= 0.0 and float(lon_vals.max()) > 180.0)

    # Slice time
    u = ds[U_NAME].sel(time=slice(START_TIME, END_TIME))
    v = ds[V_NAME].sel(time=slice(START_TIME, END_TIME))

    times = u["time"].values
    if len(times) == 0:
        raise ValueError("No timesteps found in the requested time range.")

    if u.sizes.get("time") != v.sizes.get("time"):
        raise ValueError("10u and 10v have different time lengths after slicing.")
    if not np.array_equal(u["time"].values, v["time"].values):
        raise ValueError("10u and 10v time coordinates do not match after slicing.")

    # Pull first slice for dimensions
    u0 = u.isel(time=0).values
    v0 = v.isel(time=0).values
    if u0.ndim != 2 or v0.ndim != 2:
        raise ValueError(f"Expected 2D (lat,lon) per timestep; got shapes {u0.shape} and {v0.shape}")

    # You said lat is 721 but you drop one row later to make 720.
    # We'll do it explicitly and consistently for EVERYTHING: u, v, lat.
    if u0.shape[0] == 721:
        lat_vals = np.delete(lat_vals_full, DROP_LAT_ROW, axis=0)
    else:
        lat_vals = lat_vals_full

    # initialize particles on a 2D grid matching final shape (H,W)
    H_full, W_full = u0.shape
    if H_full != lat_vals_full.shape[0]:
        # This can happen if dataset uses different dim name ordering; but generally should match.
        pass

    # Build particle grid from coords, then drop same latitude row if needed
    P_LAT, P_LON = np.meshgrid(lat_vals_full, lon_vals, indexing="ij")  # (H_full,W_full) if H_full=721,W=1440
    if P_LAT.shape[0] == 721 and u0.shape[0] == 721:
        P_LAT = np.delete(P_LAT, DROP_LAT_ROW, axis=0)
        P_LON = np.delete(P_LON, DROP_LAT_ROW, axis=0)
        
    P_LAT = P_LAT[::PARTICLE_STRIDE, ::PARTICLE_STRIDE]
    P_LON = P_LON[::PARTICLE_STRIDE, ::PARTICLE_STRIDE]

    # initial positions for reseed
    P0_LAT = P_LAT.copy()
    P0_LON = P_LON.copy()

    # per-step movement
    P_MOVE_M = np.zeros_like(P_LAT, dtype=np.float32)

    # stable particle ids
    P_ID = np.arange(P_LAT.size, dtype=np.uint32).reshape(P_LAT.shape)

    # ffmpeg init (after we know final rendered size)
    ffmpeg_proc = None
    ffmpeg_stdin = None

    # Determine output frame size after possible crops
    # We'll compute one rgba to get exact dims after dropping lat row and any flip.
    # (But avoid doing extra work: just use u0/v0 after matching row removal.)
    uu0 = u0
    vv0 = v0
    if uu0.shape[0] == 721:
        uu0 = np.delete(uu0, DROP_LAT_ROW, axis=0)
        vv0 = np.delete(vv0, DROP_LAT_ROW, axis=0)

    speed0 = np.sqrt(uu0.astype(np.float32) ** 2 + vv0.astype(np.float32) ** 2)
    rgba0 = speed_to_rgba(speed0, MIN_SPEED, MAX_SPEED)
    if lat_ascending:
        rgba0 = np.flipud(rgba0)
    rgba0 = even_crop_rgba(rgba0)
    H_img, W_img, _ = rgba0.shape
    trail = np.zeros((H_img, W_img), dtype=np.float32)  # 0..1 “ink”

    if STREAM_TO_VIDEO:
        ffmpeg_proc = start_ffmpeg_writer(width=W_img, height=H_img, fps=FPS, out_path=OUT_VIDEO)
        if ffmpeg_proc.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin.")
        ffmpeg_stdin = ffmpeg_proc.stdin
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Main loop
    for i, t in enumerate(times):
        uu = u.sel(time=t).values
        vv = v.sel(time=t).values

        # Drop the same latitude row (721->720) for data to match particle grid
        if uu.shape[0] == 721:
            uu = np.delete(uu, DROP_LAT_ROW, axis=0)
            vv = np.delete(vv, DROP_LAT_ROW, axis=0)

        speed = np.sqrt(uu.astype(np.float32) ** 2 + vv.astype(np.float32) ** 2)
        rgba = speed_to_rgba(speed, MIN_SPEED, MAX_SPEED)

        # ---- ADVECT PARTICLES (RK2) ----
        lat_prev = P_LAT
        lon_prev = P_LON

        lat_new, lon_new = rk2_step_midpoint(
            P_LAT, P_LON, DT_SECONDS,
            uu, vv,
            lat_vals, lon_vals,
            lon_is_0_360=lon_is_0_360
        )

        P_MOVE_M[...] = step_distance_m(lat_prev, lon_prev, lat_new, lon_new)
        
        # --- kill still particles quickly ---
        # p_still ~ STILL_P_MAX when move=0, goes to 0 by move>=STILL_M_PER_STEP
        stillness = np.maximum(STILL_M_PER_STEP - P_MOVE_M, 0.0)
        t = np.clip(stillness / STILL_RAMP_M, 0.0, 1.0)
        p_still = t * STILL_P_MAX

        r_still = rand01_from_id_and_frame(P_ID ^ np.uint32(0xA5A5A5A5), i)  # decorrelate stream
        still_kill = r_still < p_still

        excess = np.maximum(P_MOVE_M - RESEED_DIST_M, 0.0)
        p = np.clip(excess / RESEED_RAMP_M, 0.0, 1.0) * RESEED_P_MAX
        r = rand01_from_id_and_frame(P_ID, i)
        reseed_mask = r < p
        reseed_mask = reseed_mask | still_kill

        if np.any(reseed_mask):
            lat_new = lat_new.copy()
            lon_new = lon_new.copy()
            lat_new[reseed_mask] = P0_LAT[reseed_mask]
            lon_new[reseed_mask] = P0_LON[reseed_mask]

        P_LAT = lat_new
        P_LON = lon_new

        # ---- Render / flip / crop ----
        if lat_ascending:
            rgba = np.flipud(rgba)

        rgba = even_crop_rgba(rgba)

        # ---- Draw dots (vectorized) ----
        ii, jj = particles_to_pixel_indices(
            P_LAT, P_LON,
            lat_vals, lon_vals,
            H=rgba.shape[0], W=rgba.shape[1],
            lat_ascending=lat_ascending,
            lon_is_0_360=lon_is_0_360,
        )
        
        ii1 = np.asarray(ii).reshape(-1)
        jj1 = np.asarray(jj).reshape(-1)


        Himg, Wimg = rgba.shape[0], rgba.shape[1]

        # tile coordinates
        ti = (ii1 // DENSITY_TILE).astype(np.int64)
        tj = (jj1 // DENSITY_TILE).astype(np.int64)
        tiles_w = (Wimg + DENSITY_TILE - 1) // DENSITY_TILE

        tile_idx = ti * tiles_w + tj

        order = np.argsort(tile_idx, kind="mergesort")
        tile_sorted = tile_idx[order]

        is_new = np.empty_like(tile_sorted, dtype=bool)
        is_new[0] = True
        is_new[1:] = tile_sorted[1:] != tile_sorted[:-1]

        group_start = np.maximum.accumulate(np.where(is_new, np.arange(tile_sorted.size), 0))
        pos = np.arange(tile_sorted.size) - group_start

        keep_sorted = pos < MAX_PARTICLES_PER_TILE

        keep1 = np.zeros_like(keep_sorted, dtype=bool)
        keep1[order] = keep_sorted
        
        # decay (multiplicative fade looks natural)
        trail *= TRAIL_DECAY
        # splat particles (only those you keep after density cap)
        trail[ii1[keep1], jj1[keep1]] = np.minimum(
            1.0,
            trail[ii1[keep1], jj1[keep1]] + TRAIL_STRENGTH
        )
        
        # --- composite trails over background ---
        alpha = np.clip(trail ** TRAIL_GAMMA, 0.0, 1.0) * TRAIL_ALPHA_MAX  # (H,W)

        bg = rgba[..., :3].astype(np.float32)
        a3 = alpha[..., None].astype(np.float32)

        # 1) apply trails to whole image
        out_rgb = bg * (1.0 - a3)
        rgba[..., :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)

        # 2) optional heads on top (after trails)
        rgba[ii1[keep1], jj1[keep1], :3] = (
            rgba[ii1[keep1], jj1[keep1], :3].astype(np.float32) * (1.0 - HEAD_ALPHA)
        ).astype(np.uint8)


        # rgba[..., :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        # rgba[ii1[keep1], jj1[keep1]] = DOT_RGBA
        # rgba[ii, jj] = DOT_RGBA


        # ---- Output ----
        if STREAM_TO_VIDEO:
            assert ffmpeg_stdin is not None
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
