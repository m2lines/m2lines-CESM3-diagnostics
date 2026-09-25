#!/usr/bin/env python
"""Interpolate the de Boyer Montegut (2023) MLD climatology (prepared on
CESM's tx2_3v2 grid, no native lon/lat of its own) onto GFDL's own native
1x1 degree grid, mirroring regrid_obs_to_gfdl_grid.py's approach for WOA.

The catalog entry ('mld-deboyer-2023-tx2_3v2') has no lon/lat coordinates at
all (just bare yh/xh dims) -- its geographic grid is implicitly CESM's own
tx2_3v2 grid, so geolon/geolat are borrowed from a real CESM static file
before interpolating.

Usage:
    python regrid_mld_obs_to_gfdl_grid.py
"""
import os

import intake
import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO, "data", "gfdl_obs")
OUT_FILE = os.path.join(OUT_DIR, "mld_deboyer_2023_on_gfdl_grid.nc")
CATALOG_FILE = os.path.join(OUT_DIR, "catalog.yml")
SHARED_CATALOG = "/glade/u/home/gmarques/libs/oce-catalogs/reference-datasets.yml"
OBS_ENTRY = "mld-deboyer-2023-tx2_3v2"
CESM_STATIC_REF = "/glade/derecho/scratch/pavelp/archive/forced_CTRL/ocn/hist/forced_CTRL.mom6.h.static.nc"

GFDL_LAT = np.arange(-89.5, 90.0, 1.0)
GFDL_LON = np.arange(0.5, 360.0, 1.0)

NEW_CATALOG_ENTRY = """
  mld-deboyer-2023-gfdl-annual-all:
    description: de Boyer Montegut (2023) MLD climatology, interpolated from CESM's
      tx2_3v2 grid onto GFDL's native 1x1 degree grid (see regrid_mld_obs_to_gfdl_grid.py)
    driver: netcdf
    args:
      urlpath: '{out_file}'
      chunks: {{}}
      xarray_kwargs:
        decode_times: False
"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"loading {OBS_ENTRY} from {SHARED_CATALOG}")
    cat = intake.open_catalog(SHARED_CATALOG)
    obs = cat[OBS_ENTRY].to_dask()

    static = xr.open_dataset(CESM_STATIC_REF, decode_times=False)
    src_lon = (static.geolon.values % 360.0).ravel()
    src_lat = static.geolat.values.ravel()
    # This static file leaves geolon/geolat as NaN on masked-land cells;
    # those points carry no usable coordinate and must be dropped before
    # triangulating (Qhull refuses NaN input), along with the matching mld
    # values below.
    valid = np.isfinite(src_lon) & np.isfinite(src_lat)
    points = np.column_stack([src_lon[valid], src_lat[valid]])
    tgt_lon2d, tgt_lat2d = np.meshgrid(GFDL_LON, GFDL_LAT)

    print(f"triangulating {len(points)} source points once (reused for every month)...")
    tri = Delaunay(points)

    n_month = obs.sizes["time"]
    mld_out = np.empty((n_month, len(GFDL_LAT), len(GFDL_LON)), dtype="float32")
    mld_vals = obs.mld.values
    for m in range(n_month):
        print(f"  month {m+1}/{n_month}: interpolating mld onto GFDL grid...")
        mld_out[m] = LinearNDInterpolator(tri, mld_vals[m].ravel()[valid])(tgt_lon2d, tgt_lat2d)

    out_ds = xr.Dataset(
        {"mld": (("time", "yh", "xh"), mld_out, {"units": "m", "long_name": "Mixed Layer Depth"})},
        coords={"time": obs.time.values, "yh": GFDL_LAT, "xh": GFDL_LON},
        attrs={"description": "de Boyer Montegut (2023) MLD climatology, interpolated from CESM "
                              "tx2_3v2 grid onto GFDL's native 1x1 degree grid",
               "source": SHARED_CATALOG + "::" + OBS_ENTRY},
    )
    out_ds.to_netcdf(OUT_FILE, encoding={"mld": {"zlib": True, "complevel": 4}})
    print(f"wrote {OUT_FILE}")

    with open(CATALOG_FILE) as f:
        existing = f.read()
    if "mld-deboyer-2023-gfdl-annual-all" not in existing:
        with open(CATALOG_FILE, "a") as f:
            f.write(NEW_CATALOG_ENTRY.format(out_file=OUT_FILE))
        print(f"appended catalog entry to {CATALOG_FILE}")
    else:
        print(f"catalog entry already present in {CATALOG_FILE}")


if __name__ == "__main__":
    main()
