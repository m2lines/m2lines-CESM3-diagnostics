#!/usr/bin/env python
"""Interpolate the WOA-2018 obs climatology (prepared on CESM's tx2_3v2 grid)
onto GFDL's own native 1x1 degree grid, so mom6_tools.TS_levels can compare
GFDL model output against it without ever touching/regridding GFDL's model
data itself (see gfdl_shim.py, which keeps GFDL on its own grid).

GFDL has one extra z_l level (6500m) that CESM/WOA don't have -- rather than
truncating GFDL's own data to match, a 35th level is appended here (NaN-filled,
since there's no obs to compare against at that depth anyway), so shapes match
without touching GFDL's files at all.

This is a small, one-time, one-off dataset (a single climatology, 34 depth
levels x 480x540 curvilinear points -> 180x360 regular points -- nowhere near
the multi-decade model output this repo also handles), interpolated with
plain scipy.interpolate.griddata (linear, no conservation), written once to
data/gfdl_obs/ for visual inspection, plus a tiny local intake catalog so
mom6_tools.TS_levels's `-o`/`--obs` argument can select it by name, without
touching the shared reference-datasets.yml catalog owned by another user.

Usage:
    python regrid_obs_to_gfdl_grid.py
"""
import os

import intake
import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO, "data", "gfdl_obs")
OUT_FILE = os.path.join(OUT_DIR, "woa_2018_on_gfdl_grid.nc")
CATALOG_FILE = os.path.join(OUT_DIR, "catalog.yml")
SHARED_CATALOG = "/glade/u/home/gmarques/libs/oce-catalogs/reference-datasets.yml"
OBS_ENTRY = "woa-2018-tx2_3v2-annual-all"

# GFDL's own 1x1 degree grid (matches ocean_monthly_z_1x1deg exactly).
GFDL_LAT = np.arange(-89.5, 90.0, 1.0)
GFDL_LON = np.arange(0.5, 360.0, 1.0)

CATALOG_TEMPLATE = """\
sources:
  woa-2018-gfdl-annual-all:
    description: World Ocean Atlas 2018 all data average, interpolated from CESM's
      tx2_3v2 grid onto GFDL's native 1x1 degree grid (see regrid_obs_to_gfdl_grid.py)
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

    src_lon = (obs.lon.values % 360.0).ravel()
    src_lat = obs.lat.values.ravel()
    points = np.column_stack([src_lon, src_lat])
    tgt_lon2d, tgt_lat2d = np.meshgrid(GFDL_LON, GFDL_LAT)

    print(f"triangulating {len(points)} source points once (reused for every level/variable)...")
    tri = Delaunay(points)

    n_z = obs.sizes["z_l"]
    thetao_out = np.empty((n_z, len(GFDL_LAT), len(GFDL_LON)), dtype="float32")
    so_out = np.empty_like(thetao_out)

    thetao_vals = obs.thetao.values
    so_vals = obs.so.values
    for k in range(n_z):
        print(f"  z_l[{k}]={float(obs.z_l[k]):.1f}m: interpolating thetao/so onto GFDL grid...")
        thetao_out[k] = LinearNDInterpolator(tri, thetao_vals[k].ravel())(tgt_lon2d, tgt_lat2d)
        so_out[k] = LinearNDInterpolator(tri, so_vals[k].ravel())(tgt_lon2d, tgt_lat2d)

    # Pad with GFDL's 35th z_l level (6500m, which CESM/WOA don't have) as NaN,
    # so shapes match GFDL's own data without ever truncating/touching it.
    GFDL_DEEPEST_Z_L = 6500.0
    z_l_padded = np.append(obs.z_l.values, GFDL_DEEPEST_Z_L)
    nan_level = np.full((1, len(GFDL_LAT), len(GFDL_LON)), np.nan, dtype="float32")
    thetao_out = np.concatenate([thetao_out, nan_level], axis=0)
    so_out = np.concatenate([so_out, nan_level], axis=0)

    out_ds = xr.Dataset(
        {
            "thetao": (("z_l", "yh", "xh"), thetao_out, {"units": "degC", "long_name": "Sea Water Potential Temperature"}),
            "so": (("z_l", "yh", "xh"), so_out, {"units": "psu", "long_name": "Sea Water Salinity"}),
        },
        coords={"z_l": z_l_padded, "yh": GFDL_LAT, "xh": GFDL_LON},
        attrs={"description": "WOA-2018 climatology, interpolated from CESM tx2_3v2 grid onto GFDL's native 1x1 degree grid, "
                              f"padded with a NaN {GFDL_DEEPEST_Z_L}m level to match GFDL's own z_l",
               "source": SHARED_CATALOG + "::" + OBS_ENTRY},
    )
    out_ds.to_netcdf(OUT_FILE, encoding={v: {"zlib": True, "complevel": 4} for v in out_ds.data_vars})
    print(f"wrote {OUT_FILE}")

    with open(CATALOG_FILE, "w") as f:
        f.write(CATALOG_TEMPLATE.format(out_file=OUT_FILE))
    print(f"wrote {CATALOG_FILE} (catalog entry: woa-2018-gfdl-annual-all)")


if __name__ == "__main__":
    main()
