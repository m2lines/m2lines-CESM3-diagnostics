#!/usr/bin/env python
"""Compute winter/summer mixed-layer-depth climatology for a GFDL experiment,
matching mom6_tools.surface's MLD_winter.nc/MLD_summer.nc naming/variable
convention so Experiment.mld_winter()/mld_summer() read it back unmodified.

mom6_tools.surface (which normally produces these) can't be reused unmodified
here: its preprocess selects ['oml','mlotst','tos','SSH','SSU','SSV','speed']
from GFDL's own regridded stream in one all-or-nothing call, and GFDL has no
equivalent for SSU/SSV/speed (surface currents) anywhere -- so the whole call
fails regardless of renaming MLD_003->mlotst. But MLD_003 itself is a
perfectly real, available equivalent of mlotst (both are the density-
threshold mixed layer depth MOM6 computes online -- CMOR names it mlotst,
GFDL's own diag_table calls it MLD_003), so this computes the winter/summer
climatology directly from it, following mom6_tools.surface.get_MLD's exact
hemisphere-dependent JFM/JAS definition (NH winter = JFM, SH winter = JAS,
and the reverse for summer).

Usage:
    python gfdl_mld.py <CTRL|bANN|uANN|bANN_uANN>
"""
import argparse
import glob
import os

import numpy as np
import xarray as xr

GFDL_ROOT = "/glade/derecho/scratch/pavelp/GFDL"
AVG_START = "1988-01-01"
AVG_END = "2012-01-01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["CTRL", "bANN", "uANN", "bANN_uANN"])
    args = ap.parse_args()

    case = f"gfdl_{args.experiment}"
    src_dir = os.path.join(GFDL_ROOT, args.experiment, "ocean_monthly_1x1deg", "ts", "monthly", "5yr")
    files = sorted(glob.glob(os.path.join(src_dir, "*.MLD_003.nc")))
    print(f"[{case}] {len(files)} file(s) from {src_dir}")

    ds = xr.open_mfdataset(files, chunks={"time": 60})
    ds = ds.rename({"lat": "yh", "lon": "xh"})
    ds_sel = ds.sel(time=slice(AVG_START, AVG_END))
    print(f"[{case}] averaging {ds_sel.sizes['time']} months over {AVG_START}..{AVG_END}")

    mld_clima = ds_sel["MLD_003"].groupby("time.month").mean("time").compute()

    j = int(np.abs(mld_clima.yh.values - 0.0).argmin())  # row nearest the equator
    jfm = mld_clima.isel(month=[0, 1, 2]).mean("month").values
    jas = mld_clima.isel(month=[6, 7, 8]).mean("month").values

    winter = jas.copy()
    winter[j:, :] = jfm[j:, :]     # NH winter = JFM, SH winter = JAS
    summer = jas.copy()
    summer[:j, :] = jfm[:j, :]     # SH summer = JFM, NH summer = JAS

    os.makedirs("ncfiles", exist_ok=True)
    for name, arr in [("winter", winter), ("summer", summer)]:
        da = xr.DataArray(arr, dims=["yh", "xh"], coords={"yh": mld_clima.yh, "xh": mld_clima.xh},
                          name=f"MLD_{name}",
                          attrs={"units": "m", "description": f"{name.capitalize()} MLD (m)",
                                 "long_name": f"{name.capitalize()} mixed layer depth"})
        out_f = f"ncfiles/{case}_MLD_{name}.nc"
        da.to_netcdf(out_f)
        print(f"[{case}] wrote {out_f}")


if __name__ == "__main__":
    main()
