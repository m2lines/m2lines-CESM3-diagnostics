#!/usr/bin/env python
"""Compute a volume-weighted global-mean thetao/so time series for a GFDL
experiment, matching the thetaoga/soga naming and (time, scalar_axis) shape
of mom6_tools.stats's mon_ave_global_means.nc -- so Experiment.global_means()
reads it back completely unmodified.

GFDL has no equivalent to this: mom6_tools.stats's `-time_series` reads
thetaoga/soga as already-computed online scalar diagnostics from CESM's MOM6
native-stream history files, which GFDL's postprocessing never wrote. This
computes the actual volume-weighted mean directly from the shim's thetao/so
+ volcello (cell volume, already present in GFDL's own z-stream files,
carried through untouched by gfdl_shim.py) -- new computation, not something
metadata/renaming alone could produce.

Usage:
    python gfdl_global_means.py <CTRL|bANN|uANN|bANN_uANN>
"""
import argparse
import glob
import os

import xarray as xr

GFDL_ROOT = "/glade/derecho/scratch/pavelp/GFDL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["CTRL", "bANN", "uANN", "bANN_uANN"])
    args = ap.parse_args()

    case = f"gfdl_{args.experiment}"
    hist = os.path.join(GFDL_ROOT, args.experiment, "cesm_format", "ocn", "hist")
    files = sorted(glob.glob(os.path.join(hist, f"{case}.mom6.h.z.????-??.nc")))
    print(f"[{case}] {len(files)} file(s) from {hist}")

    ds = xr.open_mfdataset(files, chunks={"time": 60})
    weights = ds["volcello"]

    thetaoga = (ds["thetao"] * weights).sum(dim=("z_l", "yh", "xh")) / weights.sum(dim=("z_l", "yh", "xh"))
    soga = (ds["so"] * weights).sum(dim=("z_l", "yh", "xh")) / weights.sum(dim=("z_l", "yh", "xh"))

    print(f"[{case}] computing over {ds.sizes['time']} months "
          f"({ds.time.values[0]} .. {ds.time.values[-1]})...")
    thetaoga = thetaoga.compute()
    soga = soga.compute()

    out_ds = xr.Dataset(
        {
            "thetaoga": (("time", "scalar_axis"), thetaoga.values[:, None].astype("float32"),
                         {"units": "degC", "long_name": "Global Mean Ocean Potential Temperature"}),
            "soga": (("time", "scalar_axis"), soga.values[:, None].astype("float32"),
                     {"units": "psu", "long_name": "Global Mean Ocean Salinity"}),
        },
        coords={"time": ds["time"], "scalar_axis": [0]},
        attrs={"description": "Volume-weighted global-mean ocean T/S, computed directly from GFDL's "
                              "thetao/so/volcello (see gfdl_global_means.py) -- GFDL has no online-computed "
                              "thetaoga/soga the way mom6_tools.stats expects from CESM's native stream."},
    )
    out_f = f"ncfiles/{case}_mon_ave_global_means.nc"
    os.makedirs("ncfiles", exist_ok=True)
    out_ds.to_netcdf(out_f, encoding={v: {"zlib": True, "complevel": 4} for v in out_ds.data_vars})
    print(f"[{case}] wrote {out_f}")


if __name__ == "__main__":
    main()
