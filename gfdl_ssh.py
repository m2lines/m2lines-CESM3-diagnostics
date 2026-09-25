#!/usr/bin/env python
"""Compute mean SSH, SSH variance, and SSH monthly climatology for a GFDL
experiment, matching mom6_tools.surface's SSH.nc schema (mean_ssh,
ssh_variance, ssh_climatology) so Experiment.ssh() reads it back unmodified.

mom6_tools.surface can't be reused unmodified here: its preprocess selects
['oml','mlotst','tos','SSH','SSU','SSV','speed'] from GFDL's regridded
stream in one all-or-nothing call, and GFDL has no SSU/SSV/speed (surface
currents) anywhere -- the same reason gfdl_mld.py exists instead of reusing
mom6_tools.surface for MLD.

mean_ssh/ssh_climatology come from `zos` (ocean_monthly_1x1deg). ssh_variance
comes from `SSH` in the DAILY stream (ocean_daily_1x1deg/ts/daily/5yr/*.SSH.nc
-- a real, separate variable from monthly `zos`, on the identical 1x1deg
grid), following mom6_tools.surface.get_SSH's method: resample daily SSH to
5-day means, then variance of those 5-day means about their own time-mean
(so, like CESM's version, this is total variance including the seasonal
cycle, not a deseasonalized anomaly). One deliberate difference: CESM's
get_SSH only uses the first N//5 *time steps* of its (already avg-window
sliced) daily series, i.e. roughly the first 1/5 of the requested window,
before resampling to 5-day means -- this instead resamples and averages over
the FULL requested window (1988-2012), since nothing here calls for
reproducing that truncation.

Usage:
    python gfdl_ssh.py <CTRL|bANN|uANN|bANN_uANN>
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["CTRL", "bANN", "uANN", "bANN_uANN"])
    args = ap.parse_args()

    case = f"gfdl_{args.experiment}"

    # === mean_ssh / ssh_climatology: monthly zos ===
    src_dir = os.path.join(GFDL_ROOT, args.experiment, "ocean_monthly_1x1deg", "ts", "monthly", "5yr")
    files = sorted(glob.glob(os.path.join(src_dir, "*.zos.nc")))
    print(f"[{case}] {len(files)} monthly file(s) from {src_dir}")

    ds = xr.open_mfdataset(files, chunks={"time": 60})
    ds = ds.rename({"lat": "yh", "lon": "xh"})
    zos = ds["zos"]

    print(f"[{case}] computing monthly climatology over full record...")
    ssh_climatology = zos.groupby("time.month").mean("time").compute()
    ssh_climatology.name = "ssh_climatology"

    zos_sel = zos.sel(time=slice(AVG_START, AVG_END))
    print(f"[{case}] averaging {zos_sel.sizes['time']} months over {AVG_START}..{AVG_END}")

    mean_ssh = zos_sel.mean("time").compute()
    mean_ssh.name = "mean_ssh"

    # === ssh_variance: daily SSH, 5-day-resampled (mom6_tools.surface.get_SSH's method) ===
    daily_dir = os.path.join(GFDL_ROOT, args.experiment, "ocean_daily_1x1deg", "ts", "daily", "5yr")
    daily_files = sorted(glob.glob(os.path.join(daily_dir, "*.SSH.nc")))
    print(f"[{case}] {len(daily_files)} daily file(s) from {daily_dir}")

    ds_daily = xr.open_mfdataset(daily_files, chunks={"time": 300})
    ds_daily = ds_daily.rename({"lat": "yh", "lon": "xh"})
    ssh_daily = ds_daily["SSH"].sel(time=slice(AVG_START, AVG_END))
    print(f"[{case}] resampling {ssh_daily.sizes['time']} days over {AVG_START}..{AVG_END} to 5-day means...")

    ssh_5day = ssh_daily.resample(time="5D").mean("time")
    ssh_bar = ssh_5day.mean("time")
    ssh_variance = ((ssh_5day - ssh_bar) ** 2).mean("time").compute()
    ssh_variance.name = "ssh_variance"

    out_ds = xr.Dataset({
        "mean_ssh": mean_ssh,
        "ssh_variance": ssh_variance,
        "ssh_climatology": ssh_climatology,
    })
    out_ds.attrs.update({
        "description": "SSH mean and climatology (monthly zos), variance "
                       "(5-day-resampled daily SSH, full 1988-2012 window)",
        "start_date": AVG_START,
        "end_date": AVG_END,
        "casename": case,
    })

    os.makedirs("ncfiles", exist_ok=True)
    out_f = f"ncfiles/{case}_SSH.nc"
    out_ds.to_netcdf(out_f)
    print(f"[{case}] wrote {out_f}")


if __name__ == "__main__":
    main()
