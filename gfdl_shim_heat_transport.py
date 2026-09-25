#!/usr/bin/env python
"""Extend the native-grid MOC shim with a 'native'-stream heat-transport shim
for mom6_tools.poleward_heat_transport, which needs T_ady_2d/T_diffy_2d/
T_hbd_diffy_2d from a native-grid stream.

GFDL's ocean_annual_z (see gfdl_shim_moc_z.py) has a real T_ady: same
physical quantity as T_ady_2d ("Advective (by residual mean) Meridional Flux
of Heat", units 'W' -- matches poleward_heat_transport.py's own unit check
exactly, no rho0*Cp conversion needed) but still split across 35 z_l levels
instead of pre-summed. Its own cell_methods attribute ('z_l:sum') confirms
the correct reduction: T_ady_2d = T_ady.sum(dim='z_l'). This is a real
computation on real data, not a fabrication.

T_diffy_2d (diffusive) and T_hbd_diffy_2d (horizontal-boundary-layer
diffusive) are genuinely absent from GFDL's diag_table -- nothing in
ocean_annual_z is a directional (y-component) transport flux for either;
opottempdiff etc. are local tendencies, not fluxes, and reconstructing a flux
from a tendency would mean solving an inverse problem, not something this
does. poleward_heat_transport.py's own preprocess() zero-fills any missing
variable unconditionally (so its "diffusive term not found" warning path
never actually fires -- the zero-filled array satisfies its `in ds.variables`
check) -- this is flagged explicitly wherever the result is presented,
independent of the script's own (silent, in this case) warning mechanism.

Writes into the *same* CASEROOT/CASE as gfdl_shim_moc.py/gfdl_shim_moc_z.py
(cesm_format_moc/ocn/hist/gfdl_<expt>_moc.mom6.h.native.<year>-01.nc),
reusing its already-built shared static file and xmlquery shim -- no new
yaml needed, since yamls/gfdl_<expt>_moc.yaml already declares Fnames.native
= '.mom6.h.native.????-??.nc'.

Usage:
    python gfdl_shim_heat_transport.py <CTRL|bANN|uANN|bANN_uANN> [--limit N]
"""
import argparse
import glob
import os
import time

import xarray as xr

from gfdl_shim_moc import GFDL_ROOT, build_shared_static, build_xmlquery, SHARED_STATIC_FILE

STATIC_SRC_STREAM = "ocean_annual_z"


def log(msg):
    print(msg, flush=True)


def find_chunks(src_dir, limit=None):
    files = sorted(glob.glob(os.path.join(src_dir, "*.T_ady.nc")))
    if limit:
        files = files[:limit]
    return files


def build_experiment_shim(expt, limit=None):
    case = f"gfdl_{expt}_moc"
    src_dir = os.path.join(GFDL_ROOT, expt, STATIC_SRC_STREAM, "ts", "annual", "5yr")
    static_src = os.path.join(GFDL_ROOT, expt, STATIC_SRC_STREAM, f"{STATIC_SRC_STREAM}.static.nc")
    out_root = os.path.join(GFDL_ROOT, expt, "cesm_format_moc")
    out_hist = os.path.join(out_root, "ocn", "hist")
    os.makedirs(out_hist, exist_ok=True)

    # Same native grid as gfdl_shim_moc.py/gfdl_shim_moc_z.py -- reuse the
    # already-built shared static file, nothing new to build.
    build_shared_static(static_src)

    static_link = os.path.join(out_hist, f"{case}.mom6.h.static.nc")
    if not os.path.lexists(static_link):
        os.symlink(SHARED_STATIC_FILE, static_link)
        log(f"[{case}] symlinked static grid -> {SHARED_STATIC_FILE}")

    xmlquery_path = os.path.join(out_root, "xmlquery")
    if not os.path.exists(xmlquery_path):
        build_xmlquery(out_root, case)
        log(f"[{case}] wrote xmlquery shim in {out_root}")

    files = find_chunks(src_dir, limit)
    log(f"[{case}] {len(files)} T_ady chunk(s) to reduce+write from {src_dir}")
    for f in files:
        t0 = time.time()
        chunk = os.path.basename(f).split(".")[1]
        year = chunk[:4]
        merged_out = os.path.join(out_hist, f"{case}.mom6.h.native.{year}-01.nc")
        if os.path.exists(merged_out):
            log(f"  [{chunk}] {os.path.basename(merged_out)} already exists, skipping")
            continue
        tmp_out = merged_out + ".building"
        if os.path.exists(tmp_out):
            os.remove(tmp_out)

        ds = xr.open_dataset(f)
        t_ady_2d = ds["T_ady"].sum(dim="z_l", keep_attrs=True)
        t_ady_2d.name = "T_ady_2d"
        t_ady_2d.attrs["long_name"] = (
            "Advective Meridional Flux of Heat, vertically summed from GFDL's "
            "T_ady (see gfdl_shim_heat_transport.py) -- diffusive and "
            "horizontal-boundary-layer-diffusive components are unavailable "
            "for GFDL and are NOT included in this field."
        )
        t_ady_2d.to_dataset().to_netcdf(tmp_out)
        ds.close()
        os.rename(tmp_out, merged_out)
        log(f"  [{chunk}] -> {os.path.basename(merged_out)} (T_ady_2d, summed over z_l) "
            f"in {time.time()-t0:.1f}s")

    log(f"[{case}] done.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["CTRL", "bANN", "uANN", "bANN_uANN"])
    ap.add_argument("--limit", type=int, default=None, help="only use the first N available chunks (for testing)")
    args = ap.parse_args()
    build_experiment_shim(args.experiment, args.limit)


if __name__ == "__main__":
    main()
