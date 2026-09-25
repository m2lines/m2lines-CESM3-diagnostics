#!/usr/bin/env python
"""Extend the native-grid MOC shim (built by gfdl_shim_moc.py) with
z-coordinate transport chunks, enabling mom6_tools.moc (which needs its
vmo/vhml/vhGM from the *'z' Fnames stream*) in addition to moc_sigma2 (which
already runs off the 'rho2' stream this shim also builds).

This became possible once umo/vmo/uhml/vhml/uo/vo were copied into a new
native-grid, z-coordinate stream (ocean_annual_z) -- unlike
ocean_monthly_z_1x1deg (regridded, no transport variables at all), this one
carries the real thing on GFDL's own tripolar grid, 35 z-levels.

Writes into the *same* CASEROOT/CASE as gfdl_shim_moc.py
(cesm_format_moc/ocn/hist/gfdl_<expt>_moc.mom6.h.z.<year>-01.nc), reusing its
already-built shared static file and xmlquery shim -- no new yaml needed,
since yamls/gfdl_<expt>_moc.yaml already declares Fnames.z =
'.mom6.h.z.????-??.nc'.

Only vmo/vhml/vo are copied (mom6_tools.moc's own preprocess only ever
selects vmo/vhml/vhGM -- see its module docstring for the exact code -- plus
'vo' purely as a shape/dtype template for its built-in zero-fill fallback
when vhGM, genuinely absent from GFDL's diag_table, is missing). Unlike the
sigma2 shim, this vo is GFDL's own **real** surface/interior meridional
velocity, not a fabricated placeholder -- no netCDF4 zero-fill step needed at
all, since the fallback only needs *a* variable of the right shape/dtype to
template from, and this one happens to already be genuinely present.

Usage:
    python gfdl_shim_moc_z.py <CTRL|bANN|uANN|bANN_uANN> [--limit N]
"""
import argparse
import glob
import os
import shutil
import subprocess
import time

from gfdl_shim_moc import GFDL_ROOT, build_shared_static, build_xmlquery, SHARED_STATIC_FILE

STATIC_SRC_STREAM = "ocean_annual_z"


def log(msg):
    print(msg, flush=True)


def run(cmd):
    subprocess.run(cmd, check=True)


def find_chunks(src_dir, limit=None):
    vmo_files = sorted(glob.glob(os.path.join(src_dir, "*.vmo.nc")))
    chunks = []
    for vf in vmo_files:
        chunk = os.path.basename(vf).split(".")[1]
        others = {
            "vhml": vf.replace(".vmo.nc", ".vhml.nc"),
            "vo": vf.replace(".vmo.nc", ".vo.nc"),
        }
        if not all(os.path.exists(p) for p in others.values()):
            log(f"  WARNING: missing companion file(s) for chunk {chunk}, skipping")
            continue
        chunks.append((chunk, vf, others))
    if limit:
        chunks = chunks[:limit]
    return chunks


def build_experiment_shim(expt, limit=None):
    case = f"gfdl_{expt}_moc"
    src_dir = os.path.join(GFDL_ROOT, expt, STATIC_SRC_STREAM, "ts", "annual", "5yr")
    static_src = os.path.join(GFDL_ROOT, expt, STATIC_SRC_STREAM, f"{STATIC_SRC_STREAM}.static.nc")
    out_root = os.path.join(GFDL_ROOT, expt, "cesm_format_moc")
    out_hist = os.path.join(out_root, "ocn", "hist")
    os.makedirs(out_hist, exist_ok=True)

    # Reuses the exact same shared static file as gfdl_shim_moc.py -- verified
    # this session to be the identical native grid (same yh/yq/xh/xq sizes,
    # same geolat_c/deptho/dxt/dyt/dxCu/dyCu/dxCv/dyCv/basin fields) as this
    # z-stream's own static file, so there's nothing new to build here.
    build_shared_static(static_src)

    static_link = os.path.join(out_hist, f"{case}.mom6.h.static.nc")
    if not os.path.lexists(static_link):
        os.symlink(SHARED_STATIC_FILE, static_link)
        log(f"[{case}] symlinked static grid -> {SHARED_STATIC_FILE}")

    xmlquery_path = os.path.join(out_root, "xmlquery")
    if not os.path.exists(xmlquery_path):
        build_xmlquery(out_root, case)
        log(f"[{case}] wrote xmlquery shim in {out_root}")

    chunks = find_chunks(src_dir, limit)
    log(f"[{case}] {len(chunks)} z-stream chunk(s) to merge from {src_dir}")
    for chunk, vf, others in chunks:
        t0 = time.time()
        year = chunk[:4]
        merged_out = os.path.join(out_hist, f"{case}.mom6.h.z.{year}-01.nc")
        if os.path.exists(merged_out):
            log(f"  [{chunk}] {os.path.basename(merged_out)} already exists, skipping")
            continue
        tmp_out = merged_out + ".building"
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        shutil.copy(vf, tmp_out)
        for name, path in others.items():
            run(["ncks", "-A", "-v", name, path, tmp_out])
        os.rename(tmp_out, merged_out)
        log(f"  [{chunk}] -> {os.path.basename(merged_out)} (vmo+vhml+vo) "
            f"in {time.time()-t0:.1f}s")

    log(f"[{case}] done. z-stream chunks added alongside existing rho2 chunks in {out_hist}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["CTRL", "bANN", "uANN", "bANN_uANN"])
    ap.add_argument("--limit", type=int, default=None, help="only use the first N available chunks (for testing)")
    args = ap.parse_args()
    build_experiment_shim(args.experiment, args.limit)


if __name__ == "__main__":
    main()
