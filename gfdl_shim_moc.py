#!/usr/bin/env python
"""Build a native-MOM6-grid CESM-shaped case directory for GFDL MOC
diagnostics (mom6_tools.moc_sigma2), separate from gfdl_shim.py's
regridded-grid T/S shim since these genuinely need different grids: native
tripolar C-grid for transport-weighted overturning vs GFDL's own regridded
1x1 degree grid for T/S.

z-space MOC (mom6_tools.moc) is NOT built here: that script reads its
vmo/vhml/vhGM from the *'z' Fnames stream* (not 'rho2'), and GFDL's
z-coordinate stream (ocean_monthly_z_1x1deg) is regridded and has no
transport variables at all -- umo/vmo only exist natively, in density
(rho2) coordinates, via ocean_annual_rho2. So only sigma2 (density-space)
MOC is computable for GFDL.

GFDL's ocean_annual_rho2 stream already has clean, CESM-standard variable
names (umo, vmo, uhml, vhml, volcello) and dims (yh/xh/yq/xq) on a genuine
native MOM6 C-grid static file (with basin, Coriolis, dxCu/dyCu, etc.) --
no renaming needed at all, unlike the regridded grid. The only gap is
mom6_tools.moc_sigma2's vhGM (GM-parameterization overturning), which
GFDL's diag_table never saved. The script already has a built-in fallback
(zero-fill any of vmo/vhml/vhGM that's missing), but that fallback needs a
'vo' variable purely as a shape/dtype template, regardless of which
variable is actually missing. A zero-filled 'vo' (same shape as vmo)
satisfies that template requirement without fabricating any physics: the
real "total" (vmo) and "submesoscale" (vhml) overturning are both
unaffected; only the mesoscale/GM component (genuinely missing) comes out
as zero, and is flagged as unavailable rather than presented as real.

Usage:
    python gfdl_shim_moc.py <CTRL|bANN|uANN|bANN_uANN> [--limit N]
"""
import argparse
import glob
import os
import shutil
import stat
import subprocess
import time

import netCDF4

GFDL_ROOT = "/glade/derecho/scratch/pavelp/GFDL"
SHARED_STATIC_DIR = os.path.join(GFDL_ROOT, "_shared_cesm_format_moc_grid")
SHARED_STATIC_FILE = os.path.join(SHARED_STATIC_DIR, "gfdl.mom6.h.static.nc")

XMLQUERY_TEMPLATE = """#!/bin/bash
# Fake CIME xmlquery shim standing in for a real CESM CASEROOT, for {case}.
# Answers exactly the variables mom6-tools/Experiment ever query via cime_xmlquery():
# CASE, DOUT_S, DOUT_S_ROOT, RUNDIR, RUN_STARTDATE (see m6toolbox.cime_xmlquery).
if [ "$1" == "-N" ]; then shift; fi
if [ "$1" == "--value" ]; then shift; fi
case "$1" in
  CASE) printf '%s' "{case}" ;;
  DOUT_S) printf '%s' "TRUE" ;;
  DOUT_S_ROOT) printf '%s' "{root}" ;;
  RUNDIR) printf '%s' "{root}/run" ;;
  RUN_STARTDATE) printf '%s' "1958-01-01" ;;
  *) echo "ERROR: gfdl moc xmlquery shim has no answer for '$1'" >&2; exit 1 ;;
esac
"""


def log(msg):
    print(msg, flush=True)


def run(cmd):
    subprocess.run(cmd, check=True)


def build_xmlquery(out_root, case):
    path = os.path.join(out_root, "xmlquery")
    with open(path, "w") as f:
        f.write(XMLQUERY_TEMPLATE.format(case=case, root=out_root))
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def build_shared_static(native_static_src):
    """Copy GFDL's native-grid static file unmodified -- it's already a
    complete, standard MOM6 C-grid file (Coriolis, dxCu/dyCu/dxCv/dyCv,
    geolon_u/_c/_v, basin, ...), unlike the regridded grid used for T/S."""
    if os.path.exists(SHARED_STATIC_FILE):
        log(f"[shared moc static] {SHARED_STATIC_FILE} already exists, skipping build")
        return
    os.makedirs(SHARED_STATIC_DIR, exist_ok=True)
    log(f"[shared moc static] copying {native_static_src} -> {SHARED_STATIC_FILE}")
    shutil.copy(native_static_src, SHARED_STATIC_FILE)


def find_chunks(src_dir, limit=None):
    # mom6_tools.moc_sigma2's preprocess only ever selects vmo/vhml/vhGM/volcello
    # (see module docstring) -- umo/uhml are meridional-overturning-irrelevant
    # (zonal transport components) and would roughly double the I/O for nothing.
    vmo_files = sorted(glob.glob(os.path.join(src_dir, "*.vmo.nc")))
    chunks = []
    for vf in vmo_files:
        chunk = os.path.basename(vf).split(".")[1]
        others = {
            "vhml": vf.replace(".vmo.nc", ".vhml.nc"),
            "volcello": vf.replace(".vmo.nc", ".volcello.nc"),
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
    src_dir = os.path.join(GFDL_ROOT, expt, "ocean_annual_rho2", "ts", "annual", "5yr")
    static_src = os.path.join(GFDL_ROOT, expt, "ocean_annual_rho2", "ocean_annual_rho2.static.nc")
    out_root = os.path.join(GFDL_ROOT, expt, "cesm_format_moc")
    out_hist = os.path.join(out_root, "ocn", "hist")
    os.makedirs(out_hist, exist_ok=True)

    build_shared_static(static_src)

    static_link = os.path.join(out_hist, f"{case}.mom6.h.static.nc")
    if os.path.lexists(static_link):
        os.remove(static_link)
    os.symlink(SHARED_STATIC_FILE, static_link)
    log(f"[{case}] symlinked static grid -> {SHARED_STATIC_FILE}")

    build_xmlquery(out_root, case)
    log(f"[{case}] wrote xmlquery shim in {out_root}")

    chunks = find_chunks(src_dir, limit)
    log(f"[{case}] {len(chunks)} chunk(s) to merge from {src_dir}")
    for chunk, vf, others in chunks:
        t0 = time.time()
        year = chunk[:4]
        merged_out = os.path.join(out_hist, f"{case}.mom6.h.rho2.{year}-01.nc")
        if os.path.exists(merged_out):
            log(f"  [{chunk}] {os.path.basename(merged_out)} already exists, skipping")
            continue
        # Build under a .building temp name and rename into place only once
        # fully done -- a job killed mid-chunk (e.g. PBS walltime, node
        # preemption) must not leave a same-named, seemingly-complete file
        # that a later resume run would silently treat as finished.
        tmp_out = merged_out + ".building"
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        shutil.copy(vf, tmp_out)
        for name, path in others.items():
            run(["ncks", "-A", "-v", name, path, tmp_out])

        # Fake zero 'vo' -- purely a shape/dtype template for moc_sigma2.py's
        # own built-in "if missing, zero-fill" fallback (see module docstring).
        # Uncompressed (matching the source files) -- compression made writing
        # this noticeably slower for no benefit on an all-zero array this size.
        # (Creating this *before* the vhml/volcello appends instead, on the
        # smaller not-yet-merged file, was tried and measured slower overall
        # -- the cost is apparently tied to the file's HDF5 chunk/btree
        # layout churn from adding a new variable in append mode, not to the
        # file's size at the time, so there's no benefit to reordering.)
        nc = netCDF4.Dataset(tmp_out, "a")
        vmo_var = nc.variables["vmo"]
        vo = nc.createVariable("vo", vmo_var.dtype, vmo_var.dimensions)
        vo[:] = 0.0
        nc.close()
        os.rename(tmp_out, merged_out)
        log(f"  [{chunk}] -> {os.path.basename(merged_out)} (vmo+vhml+volcello+fake vo) "
            f"in {time.time()-t0:.1f}s")

    log(f"[{case}] done. CASEROOT for yaml = {out_root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["CTRL", "bANN", "uANN", "bANN_uANN"])
    ap.add_argument("--limit", type=int, default=None, help="only use the first N available chunks (for testing)")
    args = ap.parse_args()
    build_experiment_shim(args.experiment, args.limit)


if __name__ == "__main__":
    main()
