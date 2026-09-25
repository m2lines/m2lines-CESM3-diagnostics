#!/usr/bin/env python
"""Build a lightweight CESM-shaped case directory for a GFDL experiment,
WITHOUT copying or transforming the actual GFDL model output data.

GFDL's data stays on its own native 1x1 degree grid. mom6-tools/Experiment
only ever reach into a CASEROOT through cime_xmlquery() (see m6toolbox.py),
so a fake CASEROOT containing a tiny "xmlquery" shim script is enough to
satisfy that contract -- no real CIME case needed.

The only thing GFDL's own files don't already satisfy is a handful of NAMES:
  - mom6_tools.TS_levels hardcodes the dimension names 'yh'/'xh' (not just as
    labels -- myStats_da's default `dims=('yh','xh')` parameter actually
    reduces over them) and the variable name 'so' (GFDL calls it 'salt').
    Dimension/variable names are structural HDF5 metadata, not something a
    plain symlink can alias -- but NCO's `ncrename` on a netCDF4 file is a
    metadata-only rename (no data rewrite: ~1.5s for a 1.1GB file, confirmed),
    which is what this script uses instead of ever opening/recomputing the
    actual arrays.
  - MOM6grid.py unconditionally accesses nc.geolon_u (MOM6grid.py:38) to
    handle C-grid longitude wraparound. GFDL's static file only has tracer-
    point fields (geolon/geolat/wet/deptho/areacello), no C-grid corner
    points at all. Since none of this repo's T/S diagnostics ever touch
    geolon_u itself (only MOM6grid.py's internal wraparound calc does), a
    placeholder copy of geolon is enough to stop the crash.
  - The static file also has no 1D yh/xh coordinate variables at all (bare
    dims) -- added from a data file's own 1D lat/lon values.

For each of GFDL's existing 5-year chunks, this produces one merged, renamed
file combining thetao+so (still not interpolated -- just NCO copy/append).
Originally this tried to leave thetao and salt-renamed-to-so as two separate
files side by side, relying on xr.open_mfdataset's combine="by_coords" to
merge same-time, different-variable files automatically -- that combining
does happen, but only *after* TS_levels.py's own `preprocess` callback runs
per-file first (`ds[['thetao','so','time']]`), so a thetao-only file
KeyErrors on 'so' before the merge ever gets a chance to run. `ncks -A`
(NCO's append) copies the 'so' variable's bytes into the thetao file directly
-- still not a value transform (no interpolation, no recompute), just an
~17s bulk copy per chunk instead of ncrename's ~1.5s metadata-only rename.

Usage:
    python gfdl_shim.py <CTRL|bANN|uANN|bANN_uANN> [--limit N]
"""
import argparse
import glob
import os
import stat
import subprocess
import sys

import netCDF4

GFDL_ROOT = "/glade/derecho/scratch/pavelp/GFDL"
SHARED_STATIC_DIR = os.path.join(GFDL_ROOT, "_shared_cesm_format_grid")
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
  *) echo "ERROR: gfdl xmlquery shim has no answer for '$1'" >&2; exit 1 ;;
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


def build_shared_static(gfdl_static_src, sample_data_file):
    """Build (once) the augmented GFDL static/grid file: dims renamed to
    yh/xh, 1D yh/xh coordinates added, and a geolon_u placeholder added so
    MOM6grid.py's unconditional wraparound calc doesn't crash. No model
    output data is touched by this function at all -- the static file is a
    small (~20MB) grid-description file, not the T/S time series.

    geolon/geolat are left in GFDL's own natural 0-360 convention, UNTOUCHED --
    an earlier version of this function rewrapped geolon to CESM's raw
    -287..+73 range specifically so mom6_tools.m6toolbox.genBasinMasks'
    hardcoded seed points would land in the right place on this grid. That's
    no longer needed: genBasinMasks' cutting-line geometry is tuned for
    CESM's specific tripolar grid and gives wrong/empty basins here regardless
    (the notebooks build basin masks from GFDL's own native `basin` field
    instead, see gfdl_horizontal_biases.ipynb/gfdl_vertical_sections.ipynb).
    Worse, the wrap made `geolon` non-monotonic along `xh` (everything above
    73 degrees jumps backwards by 360 in-place, mid-row) -- pcolormesh
    interprets geolon/geolat as 2D cell centers, and a non-monotonic center
    sequence produces degenerate quads: visible as stripe artifacts and,
    apparently, colored cells bleeding onto land at the wrap seam. The fix
    is to not wrap it at all, and to plot maps against the clean 1D xh/yh
    axes (added below) rather than the 2D geolon/geolat fields -- exact for
    a regular grid, and immune to this whole class of bug.
    """
    if os.path.exists(SHARED_STATIC_FILE):
        log(f"[shared static] {SHARED_STATIC_FILE} already exists, skipping build")
        return
    os.makedirs(SHARED_STATIC_DIR, exist_ok=True)
    log(f"[shared static] ncrename dims lat->yh, lon->xh: {gfdl_static_src} -> {SHARED_STATIC_FILE}")
    run(["ncrename", "-O", "-d", "lat,yh", "-d", "lon,xh", gfdl_static_src, SHARED_STATIC_FILE])

    log("[shared static] adding yh/xh 1D coords, geolon_u placeholder (metadata only, no big data touched)")
    src = netCDF4.Dataset(sample_data_file, "r")
    yh_vals = src.variables["lat"][:]
    xh_vals = src.variables["lon"][:]
    src.close()

    nc = netCDF4.Dataset(SHARED_STATIC_FILE, "a")
    v_yh = nc.createVariable("yh", "f8", ("yh",))
    v_yh[:] = yh_vals
    v_xh = nc.createVariable("xh", "f8", ("xh",))
    v_xh[:] = xh_vals

    # Placeholder only: prevents MOM6grid.py:38's unconditional nc.geolon_u
    # access from raising AttributeError. Never used in any T/S computation
    # in this repo -- a plain copy of geolon (untouched, natural 0-360) suffices.
    v_u = nc.createVariable("geolon_u", "f4", ("yh", "xh"))
    v_u[:] = nc.variables["geolon"][:]
    nc.close()
    log(f"[shared static] done: {SHARED_STATIC_FILE}")


def find_chunks(src_dir, limit=None):
    thetao_files = sorted(glob.glob(os.path.join(src_dir, "*.thetao.nc")))
    chunks = []
    for tf in thetao_files:
        chunk = os.path.basename(tf).split(".")[1]
        sf = tf.replace(".thetao.nc", ".salt.nc")
        if not os.path.exists(sf):
            log(f"  WARNING: no matching salt file for chunk {chunk}, skipping")
            continue
        chunks.append((chunk, tf, sf))
    if limit:
        chunks = chunks[:limit]
    return chunks


def build_experiment_shim(expt, limit=None):
    case = f"gfdl_{expt}"
    src_dir = os.path.join(GFDL_ROOT, expt, "ocean_monthly_z_1x1deg", "ts", "monthly", "5yr")
    static_src = os.path.join(GFDL_ROOT, expt, "ocean_monthly_z_1x1deg", "ocean_monthly_z_1x1deg.static.nc")
    out_root = os.path.join(GFDL_ROOT, expt, "cesm_format")
    out_hist = os.path.join(out_root, "ocn", "hist")
    os.makedirs(out_hist, exist_ok=True)

    chunks = find_chunks(src_dir, limit)
    build_shared_static(static_src, chunks[0][1])

    static_link = os.path.join(out_hist, f"{case}.mom6.h.static.nc")
    if os.path.lexists(static_link):
        os.remove(static_link)
    os.symlink(SHARED_STATIC_FILE, static_link)
    log(f"[{case}] symlinked static grid -> {SHARED_STATIC_FILE}")

    build_xmlquery(out_root, case)
    log(f"[{case}] wrote xmlquery shim in {out_root}")

    log(f"[{case}] {len(chunks)} chunk(s) to rename+merge from {src_dir}")
    for chunk, tf, sf in chunks:
        year = chunk[:4]
        merged_out = os.path.join(out_hist, f"{case}.mom6.h.z.{year}-01.nc")
        so_tmp = merged_out + ".so_tmp.nc"
        if os.path.exists(merged_out):
            log(f"  [{chunk}] {os.path.basename(merged_out)} already exists, skipping")
            continue
        run(["ncrename", "-O", "-d", "lat,yh", "-d", "lon,xh", tf, merged_out])
        run(["ncrename", "-O", "-d", "lat,yh", "-d", "lon,xh", "-v", "salt,so", sf, so_tmp])
        # GFDL keeps all 35 of its own z_l levels untouched -- the extra 6500m
        # level (which CESM/WOA don't have) is handled by padding the obs product
        # with a matching NaN level instead (see regrid_obs_to_gfdl_grid.py).
        run(["ncks", "-A", "-v", "so", so_tmp, merged_out])
        os.remove(so_tmp)
        log(f"  [{chunk}] -> {os.path.basename(merged_out)} (thetao+so)")

    log(f"[{case}] done. CASEROOT for yaml = {out_root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["CTRL", "bANN", "uANN", "bANN_uANN"])
    ap.add_argument("--limit", type=int, default=None, help="only use the first N available chunks (for testing)")
    args = ap.parse_args()
    build_experiment_shim(args.experiment, args.limit)


if __name__ == "__main__":
    main()
