#!/usr/bin/env python
"""Compute the native-grid velocity/transport diagnostics for
gfdl_dynamics.ipynb from GFDL's real umo/uo/vo/thetao/so (ocean_annual_z
stream, native tripolar grid, annual resolution): barotropic (depth-
integrated) streamfunction, current speed at the surface and at 2500 m
(matching dynamics.ipynb's own two current-speed diagnostics), the
equatorial undercurrent section at 165E, the ACC/Drake Passage
zonal-velocity section at 70W, and the Drake Passage transport annual time
series.

All reuse the SAME per-experiment read of umo/uo/vo/thetao/so. Every
diagnostic here is either a cheap single-depth-level slice (surface, 2500
m) or a cheap single-column read (the two latitude-depth sections, and the
Drake transport time series) -- none of them need a full 3-D read of the
whole grid, so this is much lighter than an earlier version of this script
that computed a full-depth time-mean uo/vo for a "near-bottom" diagnostic
later dropped in favor of the fixed 2500 m depth dynamics.ipynb actually
uses. Averaging window: 1988-01-01 to 2012-01-01, matching every other
GFDL diagnostic this session -- except the Drake Passage transport, which
(like the AMOC/heat-transport time series) is kept as a genuine per-year
time series over each experiment's full available record, not collapsed to
a time mean.

The two latitude-depth sections (EUC at 165E, Drake at 70W) also save
thetao/so at the same native column, so gsw.sigma2(so, thetao) can overlay
potential-density isolines on the velocity section -- computed on GFDL's
own native grid (the same grid uo/vo live on), not the regridded 1x1deg
grid used for Sections 1-2, so there's no cross-grid mismatch.

Usage:
    python gfdl_native_velocity.py <CTRL|bANN|uANN|bANN_uANN>
"""
import argparse
import glob
import os

import numpy as np
import xarray as xr

GFDL_ROOT = "/glade/derecho/scratch/pavelp/GFDL"
STATIC_FILE = "/glade/derecho/scratch/pavelp/GFDL/_shared_cesm_format_moc_grid/gfdl.mom6.h.static.nc"
AVG_START = "1988-01-01"
AVG_END = "2012-01-01"

# Drake Passage: a single xq column near 70W, integrated over its
# Southern-Ocean latitude range (native geolon convention here is ~-300..60,
# so 70W is genuinely -70, not 290).
DRAKE_LON = -70.0
DRAKE_LAT_RANGE = (-71.0, -55.0)

# Equatorial undercurrent section: fixed longitude, matching dynamics.ipynb's
# own SECTION_LON (165E, in the Tropical Pacific).
EUC_LON = 165.0

# Current speed: surface (z_l=0) and 2500 m, matching dynamics.ipynb's own
# two current-speed diagnostics exactly (not a "near-bottom" -- most native
# grid columns don't reach anywhere near GFDL's deepest z-level).
SPEED_2500M = 2500.0


def _open_var(src_dir, v):
    files = sorted(glob.glob(os.path.join(src_dir, f"*.{v}.nc")))
    return xr.open_mfdataset(files, chunks={"time": 5})[v]


def _find_column(geolon, geolat, target_lon, target_lat):
    """Nearest column index to (target_lon, target_lat) on a curvilinear
    2-D (geolon, geolat) grid: pick the row nearest target_lat at the
    grid's central column, then the column nearest target_lon (circular
    longitude distance) within that row -- same technique the Drake Passage
    transport calculation already used. Returns (row_idx, col_idx)."""
    row_idx = int(np.abs(geolat[:, geolat.shape[1] // 2] - target_lat).argmin())
    dlon = np.abs(((geolon[row_idx, :] - target_lon + 180) % 360) - 180)
    col_idx = int(dlon.argmin())
    return row_idx, col_idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["CTRL", "bANN", "uANN", "bANN_uANN"])
    args = ap.parse_args()
    expt = args.experiment
    case = f"gfdl_{expt}"

    grd = xr.open_dataset(STATIC_FILE, decode_times=False)

    src_dir = os.path.join(GFDL_ROOT, expt, "ocean_annual_z", "ts", "annual", "5yr")
    print(f"[{case}] reading umo/uo/vo/thetao/so from {src_dir}")
    umo = _open_var(src_dir, "umo")       # (time, z_l, yh, xq)
    uo = _open_var(src_dir, "uo")         # (time, z_l, yh, xq)
    vo = _open_var(src_dir, "vo")         # (time, z_l, yq, xh)
    thetao = _open_var(src_dir, "thetao")  # (time, z_l, yh, xh)
    so = _open_var(src_dir, "so")          # (time, z_l, yh, xh)

    os.makedirs("ncfiles", exist_ok=True)

    # === 1. Barotropic (depth-integrated) streamfunction ===
    # Time-mean over 1988-2012, following the CDFtools algorithm
    # (Experiment.compute_barotropic_streamfunction, ported to GFDL's native
    # grid): sum umo over depth, then cumulatively integrate meridionally
    # from the southern boundary, convert kg/s -> Sv.
    print(f"[{case}] computing barotropic streamfunction...")
    umo_mean = umo.sel(time=slice(AVG_START, AVG_END)).mean("time").compute()   # (z_l, yh, xq)
    wet_u = umo_mean.notnull().any("z_l").values                                # (yh, xq)
    umo_depth_sum = umo_mean.sum("z_l").fillna(0.0)                             # (yh, xq)
    psi_vals = -umo_depth_sum.cumsum("yh").values / 1e9                        # Sv, (yh, xq)

    # This cumulative sum represents transport through the NORTH face of
    # each yh row -- under GFDL's 'outer' yq convention (len(yq)=len(yh)+1,
    # yq[0] = southern boundary before any transport is added), that's
    # genuinely yq[1:], not yh itself. The static file's own yq/xq are bare
    # dimensions (no coordinate values) -- vo's own file carries a real yq
    # coordinate (same native grid), so borrow that instead.
    yq_bsf = vo["yq"].values[1:]
    xq = umo["xq"].values
    psi_bsf = xr.DataArray(psi_vals, dims=["yq", "xq"], coords={"yq": yq_bsf, "xq": xq})
    wet_u_bsf = xr.DataArray(wet_u, dims=["yq", "xq"], coords={"yq": yq_bsf, "xq": xq})

    # Zero the streamfunction over the Americas (same reference point
    # Experiment.compute_barotropic_streamfunction uses on CESM's grid;
    # GFDL's native geolon convention here is the same raw -300..60 range).
    psi_bsf = psi_bsf - psi_bsf.sel(xq=-80, yq=40, method="nearest")
    psi_bsf = psi_bsf.where(wet_u_bsf)
    psi_bsf.name = "psi_barotropic"

    # geolon_c/geolat_c are plain data variables on the static file (whose
    # own yq/xq are bare dimensions, not coordinates) -- index by position
    # (drop row 0) rather than by value.
    geolonb = grd["geolon_c"].isel(yq=slice(1, None)).values
    geolatb = grd["geolat_c"].isel(yq=slice(1, None)).values

    bsf_out = psi_bsf.to_dataset()
    bsf_out["geolonb"] = (("yq", "xq"), geolonb)
    bsf_out["geolatb"] = (("yq", "xq"), geolatb)
    bsf_out.attrs.update({"description": "Barotropic (depth-integrated) streamfunction",
                          "units": "Sv", "reference": "zero at Americas (xq=-80, yq=40)",
                          "start_date": AVG_START, "end_date": AVG_END, "casename": case})
    bsf_out.to_netcdf(f"ncfiles/{case}_barotropic_streamfunction.nc")
    print(f"[{case}] wrote ncfiles/{case}_barotropic_streamfunction.nc")

    # === 2. Current speed (surface and 2500 m) ===
    print(f"[{case}] computing surface and 2500 m current speed...")
    uo_surf = uo.sel(time=slice(AVG_START, AVG_END)).isel(z_l=0).mean("time").compute()   # (yh, xq)
    vo_surf = vo.sel(time=slice(AVG_START, AVG_END)).isel(z_l=0).mean("time").compute()   # (yq, xh)
    uo_2500 = (uo.sel(time=slice(AVG_START, AVG_END)).sel(z_l=SPEED_2500M, method="nearest")
               .mean("time").compute())   # (yh, xq)
    vo_2500 = (vo.sel(time=slice(AVG_START, AVG_END)).sel(z_l=SPEED_2500M, method="nearest")
               .mean("time").compute())   # (yq, xh)

    # Target tracer-grid axes: uo's own real yh and vo's own real xh (the
    # static file's yh/xh are bare dimensions without coordinate values).
    xh = vo["xh"].values
    yh = uo["yh"].values

    def _speed_on_tgrid(uo_2d, vo_2d):
        """uo_2d (yh, xq), vo_2d (yq, xh) -> speed on the tracer grid (yh, xh).
        Drops the leftover scalar z_l coordinate (from the depth selection
        upstream) -- otherwise combining the surface and 2500 m speed
        DataArrays (each carrying a different scalar z_l value) into one
        Dataset raises a MergeError."""
        u_h = uo_2d.interp(xq=xh).rename({"xq": "xh"})
        v_h = vo_2d.interp(yq=yh).rename({"yq": "yh"})
        return np.sqrt(u_h ** 2 + v_h ** 2).reset_coords("z_l", drop=True)

    speed_surf = _speed_on_tgrid(uo_surf, vo_surf)
    speed_surf.name = "speed_surf"
    speed_2500 = _speed_on_tgrid(uo_2500, vo_2500)
    speed_2500.name = "speed_2500m"

    speed_out = xr.Dataset({"speed_surf": speed_surf, "speed_2500m": speed_2500})
    speed_out.attrs.update({"description": "Current speed sqrt(uo^2+vo^2) at the surface "
                                           "(shallowest z_l) and at 2500 m depth (nearest "
                                           "z_l), interpolated to the tracer grid",
                            "units": "m s-1", "depth_2500m_actual_m": float(uo_2500["z_l"].values),
                            "start_date": AVG_START, "end_date": AVG_END, "casename": case})
    speed_out.to_netcdf(f"ncfiles/{case}_current_speed.nc")
    print(f"[{case}] wrote ncfiles/{case}_current_speed.nc "
         f"(2500m level actually at {float(uo_2500['z_l'].values):.1f} m)")

    # === 3. Equatorial undercurrent section at 165E (latitude-depth) ===
    print(f"[{case}] computing equatorial undercurrent section at {EUC_LON}E...")
    geolon_u = grd["geolon_u"].values   # (yh, xq)
    geolat_u = grd["geolat_u"].values
    geolon_t = grd["geolon"].values     # (yh, xh)
    geolat_t = grd["geolat"].values

    row_idx_euc_u, xq_idx_euc = _find_column(geolon_u, geolat_u, EUC_LON, 0.0)
    row_idx_euc_t, xh_idx_euc = _find_column(geolon_t, geolat_t, EUC_LON, 0.0)

    uo_euc = uo.isel(xq=xq_idx_euc).sel(time=slice(AVG_START, AVG_END)).mean("time").compute()      # (z_l, yh)
    thetao_euc = thetao.isel(xh=xh_idx_euc).sel(time=slice(AVG_START, AVG_END)).mean("time").compute()  # (z_l, yh)
    so_euc = so.isel(xh=xh_idx_euc).sel(time=slice(AVG_START, AVG_END)).mean("time").compute()          # (z_l, yh)

    euc_out = xr.Dataset({"uo": uo_euc.rename({"yh": "yh_u"}),
                          "thetao": thetao_euc, "so": so_euc})
    euc_out.attrs.update({"description": f"Zonal velocity (nearest native xq column) and "
                                         f"thetao/so (nearest native xh column) at {EUC_LON}E, "
                                         "latitude-depth section",
                          "units": "m s-1 (uo), degC (thetao), psu (so)",
                          "lon_u_used": float(geolon_u[row_idx_euc_u, xq_idx_euc]),
                          "lon_t_used": float(geolon_t[row_idx_euc_t, xh_idx_euc]),
                          "start_date": AVG_START, "end_date": AVG_END, "casename": case})
    euc_out.to_netcdf(f"ncfiles/{case}_euc_section.nc")
    print(f"[{case}] wrote ncfiles/{case}_euc_section.nc")

    # === 4. ACC/Drake Passage: zonal-velocity section (latitude-depth) and
    #        annual transport time series, both at the same 70W column ===
    print(f"[{case}] computing Drake Passage section and transport at {DRAKE_LON}W...")
    lat_mid = 0.5 * (DRAKE_LAT_RANGE[0] + DRAKE_LAT_RANGE[1])
    # Nearest xq column to Drake Passage longitude: pick the column whose
    # geolon_u is closest to DRAKE_LON, using the row nearest the Drake
    # Passage latitude band's center (native grid is curvilinear, so a
    # single-longitude column isn't literally one xq index at every row --
    # this picks the best xq index at the representative latitude, then
    # uses that fixed xq column across the full latitude band, matching how
    # a real hydrographic section is defined).
    row_mid, xq_idx = _find_column(geolon_u, geolat_u, DRAKE_LON, lat_mid)
    _, xh_idx_drake = _find_column(geolon_t, geolat_t, DRAKE_LON, lat_mid)

    uo_drake_sec = uo.isel(xq=xq_idx).sel(time=slice(AVG_START, AVG_END)).mean("time").compute()          # (z_l, yh)
    thetao_drake = thetao.isel(xh=xh_idx_drake).sel(time=slice(AVG_START, AVG_END)).mean("time").compute()  # (z_l, yh)
    so_drake = so.isel(xh=xh_idx_drake).sel(time=slice(AVG_START, AVG_END)).mean("time").compute()          # (z_l, yh)

    drake_sec_out = xr.Dataset({"uo": uo_drake_sec.rename({"yh": "yh_u"}),
                                "thetao": thetao_drake, "so": so_drake})
    drake_sec_out.attrs.update({"description": f"Zonal velocity (nearest native xq column) and "
                                               f"thetao/so (nearest native xh column) at {DRAKE_LON}W, "
                                               "latitude-depth section across the ACC",
                                "units": "m s-1 (uo), degC (thetao), psu (so)",
                                "lon_u_used": float(geolon_u[row_mid, xq_idx]),
                                "start_date": AVG_START, "end_date": AVG_END, "casename": case})
    drake_sec_out.to_netcdf(f"ncfiles/{case}_drake_section.nc")
    print(f"[{case}] wrote ncfiles/{case}_drake_section.nc")

    umo_drake = umo.isel(xq=xq_idx)   # (time, z_l, yh)
    lat_col = geolat_u[:, xq_idx]
    lat_mask = (lat_col >= DRAKE_LAT_RANGE[0]) & (lat_col <= DRAKE_LAT_RANGE[1])
    umo_drake_ann = umo_drake.sel(yh=umo_drake["yh"].values[lat_mask]).sum(("z_l", "yh")).compute() / 1e9  # Sv
    umo_drake_ann.name = "drake_transport"
    drake_out = umo_drake_ann.to_dataset()
    drake_out.attrs.update({"description": "ACC (Drake Passage) volume transport, annual",
                            "units": "Sv", "xq_index": xq_idx, "drake_lon_used": float(geolon_u[row_mid, xq_idx]),
                            "casename": case})
    drake_out.to_netcdf(f"ncfiles/{case}_drake_transport.nc")
    print(f"[{case}] wrote ncfiles/{case}_drake_transport.nc "
         f"(xq_index={xq_idx}, lon={float(geolon_u[row_mid, xq_idx]):.1f})")

    print(f"[{case}] done.")


if __name__ == "__main__":
    main()
