"""Shared Experiment class for CESM3 / MOM6 diagnostics notebooks.

This module centralises the ``Experiment`` helper so it can be reused across
multiple analysis notebooks. Each ``Experiment`` bundles together:

* the paths to a CESM/MOM6 case (read from a diagnostics YAML config),
* the model grid / basin masks (loaded lazily, best-effort), and
* convenient readers for the *postprocessed* NetCDF files produced by the
  diagnostics pipeline (temperature drift, MOC, MLD, ice volume, ...).

Design notes
------------
The postprocessed files are named ``<name>_<metric>.nc`` and live in ``folder``.
The ``name`` is an explicit label for the experiment (e.g. ``"CTRL"``,
``"forced_bANN_1.0"``) and is **independent** of the CIME ``CASE``/``SNAME``
value. This matters because some experiments (e.g. CTRL) are run by a different
user, yet their postprocessed data is stored locally under a chosen ``name``.

Initialisation is *read-only friendly*: querying CIME and loading the grid are
wrapped so that a missing/inaccessible case directory does not prevent you from
reading postprocessed data.
"""

from glob import glob
import os

import gsw
import numpy as np
import xarray as xr
import yaml
from dask.diagnostics import ProgressBar

from mom6_tools.MOM6grid import MOM6grid
from mom6_tools.m6toolbox import (
    add_global_attrs,
    cime_xmlquery,
    genBasinMasks,
    weighted_temporal_mean_vars,
)


class Experiment:
    """A single model experiment with readers for its postprocessed diagnostics.

    Parameters
    ----------
    diag_config_yml_path : str
        Path to the diagnostics YAML config (contains ``Case``, ``Avg`` and
        ``Fnames`` sections).
    folder : str
        Directory containing the postprocessed ``<name>_<metric>.nc`` files.
    name : str, optional
        Label used as the prefix for postprocessed files. Defaults to the YAML
        file name without extension (e.g. ``CTRL.yaml`` -> ``"CTRL"``).
    load_grid : bool, optional
        If ``True`` (default) attempt to load the model grid and basin masks.
        Failures are caught so read-only workflows still succeed.
    verbose : bool, optional
        Print progress information during initialisation.
    """

    def __init__(self, diag_config_yml_path, folder, name=None,
                 load_grid=True, verbose=True):
        self.yaml_path = diag_config_yml_path
        self.folder = folder
        self.name = name or os.path.splitext(os.path.basename(diag_config_yml_path))[0]

        # Read config file
        diag_config_yml = yaml.load(open(diag_config_yml_path, 'r'), Loader=yaml.Loader)
        self.config = diag_config_yml

        caseroot = diag_config_yml['Case']['CASEROOT']

        # Organise paths in a lightweight namespace (kept as ``args`` for
        # backwards compatibility with the original notebook code).
        class args:
            pass

        args.caseroot = caseroot
        args.nw = 1
        args.savefigs = False

        # --- CIME query (best-effort) --------------------------------------
        # Needed for the compute_* helpers that read raw model history, but not
        # for reading postprocessed data. Guarded so read-only init never fails.
        args.casename = None
        args.rundir = None
        args.OUTDIR = None
        args.ICEDIR = None
        args.ATMDIR = None
        try:
            casename = cime_xmlquery(caseroot, 'CASE')
            DOUT_S = cime_xmlquery(caseroot, 'DOUT_S')
            rundir = cime_xmlquery(caseroot, 'RUNDIR')
            if DOUT_S:
                OUTDIR = cime_xmlquery(caseroot, 'DOUT_S_ROOT') + '/ocn/hist/'
            else:
                OUTDIR = rundir

            args.casename = casename
            args.rundir = rundir
            args.OUTDIR = OUTDIR
            args.ICEDIR = cime_xmlquery(caseroot, 'DOUT_S_ROOT') + '/ice/hist'
            args.ATMDIR = cime_xmlquery(caseroot, 'DOUT_S_ROOT') + '/atm/hist'

            if verbose:
                print(f'[{self.name}] casename={casename}, rundir={rundir}')
        except Exception as e:
            if verbose:
                print(f'[{self.name}] CIME query skipped (read-only mode): {e}')

        # File-name conventions (used by compute_* helpers reading raw history).
        fnames = diag_config_yml.get('Fnames', {})
        cn = args.casename or self.name
        args.static = cn + fnames.get('static', '')
        args.native = cn + fnames.get('native', '')
        args.z = cn + fnames.get('z', '')
        args.ice = cn + fnames.get('ice', '')
        args.geom = cn + fnames.get('geom', '')

        avg = diag_config_yml.get('Avg', {})
        args.start_date = avg.get('start_date')
        args.end_date = avg.get('end_date')

        self.args = args

        # --- Grid / basin masks (best-effort) ------------------------------
        self.grd = None
        self.depth = None
        self.area = None
        self.basin_code = None
        self.basin_code_xr = None
        if load_grid and args.OUTDIR is not None:
            try:
                self._load_grid(verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f'[{self.name}] grid not loaded (read-only mode): {e}')

    # ------------------------------------------------------------------ #
    # Grid loading
    # ------------------------------------------------------------------ #
    def _load_grid(self, verbose=True):
        OUTDIR = self.args.OUTDIR
        geom_file = OUTDIR + '/' + self.args.geom
        if os.path.exists(geom_file):
            grd = MOM6grid(OUTDIR + '/' + self.args.static, geom_file, xrformat=True)
        else:
            grd = MOM6grid(OUTDIR + '/' + self.args.static, xrformat=True)

        try:
            depth = grd.depth_ocean.values
        except Exception:
            depth = grd.deptho.values

        try:
            area = grd.area_t.where(grd.wet > 0)
        except Exception:
            area = grd.areacello.where(grd.wet > 0)

        # basin masks - remove NaNs, otherwise genBasinMasks won't work
        depth[np.isnan(depth)] = 0.0
        basin_code = genBasinMasks(grd.geolon.values, grd.geolat.values, depth, verbose=False)
        basin_code_xr = genBasinMasks(grd.geolon.values, grd.geolat.values, depth,
                                      verbose=False, xda=True)

        self.grd = grd
        self.depth = depth
        self.area = area
        self.basin_code = basin_code
        self.basin_code_xr = basin_code_xr

    # ------------------------------------------------------------------ #
    # Postprocessed-file path helper
    # ------------------------------------------------------------------ #
    def _path(self, suffix):
        """Return ``<folder>/<name>_<suffix>``."""
        return f'{self.folder}/{self.name}_{suffix}'

    # ------------------------------------------------------------------ #
    # Readers for postprocessed data (read-only)
    # ------------------------------------------------------------------ #
    def global_means(self):
        """Monthly global-mean scalar timeseries (contains ``thetaoga`` etc.)."""
        return xr.open_dataset(self._path('mon_ave_global_means.nc'))

    def thetao(self):
        return xr.open_dataset(self._path('thetao_time_mean.nc'))

    def so(self):
        return xr.open_dataset(self._path('so_time_mean.nc'))

    def sigma2(self):
        thetao = self.thetao().thetao
        so = self.so().so
        return gsw.sigma2(so, thetao)

    def hfds_time_mean(self):
        return xr.open_dataset(self._path('hfds_time_mean.nc'))['hfds']

    def diftrelo_surface(self):
        """(yh, xh) time-mean surface epineutral diffusivity [m2 s-1]."""
        return xr.open_dataset(self._path('diftrelo_surface.nc'))['diftrelo_surface']

    def meke_gm_src(self):
        """(yh, xh) time-mean APE dissipation by the buoyancy (GM/MEKE)
        parameterization, ``MEKE_GM_src`` [W m-2]."""
        return xr.open_dataset(self._path('meke_gm_src.nc'))['MEKE_GM_src']

    def gm_overturning_1km(self):
        """Dataset with the parameterized (GM) overturning streamfunction at
        1000 m depth as a global (yq, xh) map [Sv], plus ``geolonv``/``geolatv``
        2-D coordinates for plotting."""
        return xr.open_dataset(self._path('gm_overturning_1km.nc'))

    def ke_zb2020_vertsum(self):
        """(yh, xh) time-mean, vertically-integrated kinetic-energy source of the
        ZB2020 momentum parameterization, ``rho0 * KE_ZB2020.sum('z_l')``
        [W m-2]."""
        return xr.open_dataset(self._path('ke_zb2020_vertsum.nc'))['KE_ZB2020_vertsum']

    def atm_mean(self):
        return xr.open_dataset(self._path('atm_time_mean.nc'))

    def heat_transport(self):
        ds = xr.open_dataset(self._path('heat_transport.nc')).fillna(0.)
        return ds['T_ady_2d'] + ds['T_diffy_2d'] + ds['T_hbd_diffy_2d']

    def heat_transport_components(self):
        return xr.open_dataset(self._path('heat_transport.nc'))

    def moc(self):
        return xr.open_dataset(self._path('MOC.nc'))

    def moc_sigma2(self):
        return xr.open_dataset(self._path('MOC_sigma2.nc'))

    def section_transports(self):
        return xr.open_dataset(self._path('section_transports.nc'))

    def ice_volume(self):
        return xr.open_dataset(self._path('ice_volume.nc'))

    def mld_winter(self):
        return xr.open_dataset(self._path('MLD_winter.nc'))

    def mld_summer(self):
        return xr.open_dataset(self._path('MLD_summer.nc'))

    # ------------------------------------------------------------------ #
    # Compute helpers (write postprocessed files into ``self.folder``)
    # ------------------------------------------------------------------ #
    # NOTE: these read raw model history and therefore require a valid CIME
    # case (args.OUTDIR / ICEDIR / ATMDIR). They save results as
    # ``<folder>/<name>_<metric>.nc`` so that the readers above find them.
    def compute_and_average_heat_fluxes(self):
        print('Reading dataset...')
        ds = xr.open_mfdataset(
            self.args.OUTDIR + '/' + self.args.native,
            parallel=True, data_vars='minimal', chunks={'time': 12},
            coords='minimal', compat='override',
        )[['T_ady_2d', 'T_diffy_2d', 'T_hbd_diffy_2d']]
        print('Dataset is initialized.')

        print('\n Selecting and loading data between {} and {}...'.format(
            self.args.start_date, self.args.end_date))
        ds_sel = ds.sel(time=slice(self.args.start_date, self.args.end_date)).load()

        attrs = {
            'description': 'Annual mean of poleward heat transport by components ',
            'start_date': self.args.start_date,
            'end_date': self.args.end_date,
            'reduction_method': 'annual mean weighted by days in each month',
            'casename': self.name,
        }

        print('Computing annual mean...')
        ds_ann = weighted_temporal_mean_vars(ds_sel, attrs=attrs)

        print('Computing time-average over all selected data...')
        ds_mean = ds_ann.mean('time').load()

        print('Saving heat fluxes to file...')
        attrs = {'description': 'Time-mean poleward heat transport by components ',
                 'units': ds['T_ady_2d'].units,
                 'start_date': self.args.start_date, 'end_date': self.args.end_date,
                 'casename': self.name}
        add_global_attrs(ds_mean, attrs)
        ds_mean.to_netcdf(self._path('heat_transport.nc'))

    def compute_ice(self):
        print('Reading dataset...')
        ds = xr.open_mfdataset(
            self.args.ICEDIR + '/' + self.args.ice,
            parallel=True, data_vars='minimal',
            coords='minimal', compat='override',
            preprocess=lambda ds: ds[['hi', 'aice']])

        ice_volume = ds.hi.rename({'nj': 'yh', 'ni': 'xh'}) * self.area
        ice_thickness = ds.hi.rename({'nj': 'yh', 'ni': 'xh'})
        ice_extent = ds.aice.rename({'nj': 'yh', 'ni': 'xh'})
        yh = ice_volume.yh

        dataset = xr.Dataset()
        with ProgressBar():
            dataset['ice_volume_NH'] = (ice_volume).where(yh > 0).sum(['xh', 'yh']).compute()
            dataset['ice_volume_SH'] = (ice_volume).where(yh < 0).sum(['xh', 'yh']).compute()
            dataset['ice_thickness'] = ice_thickness.sel(time=slice('0031', '0054')).mean('time').compute()
            dataset['ice_extent'] = ice_extent.sel(time=slice('0031', '0054')).mean('time').compute()

        dataset.to_netcdf(self._path('ice_volume.nc'))

    def compute_atm(self):
        casename = self.args.casename
        files = [f'{self.args.ATMDIR}/{casename}.cam.h0a.00{year}-{month:02d}.nc'
                 for year in range(31, 55) for month in range(1, 13)]
        ds = xr.open_mfdataset(files, combine='by_coords')

        dataset = xr.Dataset()
        dataset['TS_mean'] = ds.TS.mean('time').compute()
        dataset['T_mean'] = ds.T.mean('lon').compute().mean('time')
        dataset['U_mean'] = ds.U.mean('lon').compute().mean('time')

        with ProgressBar():
            dataset.to_netcdf(self._path('atm_time_mean.nc'))

    def compute_hfds_time_mean(self):
        ds = xr.open_mfdataset(
            self.args.OUTDIR + '/' + self.args.native,
            parallel=True, data_vars='minimal',
            coords='minimal', compat='override',
            preprocess=lambda ds: ds[['hfds']])

        ds = ds.sel(time=slice(self.args.start_date, self.args.end_date))

        with ProgressBar():
            ds_mean = ds.mean('time').compute()

        ds_mean.to_netcdf(self._path('hfds_time_mean.nc'))

    def compute_diftrelo_surface(self):
        """Time-mean surface epineutral (Redi) tracer diffusivity.

        Reads raw native history, averages ``diftrelo`` (Ocean Tracer Epineutral
        Laplacian Diffusivity, m2 s-1) over model years 0050-0054 and keeps the
        surface interface (``zi`` = 0 m). Restricting to a few years keeps the
        read fast. The result is a 2-D ``(yh, xh)`` map saved to
        ``<folder>/<name>_diftrelo_surface.nc``.
        """
        # Only open the year-0050..0054 files (opening every year is the slow part).
        native_years = self.args.native.replace('????-??', '005[0-4]-??')
        files = self.args.OUTDIR + '/' + native_years
        print('Files to be read:', glob(files))

        ds = xr.open_mfdataset(
            files,
            parallel=True, data_vars='minimal',
            coords='minimal', compat='override',
            preprocess=lambda ds: ds[['diftrelo']])

        with ProgressBar():
            da = ds['diftrelo'].isel(zi=0).mean('time').compute()

        da.name = 'diftrelo_surface'
        da.attrs.update({
            'long_name': 'Time-mean surface ocean tracer epineutral '
                         'Laplacian diffusivity',
            'units': 'm2 s-1',
            'start_date': '0050-01-01',
            'end_date': '0055-01-01',
            'casename': self.name,
        })
        da.to_dataset().to_netcdf(self._path('diftrelo_surface.nc'))

    def compute_meke_gm_src(self):
        """Time-mean APE dissipation by the buoyancy (GM/MEKE) parameterization.

        Reads the raw native history and averages ``MEKE_GM_src`` (source of
        mesoscale eddy kinetic energy from the GM parameterization, already a
        vertically-integrated 2-D field, W m-2) over the standard averaging
        window (``Avg.start_date`` .. ``Avg.end_date``). The result is a 2-D
        ``(yh, xh)`` map saved to ``<folder>/<name>_meke_gm_src.nc``.
        """
        ds = xr.open_mfdataset(
            self.args.OUTDIR + '/' + self.args.native,
            parallel=True, data_vars='minimal',
            coords='minimal', compat='override',
            preprocess=lambda ds: ds[['MEKE_GM_src']])

        ds = ds.sel(time=slice(self.args.start_date, self.args.end_date))

        with ProgressBar():
            da = ds['MEKE_GM_src'].mean('time').compute()

        da.name = 'MEKE_GM_src'
        da.attrs.update({
            'long_name': 'Time-mean APE dissipation by the buoyancy (GM/MEKE) '
                         'parameterization',
            'units': 'W m-2',
            'start_date': self.args.start_date,
            'end_date': self.args.end_date,
            'casename': self.name,
        })
        da.to_dataset().to_netcdf(self._path('meke_gm_src.nc'))

    def compute_gm_overturning_1km(self):
        """Parameterized (GM) overturning streamfunction at 1000 m as a map.

        Reads the parameterized meridional bolus transport ``vhGM`` (kg s-1)
        from the z-coordinate history (``*.mom6.h.z.*``), averages it over the
        standard window, converts to Sv and vertically accumulates it from the
        bottom to obtain a local overturning streamfunction. The layer nearest
        1000 m is stored as a global ``(yq, xh)`` map together with the
        ``geolonv``/``geolatv`` 2-D coordinates, in
        ``<folder>/<name>_gm_overturning_1km.nc``.
        """
        files = self.args.OUTDIR + '/' + self.args.z
        ds = xr.open_mfdataset(
            files,
            parallel=True, data_vars='minimal',
            coords='minimal', compat='override',
            preprocess=lambda ds: ds[['vhGM']])

        ds = ds.sel(time=slice(self.args.start_date, self.args.end_date))

        with ProgressBar():
            vhGM = ds['vhGM'].mean('time').compute()      # (z_l, yq, xh), kg s-1

        vh = vhGM * 1e-9                                   # -> Sv
        # Local overturning streamfunction: negative cumulative transport from
        # the bottom up to the top interface of each layer (sign convention of
        # spatial_overturning.ipynb: psi[k-1] = psi[k] - vh[k-1], psi_bottom=0),
        # so the Southern Ocean cell is negative and the North Atlantic positive.
        psi = -(vh.fillna(0.0)
                  .isel(z_l=slice(None, None, -1))
                  .cumsum('z_l')
                  .isel(z_l=slice(None, None, -1)))

        psi_1km = psi.sel(z_l=1000.0, method='nearest')
        # Blank cells whose water column does not reach ~1000 m (NaN at that
        # level) so land / shallow shelves render as land, not zero.
        psi_1km = psi_1km.where(vhGM.sel(z_l=1000.0, method='nearest').notnull())

        psi_1km.name = 'psi_gm_1km'
        psi_1km.attrs.update({
            'long_name': 'Parameterized (GM) overturning streamfunction at ~1000 m',
            'units': 'Sv',
            'z_l': float(psi_1km['z_l']),
            'start_date': self.args.start_date,
            'end_date': self.args.end_date,
            'casename': self.name,
        })

        out = psi_1km.to_dataset()
        # Attach v-point geographic coordinates from the ocean geometry file.
        geom = xr.open_dataset(self.args.OUTDIR + '/' + self.args.geom)
        out['geolonv'] = (('yq', 'xh'), geom['geolonv'].values)
        out['geolatv'] = (('yq', 'xh'), geom['geolatv'].values)
        out.to_netcdf(self._path('gm_overturning_1km.nc'))

    def compute_ke_zb2020_vertsum(self):
        """Vertically-integrated kinetic-energy source of the ZB2020 scheme.

        Reads ``KE_ZB2020`` (m3 s-3) from the z-coordinate history
        (``*.mom6.h.z.*``), forms the column-integrated energy-injection rate
        ``rho0 * KE_ZB2020.sum('z_l')`` (rho0 = 1023 kg m-3, giving W m-2) and
        averages it over the standard window. The 2-D ``(yh, xh)`` map is saved
        to ``<folder>/<name>_ke_zb2020_vertsum.nc``.
        """
        rho0 = 1023.0

        files = self.args.OUTDIR + '/' + self.args.z
        ds = xr.open_mfdataset(
            files,
            parallel=True, data_vars='minimal',
            coords='minimal', compat='override',
            preprocess=lambda ds: ds[['KE_ZB2020']])

        ds = ds.sel(time=slice(self.args.start_date, self.args.end_date))

        with ProgressBar():
            da = (rho0 * ds['KE_ZB2020'].sum('z_l', min_count=1)).mean('time').compute()

        da.name = 'KE_ZB2020_vertsum'
        da.attrs.update({
            'long_name': 'Time-mean vertically-integrated kinetic-energy source '
                         'of the ZB2020 momentum parameterization '
                         '(rho0 * KE_ZB2020.sum(z_l), rho0 = 1023 kg m-3)',
            'units': 'W m-2',
            'start_date': self.args.start_date,
            'end_date': self.args.end_date,
            'casename': self.name,
        })
        da.to_dataset().to_netcdf(self._path('ke_zb2020_vertsum.nc'))
