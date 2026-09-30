"""Regional, high-resolution, sub-monthly reconstruction of the ocean
disequilibrium term.

ocean14c.pipeline.reconstruct_year is GLOBAL, 1x1 and MONTHLY, because that is
the resolution of the science: GLODAP is 1x1, Takahashi is 4x5, and the
published validation (Table 1 / Table S2) is an annual global number. Nothing
here changes it -- every function in this module is new, and the validated path
is not imported into, only from.

What this adds is a target GRID and a target TIME AXIS:

    reconstruct_regional(year, grid=Grid(0.1, 33, 73, -15, 35), hourly=True)

and the honest statement of what that does and does not buy:

  * HOURLY IS REAL, and it comes from the wind. k = gamma * U^2 * (Sc/660)^-0.5
    is LINEAR in U^2, so driving it with hourly ERA5 U^2 instead of the monthly
    mean <U^2> gives an hourly flux whose monthly mean is the monthly product
    again -- EXACTLY, if SST is held monthly. Measured: 0.0 relative difference
    on the domain mean, not merely small. That identity is the reason to trust
    the disaggregation, and scripts/make_regional_hourly.py measures it rather
    than asserting it.

    Switching SST from the Takahashi climatology to hourly ERA5 moves the flux
    by ~9% (January, EUROCOM domain mean), and NOT because of the mild
    non-linearity of Sc and the solubility in temperature: the two SST fields
    themselves differ by 3.9 K over this domain (6.76 vs 10.66 degC), because a
    4x5 climatology sampled nearest-neighbour is the wrong field over marginal
    seas. The identity above is about the wind; the SST choice is a separate,
    larger decision.

  * 0.1 DEGREE IS INTERPOLATION, NOT INFORMATION. Delta_oce comes from a 1x1
    field, pCO2 from a 4x5 climatology, Delta_atm from a 1x1 kriged field. On a
    0.1 grid they are bilinear interpolations of those. The only input with
    genuine sub-degree structure is the wind, at 0.25. A 0.1 file is what the
    transport model wants to read; it is not a 0.1-degree measurement of the
    ocean.

The land/sea distinction also changes meaning when the grid gets fine. GLODAP
has no data in coastal cells, so a bilinear interpolation of it would spread
NaN inland over several 0.1 cells. `fill_coastal` extends the ocean fields into
those cells by nearest-neighbour before interpolating, so a 0.1 coastal cell
that is genuinely ocean gets its neighbour's value instead of a hole, and the
final mask is taken from the target-grid land-sea mask, not from GLODAP's.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from ocean14c.foa import compute_foa
from ocean14c.gas_exchange import DEFAULT_GAMMA
from ocean14c.glodap import surface_delta_1995
from ocean14c.isoflux import to_umol_flux
from ocean14c.pipeline import (
    MONTHS,
    _load_atm_station_series,
    _monthly_delta_atm,
    _monthly_delta_oce,
    _monthly_hemispheric_series,
    build_delta_atm,
)
from ocean14c.atmosphere import (
    DEFAULT_CURVE_FIT,
    NORTHERN_HEMISPHERE_WINDOWS,
    SOUTHERN_HEMISPHERE_WINDOWS,
)
from ocean14c.surface_ocean import project_delta_oce
from ocean14c.takahashi import (
    load_pco2_climatology,
    load_sst_climatology,
    project_pco2_ocean,
    regrid_nearest,
)


@dataclass(frozen=True)
class Grid:
    """A regular lat/lon target grid, labelled by CELL CENTRES.

    resolution : cell size in degrees (same in both directions)
    lat0, lat1, lon0, lon1 : domain EDGES, so Grid(0.1, 33, 73, -15, 35) has its
        first cell centre at 33.05 and its last at 72.95 -- 400 x 500 cells,
        which is what the EUROCOM 0.1 files (CTE-HR, LPJ-GUESS) use.
    """

    resolution: float
    lat0: float
    lat1: float
    lon0: float
    lon1: float

    @property
    def lat(self):
        n = int(round((self.lat1 - self.lat0) / self.resolution))
        return xr.DataArray(
            self.lat0 + (np.arange(n) + 0.5) * self.resolution, dims="lat", name="lat"
        )

    @property
    def lon(self):
        n = int(round((self.lon1 - self.lon0) / self.resolution))
        return xr.DataArray(
            self.lon0 + (np.arange(n) + 0.5) * self.resolution, dims="lon", name="lon"
        )

    def describe(self):
        return (f"{self.resolution:g} deg, lat {self.lat0:g}..{self.lat1:g}, "
                f"lon {self.lon0:g}..{self.lon1:g} "
                f"({self.lat.size} x {self.lon.size} cells)")


def fill_coastal(da, passes: int = 3):
    """Extend a field into its NaN neighbours, `passes` times.

    GLODAP and Takahashi both stop short of the coast, and on a 1x1 grid that
    is a cell or two. Interpolated straight onto 0.1 it becomes a ragged NaN
    fringe tens of cells wide in places, which the transport model would read
    as "no air-sea exchange in the German Bight". Each pass replaces a NaN by
    the mean of its non-NaN 4-neighbours, i.e. the field grows one cell into
    the land; the result is only used where the target grid says ocean.
    """
    out = da
    for _ in range(passes):
        if not bool(np.isnan(out).any()):
            break
        stack = xr.concat(
            [out.shift(lat=1), out.shift(lat=-1), out.shift(lon=1), out.shift(lon=-1)],
            dim="neighbour",
        )
        out = out.where(~np.isnan(out), stack.mean(dim="neighbour", skipna=True))
    return out


def to_grid(da, grid: Grid, coastal_passes: int = 3, method: str = "linear"):
    """Interpolate a (…, lat, lon) field onto `grid`, filling the coast first."""
    da = fill_coastal(da, passes=coastal_passes)
    return da.interp(lat=grid.lat, lon=grid.lon, method=method)


def hourly_axis(year: int, tz_stamp: str = "mid"):
    """Hourly interval starts and, if asked, mid-points, for one calendar year.

    The flux files this feeds (CTE-HR, LPJ-GUESS on the EUROCOM grid) stamp the
    MID of the interval -- 00:30 for the hour beginning at 00:00 -- so that is
    the default. ERA5's instantaneous values are stamped at the start.
    """
    starts = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01",
                           freq="h", inclusive="left")
    if tz_stamp == "start":
        return starts
    return starts + pd.Timedelta(minutes=30)


def load_era5_hourly(paths, grid: Grid, months=None):
    """Hourly u2 [m2 s-2] and SST [degC] on the target grid.

    `paths` are the monthly files written by data/ERA5_hourly_regional.py.
    ERA5 comes on a 0.25 grid with latitude DESCENDING; both are fixed here.
    """
    ds = xr.open_mfdataset(sorted(paths), combine="by_coords")
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    if ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")
    if months is not None:
        ds = ds.sel(time=np.isin(ds["time"].dt.month, months))
    u2 = ds["u2"].interp(lat=grid.lat, lon=grid.lon, method="linear")
    # SST is NaN over land; fill before interpolating so coastal ocean cells on
    # the fine grid do not inherit a hole from a 0.25 land cell.
    sst = fill_coastal(ds["sst"], passes=3).interp(
        lat=grid.lat, lon=grid.lon, method="linear"
    )
    return u2, sst


def monthly_fields(year: int, grid: Grid, data_dir="data", krige_dir="output",
                   regression_version="original", curve_fit=DEFAULT_CURVE_FIT,
                   atm_variant=None, coastal_passes: int = 3):
    """The slow-varying monthly inputs, interpolated onto `grid`.

    Returns delta_oce, delta_atm [permil], pco2_ocean [uatm] and the
    climatological SST [degC], all with dims (month, lat, lon).
    """
    from pathlib import Path

    data_dir = Path(data_dir)
    delta_1995 = surface_delta_1995(data_dir / "raw" / "GLODAP")
    lat, lon = delta_1995["lat"], delta_1995["lon"]

    delta_oce = _monthly_delta_oce(delta_1995, year, regression_version)

    nh_series, sh_series = _load_atm_station_series(data_dir / "raw" / "Delta_atm")
    nh_monthly, nh_station, nh_gap = _monthly_hemispheric_series(
        nh_series, NORTHERN_HEMISPHERE_WINDOWS, year, curve_fit=curve_fit)
    sh_monthly, sh_station, sh_gap = _monthly_hemispheric_series(
        sh_series, SOUTHERN_HEMISPHERE_WINDOWS, year, curve_fit=curve_fit)
    delta_atm = build_delta_atm(lat, lon, year, nh_monthly, sh_monthly,
                                variant=atm_variant, krige_dir=krige_dir)
    if "lon" not in delta_atm.dims:
        # The zonal geometries return (month, lat); broadcast so the regional
        # product always carries a full 2-D field, whatever the variant.
        delta_atm = delta_atm.broadcast_like(delta_oce.isel(month=0)).transpose(
            "month", ...) if "month" in delta_atm.dims else delta_atm

    taka = data_dir / "raw" / "Takahashi2009" / "month_flux_2006c.txt"
    pco2_1x1 = regrid_nearest(load_pco2_climatology(taka), lat, lon)
    sst_1x1 = regrid_nearest(load_sst_climatology(taka), lat, lon)
    pco2_ocean = project_pco2_ocean(pco2_1x1, year=year)

    fields = {}
    for name, da in (("delta_oce", delta_oce), ("delta_atm", delta_atm),
                     ("pco2_ocean", pco2_ocean), ("sst_clim", sst_1x1)):
        da = da.transpose("month", "lat", "lon")
        fields[name] = to_grid(da, grid, coastal_passes=coastal_passes)

    fields["provenance"] = {
        "atm_station_nh": nh_station,
        "atm_station_sh": sh_station,
        "atm_extrapolation_yr": max(nh_gap, sh_gap),
    }
    return fields


def reconstruct_regional(year: int, grid: Grid, era5_paths=None, data_dir="data",
                         krige_dir="output", gamma: float = DEFAULT_GAMMA,
                         sc_poly: str = "1992", regression_version="original",
                         curve_fit=DEFAULT_CURVE_FIT, atm_variant=None,
                         sst_source: str = "era5", months=None,
                         coastal_passes: int = 3):
    """Hourly C_D14C flux and its ingredients on a regional grid.

    era5_paths : the monthly hourly-ERA5 files (data/ERA5_hourly_regional.py).
                 If None, the result is MONTHLY -- the native resolution of the
                 reconstruction, with no wind disaggregation.
    sst_source : "era5" uses the hourly SST for the Schmidt number and the
                 solubility; "climatology" holds the Takahashi monthly SST,
                 which makes the monthly mean of the hourly flux identical to
                 the monthly product (k is linear in U^2, and nothing else then
                 varies within the month).

    Returns a dict of DataArrays: c14flux [umol m-2 s-1], foa [mol m-2 yr-1],
    delta_oce / delta_atm / delta_odis [permil], plus provenance.
    """
    fields = monthly_fields(year, grid, data_dir=data_dir, krige_dir=krige_dir,
                            regression_version=regression_version,
                            curve_fit=curve_fit, atm_variant=atm_variant,
                            coastal_passes=coastal_passes)
    delta_odis = fields["delta_oce"] - fields["delta_atm"]

    if era5_paths is None:
        foa = compute_foa(_monthly_u2(year, grid, data_dir), fields["sst_clim"],
                          fields["pco2_ocean"], gamma=gamma, sc_poly=sc_poly)
        out = {"c14flux": to_umol_flux(foa, delta_odis), "foa": foa}
        out.update({k: fields[k] for k in ("delta_oce", "delta_atm", "pco2_ocean")})
        out["delta_odis"] = delta_odis
        out["provenance"] = fields["provenance"]
        out["freq"] = "monthly"
        return out

    u2_h, sst_h = load_era5_hourly(era5_paths, grid, months=months)

    # The monthly fields are held constant within their month: Delta_oce moves
    # on a decadal trend, Delta_atm on a seasonal cycle, pCO2 is a climatology.
    # Only the wind (and optionally SST) carries sub-monthly structure.
    month_of = u2_h["time"].dt.month
    delta_odis_h = delta_odis.sel(month=month_of)
    pco2_h = fields["pco2_ocean"].sel(month=month_of)
    sst = sst_h if sst_source == "era5" else fields["sst_clim"].sel(month=month_of)

    foa_h = compute_foa(u2_h, sst, pco2_h, gamma=gamma, sc_poly=sc_poly)
    c14flux_h = to_umol_flux(foa_h, delta_odis_h)

    return {
        "c14flux": c14flux_h,
        "foa": foa_h,
        "delta_oce": fields["delta_oce"],
        "delta_atm": fields["delta_atm"],
        "delta_odis": delta_odis,
        "pco2_ocean": fields["pco2_ocean"],
        "sst_hourly": sst_h if sst_source == "era5" else None,
        "sst_clim": fields["sst_clim"],
        "pco2_hourly_month": month_of,
        "provenance": fields["provenance"],
        "freq": "hourly",
        "sst_source": sst_source,
    }


def _monthly_u2(year, grid, data_dir):
    """The global monthly <U^2> of the published path, on the regional grid."""
    from pathlib import Path

    from ocean14c.winds import load_u2

    u2 = load_u2(Path(data_dir) / "processed" / "era5_u2" / f"u2_{year}.nc")
    if u2["lon"].values.max() > 180:
        u2 = u2.assign_coords(lon=((u2["lon"] + 180) % 360) - 180).sortby("lon")
    return u2.interp(lat=grid.lat, lon=grid.lon, method="linear")
