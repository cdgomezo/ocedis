import numpy as np
import xarray as xr

from ocean14c.regions import (
    drake_passage_mask,
    is_cold_population,
    northern_north_pacific_mask,
    southern_ocean_mask,
)


def _field(values, lats, lons):
    return xr.DataArray(
        np.array(values, dtype=float),
        dims=("lon", "lat"),
        coords={"lon": np.array(lons, dtype=float), "lat": np.array(lats, dtype=float)},
    )


def test_southern_ocean_mask_is_contiguous_from_south_pole():
    # single longitude column: cold at the pole, crosses 20 permil at -60,
    # then dips back below 20 further north (a local, unconnected patch)
    # that must NOT be included -- the boundary is the *first* crossing.
    lats = [-89.5, -70, -60, -50, -40, -20]
    values = [[-100.0, -50.0, 25.0, 10.0, 30.0, 30.0]]  # shape (lon=1, lat=6)
    field = _field(values, lats, [0.0])
    mask = southern_ocean_mask(field, threshold=20.0)
    assert mask.sel(lon=0.0, lat=-89.5).item()
    assert mask.sel(lon=0.0, lat=-70).item()
    assert not mask.sel(lon=0.0, lat=-60).item()  # first cell >= threshold
    assert not mask.sel(lon=0.0, lat=-50).item()  # below threshold again, but not connected
    assert not mask.sel(lon=0.0, lat=-40).item()


def test_northern_north_pacific_west_of_149_5w_uses_32n_boundary():
    field = _field([[0.0, 0.0]], [31.0, 33.0], [-160.0])
    mask = northern_north_pacific_mask(field)
    assert not mask.sel(lon=-160.0, lat=31.0).item()
    assert mask.sel(lon=-160.0, lat=33.0).item()


def test_northern_north_pacific_east_of_149_5w_uses_29n_boundary():
    field = _field([[0.0, 0.0]], [28.0, 30.0], [-130.0])
    mask = northern_north_pacific_mask(field)
    assert not mask.sel(lon=-130.0, lat=28.0).item()
    assert mask.sel(lon=-130.0, lat=30.0).item()


def test_northern_north_pacific_excludes_north_atlantic():
    # same lat/lon threshold conditions satisfied, but this is the Atlantic
    field = _field([[0.0]], [45.0], [0.0])
    mask = northern_north_pacific_mask(field)
    assert not mask.sel(lon=0.0, lat=45.0).item()


def test_drake_passage_mask_covers_the_named_box():
    field = _field([[0.0, 0.0]], [-60.0, 0.0], [-65.0])
    mask = drake_passage_mask(field)
    assert mask.sel(lon=-65.0, lat=-60.0).item()
    assert not mask.sel(lon=-65.0, lat=0.0).item()


def test_is_cold_population_combines_southern_ocean_and_n_pacific():
    lats = [-89.5, -70, -60, 33.0]
    values = [[-100.0, -50.0, 25.0, 0.0]]
    field = _field(values, lats, [-160.0])
    cold = is_cold_population(field, threshold=20.0)
    assert cold.sel(lon=-160.0, lat=-89.5).item()  # Southern Ocean
    assert cold.sel(lon=-160.0, lat=33.0).item()  # northern N. Pacific
    assert not cold.sel(lon=-160.0, lat=-60.0).item()  # neither
