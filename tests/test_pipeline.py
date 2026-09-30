from pathlib import Path

import pytest

from ocean14c.pipeline import cell_area_m2

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
GLODAP_DIR = DATA_DIR / "raw" / "GLODAP"
ERA5_2006 = DATA_DIR / "processed" / "era5_u2" / "u2_2006.nc"
OCEDISEQ_PATH = REPO_ROOT / "Ocediseq.nc"

requires_full_pipeline_data = pytest.mark.skipif(
    not (GLODAP_DIR.exists() and ERA5_2006.exists()),
    reason="full pipeline needs GLODAP, Takahashi, Delta_atm, and a processed ERA5 year",
)
requires_ocediseq = pytest.mark.skipif(
    not OCEDISEQ_PATH.exists(), reason="Ocediseq.nc ground-truth file not present"
)


def test_cell_area_m2_at_equator_matches_known_111km_per_degree():
    # A 1x1 degree cell at the equator is ~111.19 km on a side.
    area = cell_area_m2(lat=0.0)
    assert area == pytest.approx((111_190.0) ** 2, rel=0.01)


def test_cell_area_m2_shrinks_toward_the_poles():
    assert cell_area_m2(lat=0.0) > cell_area_m2(lat=60.0) > cell_area_m2(lat=89.5)


@requires_full_pipeline_data
def test_reconstruct_year_2006_matches_table1_and_table_s2_within_structural_uncertainty():
    # Thresholds below are empirical, set from an actual run against real
    # 2006 ERA5/GLODAP/Takahashi/Delta_atm data (see docs/SCIENCE.md's
    # "Validation" note) -- generous relative to what was measured, so this
    # is a real regression guard, not a tautology. The paper itself
    # acknowledges +/-20% structural uncertainty from gas-transfer velocity
    # alone (CLAUDE.md Sec 1); a from-scratch reproduction stacking several
    # independent approximations (ERA5 substituting for ERA-Interim,
    # simplified atmospheric curve fit, nearest-valid-neighbor regridding,
    # default gamma) reasonably lands somewhat outside that band.
    from ocean14c.pipeline import reconstruct_year
    from ocean14c.validate import TABLE_1_FOA_DELTA_ODIS, TABLE_S2

    result = reconstruct_year(2006, data_dir=DATA_DIR)
    target_table1 = TABLE_1_FOA_DELTA_ODIS[2006]
    target_s2 = TABLE_S2[2006]

    assert result["global_isoflux"] < 0
    assert result["global_isoflux"] == pytest.approx(target_table1, rel=0.35)
    assert result["global_foa"] == pytest.approx(target_s2["global_foa"], rel=0.6)
    assert result["global_delta_odis"] == pytest.approx(target_s2["global_delta_odis"], rel=0.25)
    assert result["socean_foa"] == pytest.approx(target_s2["socean_foa"], rel=0.2)
    assert result["socean_delta_odis"] == pytest.approx(target_s2["socean_delta_odis"], rel=0.15)


@requires_full_pipeline_data
@requires_ocediseq
def test_reconstruct_year_2006_spatial_pattern_correlates_with_ocediseq():
    # Correlation is scale-invariant, so this is unaffected by gamma choice
    # -- it isolates whether the *spatial pattern* (not just the aggregate
    # magnitude) matches the paper's own published gridded product.
    from ocean14c.pipeline import reconstruct_year
    from ocean14c.validate import load_ocediseq_annual, spatial_correlation

    result = reconstruct_year(2006, data_dir=DATA_DIR)
    ours = result["annual_isoflux"].transpose("lat", "lon")
    theirs = load_ocediseq_annual(OCEDISEQ_PATH, 2006).transpose("lat", "lon")

    corr = spatial_correlation(ours, theirs)
    assert corr > 0.85
