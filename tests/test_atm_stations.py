from pathlib import Path

import pytest

from ocean14c.atm_stations import load_c14_station, load_drp_curl_event

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "Delta_atm"
requires_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="Delta_atm raw data not present")

C14_FIXTURE = """\
# some metadata
# more metadata
#Site;SamplingHeight;Year;Month;Day;Hour;Minute;DecimalDate;IntegrationTime;SamplingPattern;14C;WeightedStdErr;NbPoints;Flag;Extra
NWR;3;2003;5;27;19;0;2003.4019;;spot sampling ;72.29;2.82;;O;x
NWR;3;2003;7;29;16;30;2003.5741;;spot sampling ;62.73;2.6;;N;x
NWR;3;2004;1;1;0;0;2004.0;;spot sampling ;70.0;2.0;;O;x
"""

CURL_EVENT_FIXTURE = """\
# header_lines : 3
# more metadata
site_code year month day hour minute second datetime time_decimal air_sample_container_id value value_unc latitude longitude altitude elevation intake_height method event_number instrument analysis_datetime qcflag
DRP 2005 3 16 20 33 0 2005-03-16T20:33:00Z 2005.205 2979-99 65.33 1.8 -55.25 -64.88 10.0 0.0 10.0 N 198386 ci 2005-10-24T12:00:00 ..C
DRP 2005 3 17 20 33 0 2005-03-17T20:33:00Z 2005.208 2980-99 70.00 1.8 -55.25 -64.88 10.0 0.0 10.0 N 198387 ci 2005-10-24T12:00:00 C..
DRP 2005 3 18 20 33 0 2005-03-18T20:33:00Z 2005.210 2981-99 60.00 1.8 -55.25 -64.88 10.0 0.0 10.0 N 198388 ci 2005-10-24T12:00:00 ...
"""


def test_load_c14_station_keeps_only_flag_o_rows(tmp_path):
    path = tmp_path / "fixture.c14"
    path.write_text(C14_FIXTURE)

    series = load_c14_station(path)

    assert len(series) == 2
    assert series[0] == pytest.approx((2003.4019, 72.29))
    assert series[1] == pytest.approx((2004.0, 70.0))


def test_load_drp_curl_event_rejects_column1_flagged_rows(tmp_path):
    path = tmp_path / "fixture.txt"
    path.write_text(CURL_EVENT_FIXTURE)

    series = load_drp_curl_event(path)

    # "C.." has a rejection flag in column 1 and must be dropped;
    # "..C" and "..." are both usable (columns 2/3 are informational only).
    assert len(series) == 2
    assert series[0] == pytest.approx((2005.205, 65.33))
    assert series[1] == pytest.approx((2005.210, 60.00))


@requires_data
def test_load_c14_station_on_real_nwr_file_returns_plausible_series():
    series = load_c14_station(RAW_DIR / "noaa_l2_2025_1_nwr_3m_spot_min_clean.c14")
    assert len(series) > 500
    decimal_dates = [d for d, _ in series]
    assert min(decimal_dates) > 2003.0
    assert max(decimal_dates) < 2022.0


@requires_data
def test_load_drp_curl_event_on_real_file_covers_2006_to_2008():
    series = load_drp_curl_event(RAW_DIR / "co2c14_drp_shipboard-flask_376_curl_event.txt")
    decimal_dates = [d for d, _ in series]
    assert min(decimal_dates) < 2007.0
    assert max(decimal_dates) > 2008.0
