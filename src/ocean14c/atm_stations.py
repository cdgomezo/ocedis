"""Parsers for the raw atmospheric Delta14C station records under
data/raw/Delta_atm/ (CLAUDE.md Sec 3.4).

Two formats are present:
  - a shared NOAA/ICOS ".c14" semicolon format for NWR, JFJ, SPO, PSA, SMO.
  - a NOAA ObsPack-style space-delimited format, used here only for DRP,
    whose own .c14 file doesn't cover the 2006-2008 window Miller 2025 uses.
"""

import csv
from pathlib import Path


def load_c14_station(path) -> list[tuple[float, float]]:
    """(decimal_date, value) pairs from a NOAA/ICOS .c14 file, Flag=='O' only.

    Column layout (from the file's own header row):
    Site;SamplingHeight;Year;Month;Day;Hour;Minute;DecimalDate;IntegrationTime;
    SamplingPattern;14C;WeightedStdErr;NbPoints;Flag;...
    """
    lines = Path(path).read_text().splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("#Site;"))
    columns = lines[header_idx].lstrip("#").split(";")
    date_col = columns.index("DecimalDate")
    value_col = columns.index("14C")
    flag_col = columns.index("Flag")

    series = []
    for row in csv.reader(lines[header_idx + 1 :], delimiter=";"):
        if len(row) <= max(date_col, value_col, flag_col):
            continue
        if row[flag_col] != "O":
            continue
        # A handful of rows carry flag 'O' but a blank 14C value (2 in ALT,
        # 1 in CGO, 2 in IZO, 1 in NMY as of the L2-2025_1 release). Skip
        # them rather than raising -- none of the files the Ocedis pipeline
        # itself reads (NWR, JFJ, SPO, PSA, SMO, DRP) contain any, so this
        # cannot change the validated 2000-2012 reconstruction.
        try:
            series.append((float(row[date_col]), float(row[value_col])))
        except ValueError:
            continue
    return series


def load_drp_curl_event(path) -> list[tuple[float, float]]:
    """(decimal_date, value) pairs from the DRP ObsPack-style curl_event file.

    Per the file's own qcflag documentation: column 1 of the 3-character
    qcflag is the REJECTION flag -- any non-'.' there means "should not be
    interpreted." Columns 2/3 (selection, information) don't invalidate the
    measurement, so only column 1 is checked.
    """
    lines = Path(path).read_text().splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("site_code "))
    columns = lines[header_idx].split()
    date_col = columns.index("time_decimal")
    value_col = columns.index("value")
    flag_col = columns.index("qcflag")

    series = []
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            continue
        row = line.split()
        if row[flag_col][0] != ".":
            continue
        series.append((float(row[date_col]), float(row[value_col])))
    return series
