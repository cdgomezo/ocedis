# Ocean ¹⁴C Isotopic Disequilibrium Flux — Reconstruction Project

## Purpose

Reconstruct, in Python, the ocean isotopic disequilibrium isoflux (**FoaΔodis**) calculation
of Miller, Lehman & Lindsay (2025, *Global Biogeochemical Cycles*, "Numerical representation
of contemporary atmospheric Δ¹⁴CO₂. Part 1"), validated against the paper's own published
numbers (Table 1, Table S2). This is one term (the ocean term) of a larger atmospheric ¹⁴C
mass balance; the other terms (fossil, terrestrial, cosmogenic, nuclear) are out of scope for
v1 but the repo should be structured so they can be added later (see "Extending" at the end).

Read this whole file before writing any code. It contains every formula, constant, and data
source identified from the project's reference PDFs, organized so the important ~4 papers can
be read closely and the rest treated as background/validation only.

---

## 1. What we are computing

For each 1°×1° ocean grid cell and month *t*:

```
Δodis(t) = Δo(t) − Δa(t)                       # ocean-atmosphere ¹⁴C disequilibrium, ‰
Foa(t)   = one-way ocean→atmosphere CO2 flux    # PgC/yr (gross, not net)
isoflux(t) = Foa(t) × Δodis(t)                  # PgC‰/yr
```

Annual global isoflux = spatial + monthly aggregate of `isoflux(t)`. The paper also reports a
**Southern Ocean** subtotal (Table S2), where the Southern Ocean is defined dynamically as
everything south of the **20‰ isoline in the 1995 GLODAP Δ¹⁴C field** (not a fixed latitude).

Target validation numbers (from Miller et al. 2025, Table 1 and SI Table S2) — reproduce these
to a reasonable tolerance (a few percent; the paper itself reports a ±20% structural uncertainty
from the gas-transfer velocity alone):

| Year | Global FoaΔodis (PgC‰/yr) | Global Foa (PgC/yr) | Global Δodis (‰) | S.Ocean Foa | S.Ocean Δodis |
|---|---|---|---|---|---|
| 2000 | −3426.6 | 69.1 | −51.4 | 21.3 | −114.4 |
| 2006 | −2278.8 | 71.6 | −32.9 | 22.3 | −97.3 |
| 2012 | −1303.3 | 74.6 | −18.1 | 23.1 | −83.9 |

Full 2000–2012 series for Foa/Δodis is in Table S2 (both files reproduced below in §6). Sum of
FoaΔodis over 2000–2012 = **−29,683 PgC‰**. Global ocean isoflux magnitude decreases by **>60%**
(the paper's own sensitivity check with newer data revises this to ~59.5%).

---

## 2. Priority reading order for source PDFs

Not all references in the project are equally load-bearing. Read in this order:

**Tier 1 — contains the actual method and must be read closely:**
1. `Global_Biogeochemical_Cycles__2025__Miller...pdf` — main paper, Section 3.2 "Ocean
   Disequilibrium" and Section 2 (mass-balance definitions). Gives the overall recipe and the
   final numbers to validate against.
2. `2025gb008522sup0001supporting_information_sis01.pdf` — the Supporting Information. This has
   the actual regression equations, coefficients, region boundaries, and Tables S1/S2. **This is
   the single most important document** — the main text only summarizes it.
3. `carbonDynamicsOfTheDeglacialAndContemporaryOceanInferred.pdf` — Lindsay (2016) PhD
   dissertation, the original source of the empirical method (SI Text above is essentially a
   condensed version of this dissertation's relevant chapter). Read the chapter containing
   sections "Map of GLODAP surface ocean ∆14C...", "∆14C rates of change vs. latitude", etc.
   (per its table of contents, this is the chapter with figures numbered 5.1–5.6).
4. `Limnology___Ocean_Methods__2014__Wanninkhof...pdf` — gives the exact, current gas-transfer
   velocity formula and a full Schmidt-number polynomial table (reproduced in §4 below).

**Tier 2 — needed for specific sub-components:**
5. `Global_Biogeochemical_Cycles__2007__Sweeney...pdf` — how the gas-transfer coefficient is
   *calibrated* against the global bomb-¹⁴C inventory (this calibration constant is what Miller
   2025 says they "re-scaled...using the scaling factor appropriate for ERA-Interim winds" —
   see the open issue in §4).
6. `Global_Biogeochemical_Cycles__2005__Key...pdf` — GLODAP construction/gridding method (only
   needed if you rebuild the 1995 Δ¹⁴C-DIC field from bottle data rather than using the
   published gridded product or the regression coefficients directly — see shortcut in §3).
7. `Quart_J_Royal_Meteoro_Soc__2011__Dee...pdf` — ERA-Interim reanalysis (the wind product used
   for Foa). Only need the basics: what it is, resolution, variables available.
8. `acp1656652016.pdf` — Basu, Miller & Lehman (2016). A companion paper using the *same* ocean
   disequilibrium framework inside a full atmospheric inversion. Useful as an independent
   cross-check of the formula and to see a slightly different implementation choice (they use
   a +10‰ correction between 20°N–20°S, vs. 10°N–40°S in Miller 2025 — see open issue in §3).

**Tier 3 — context / provenance only, not needed to write code:**
9. `Journal_of_Geophysical_Research__Oceans__2012__Graven...pdf`, `Geophysical_Research_Letters__2017__Zhai...pdf`,
   `climjclid160725_1.pdf` (Toyama et al. 2017) — mechanistic explanation of *why* the two
   regression populations exist (bomb-¹⁴C reemergence via mode waters). Good background, not
   needed for arithmetic.
10. `decadalchangesofradiocarboninthesurfacebayofbengal...pdf` (Dutta 2010),
    `seawaterradiocarbonevolutioninthegulfofalaska2002observations.pdf` (Guilderson 2006),
    `1s2_0S007966111000159Xmain.pdf` (Kumamoto 2011) — raw cruise data that fed the regression.
    **You do not need these** unless you want to rebuild the regression from scratch instead of
    using the published coefficients (see shortcut below).
11. `1s2_0S0967064508004311mainext.pdf` — Takahashi et al. (2009) pCO2 climatology paper, needed
    only if reconstructing Foa's pCO2 climatology yourself rather than approximating it.

---

## 3. The empirical Δ¹⁴C-DIC method (Lindsay 2016 / Miller 2025 SI) — full recipe

### 3.1 Starting field
Use the **GLODAP gridded Δ¹⁴C-DIC climatology** (Key et al. 2004), 1°×1°, treated as
representative of **1995** conditions. Call this `delta_oce_1995(lat, lon)`.

> **Shortcut for v1:** you do not need to re-derive the regression from raw cruise data. The
> published regression lines already give you `d(Δ14C)/dt` as a *function of* the 1995 GLODAP
> value, so all you need is the gridded 1995 field itself, not the underlying post-2000
> observations (Table S1). Only rebuild the regression from scratch if you want to reproduce
> Figure S2/S5 itself or test sensitivity to input data.

### 3.2 Two regression populations
Two ocean-region groups, each with its own linear regression of
`dΔ14CDIC/dt [‰/yr]` vs. `Δ14CDIC,1995 [‰]`:

**Region boundaries:**
- Southern Ocean = south of the **20‰ isoline** in the 1995 GLODAP field (dynamic boundary, not
  a fixed latitude).
- Northern N. Pacific / low-latitude N. Pacific boundary: **32°N** west of 149.5°W, **29°N**
  east of 149.5°W.
- "Cold/deep" population = Southern Ocean + northern North Pacific.
- "Warm" population = everywhere else (low-latitude oceans + northern North Atlantic mode
  waters).
- Drake Passage excluded from both regressions (too much cross-frontal scatter).

**Regression coefficients** (`rate = m × Δ14C_1995 + b`):

| Population | m (yr⁻¹) | b (‰ yr⁻¹) | Source |
|---|---|---|---|
| Cold/deep ocean (original) | −0.0200 ± 0.0017 | −3.0851 ± 0.1136 | Fig. S2 |
| Warm ocean (original) | −0.0549 ± 0.0023 | 2.9832 ± 0.2000 | Fig. S2 |
| Cold/deep ocean (GLODAP v2 update) | −0.0146 | −3.1359 | Fig. S5 |
| Warm ocean (GLODAP v2 update) | −0.0543 | 2.8324 | Fig. S5 |

Use the **original** coefficients to match Table 1/Table S2 exactly. The v2 coefficients are
provided for a documented sensitivity check (they reduce the 2000–2012 isoflux change from
−62% to ~−59.5%, and change the cumulative 13-yr atmospheric trend by ~0.1‰ — use this as a
secondary regression test, not the primary target).

### 3.3 Project forward in time
```
Δoce(lat, lon, t) = Δoce_1995(lat, lon) + rate(lat, lon) × (t − 1995.0)
```
where `rate(lat, lon)` comes from whichever regression population the grid cell belongs to.
Grid-cell boundaries between populations will be sharp (acknowledged in the SI as a known,
accepted artifact).

### 3.4 Atmospheric Δ14C field, Δatm(t)

> **Since v1 there are five more ways to build this field**, all switchable via
> `config/default.yaml: atmosphere.{geometry,level,blend_deg}` and all scored against each
> other in **docs/SCIENCE.md §12**: the shipped 1°×1° kriged station field
> (`geometry: krige` — best measured skill, and the recommendation), a plain hemispheric
> step (`hemispheric` — what Hua et al. 2021 reduces to for 1973 onward, and it ties the
> kriged field exactly), Basu's band, Graven's three CMIP6 sectors, and Hua's published
> curves (`hua`, whose absolute level sits ~6–8‰ above Graven and Miller — hence
> `level: stations`, which keeps Hua's shape and takes the level from our own network).
> The recipe below is `geometry: miller_band`, still the default.

Built from filtered/smoothed station records (Thoning et al. 1989 curve-fitting algorithm),
averaged into hemispheric means, then a **global mean** = average of N and S hemisphere means.
Stations used and their coverage windows:
- Northern Hemisphere: Niwot Ridge, CO, USA (NWR), 2004–2013; Jungfraujoch, Switzerland (JFJ),
  2000–2003 (or 2000–2008 depending on which figure/table — check both figures/table S1
  captions when implementing; Miller 2025 uses NWR from 2004 and JFJ for the earlier years to
  splice a continuous NH record).
- Southern Hemisphere: South Pole (SPO) 1999–2007; Palmer Station, Antarctica (PSA) 2005–2007;
  American Samoa (SMO) 2001–2007; Drake Passage (DRP) 2006–2012.
- **Δ14CG (global mean, Table 1)** = average of hemispheric annual year-edge values.

For the *ocean disequilibrium field specifically*, Δatm is a **zonally uniform, latitude-banded**
field (not a single global scalar):
- Base value: interpolated/smoothed N and S hemisphere Δatm.
- **Correction band:** an additional **+10‰** is added between two latitudes to represent an
  observed Pacific meridional gradient (UC Irvine shipboard transects).
  - Miller et al. (2025) main text: **+10‰ between 10°N and 40°S**.
  - Basu et al. (2016): **+10‰ between 20°N and 20°S**.
  - **Open issue — RESOLVED, but not in the band's favour.** These are two different papers
    by overlapping authors, with different latitude bands for what is nominally the same
    correction. Both are implemented (`config/default.yaml: atmosphere.tropical_band`), and
    Miller's 10°N–40°S remains the DEFAULT because every shipped number was built with it.

    But the band has since been scored against this repo's own 22-station network by
    leave-one-station-out cross-validation, and it **loses**: 3.847‰ against 2.918‰ for a
    plain hemispheric step and 2.918‰ for the kriged field — worse than adding no tropical
    structure at all. Basu's 20°N–20°S is worse still (4.365‰). Fitting the hemispheric
    values from extratropical stations only, which is how Miller's construction actually
    works, does not rescue it (3.867‰), so it is not an artefact of double-counting.

    Two caveats before concluding the band is simply wrong: the +10‰ comes from UC Irvine
    Pacific shipboard transects that are **not** in this network, and our coverage between
    10°N and 40°S is three sites (SMO −14°, ATO −2°, MLO/KUM +19.5°). The network
    positively disagrees with the band, which is evidence, not proof. Dropping it is a
    decision, not a derivation.

    The choice is worth up to 27% of the global ocean isoflux. See **docs/SCIENCE.md §12**
    and `scripts/compare_delta_atm_variants.py`.

### 3.5 Combine
```
Δodis(lat, lon, t) = Δoce(lat, lon, t) − Δatm(lat, t)     # Δatm varies by latitude band + time, not longitude
```

---

## 4. Gas-transfer velocity and Foa (the "one-way ocean→atmosphere flux")

### 4.1 Core formula (Wanninkhof 2014 — current best estimate)
```
k [cm/hr] = 0.251 × <U²> × (Sc / 660)^(-0.5)
```
- `<U²>` = mean of squared wind speed (**second moment**, not the square of the mean — this
  matters, see §4.3), in (m/s)².
- `Sc` = Schmidt number of CO2 at the local sea surface temperature (dimensionless).
- Reported uncertainty: **±20%**.

Simplified CO2-flux shortcut (valid because `(Sc/660)^-0.5 × K0` is nearly temperature-invariant):
```
F [mol m⁻² yr⁻¹] = 7.7e-4 × <U²> [(m/s)²] × ΔpCO2 [μatm]
```
(This shortcut computes the **net** flux from ΔpCO2 directly and is convenient for validation,
but for the isotope disequilibrium calculation you need the **gross one-way** flux Foa, which
uses `pCO2_ocean` alone rather than `ΔpCO2` — see §4.4.)

### 4.2 Schmidt number polynomial (Wanninkhof 2014, Table 1 — seawater, 35‰, −2 to 40°C)
```
Sc(T) = A + B·T + C·T² + D·T³ + E·T⁴          (T in °C)
```
For CO2 in seawater: **A = 2116.8, B = −136.25, C = 4.7353, D = −0.092307, E = 0.0007555**
(Sc at 20°C = 668). Use these coefficients (they supersede the older Wanninkhof 1992 3rd-order
fit, valid only 0–30°C).

### 4.3 Calibration constant (γ / the "0.251" coefficient) — history and the open issue
The leading coefficient in the k formula is *not* a universal constant — it is fit to the
global ocean bomb-¹⁴C inventory for whatever wind product is being used, because different
reanalysis wind products have different statistical moments. This is the crux of Sweeney et al.
(2007) and Wanninkhof (2014):
```
γ = ΔInv(bomb ¹⁴C) / ∬∬ (Sc/660)^-0.5 × <U²> × s × Δp¹⁴CO2_bomb dA dt
k = γ × (Sc/660)^-0.5 × <U²>
```
- Wanninkhof (1992) original: γ = 0.39 (long-term mean winds) / 0.31 (instantaneous, Rayleigh-
  distributed winds), calibrated to a mean wind of 7.4 m/s and the (now outdated, too-high)
  Broecker et al. (1985) bomb-¹⁴C inventory.
- Sweeney et al. (2007): recalibrated with a lower, revised bomb-¹⁴C inventory and NCEP/NCAR
  Reanalysis-1 winds (1954–2000, mean 6.9 m/s) → **γ = 0.27** using `<U²>` (second moment) or
  **γ = 0.31** using `(<U>)²` (square of the mean) — these differ because γ depends on *which*
  moment of the wind product you use, not just the wind product's mean.
- Wanninkhof (2014): **γ = 0.251**, using an improved wind product (CCMP, 6-hr, 0.25°) and
  further-revised bomb-¹⁴C inventory; uncertainty reduced from ~30% (Sweeney) to ~20%.

**Open issue for this project:** Miller et al. (2025) state they used ERA-Interim winds
"re-scaled to the whole ocean bomb ¹⁴C inventory (Sweeney et al., 2007) using the scaling
factor appropriate for ERA-Interim winds" — i.e., **they derived their own γ_ERA specific to
ERA-Interim's wind statistics**, following Sweeney's method, and this specific numeric value is
**not given anywhere in the text you have access to**. Do not assume 0.27 or 0.251 are directly
correct for ERA-Interim without recalibration. Two options:
1. **(Recommended for v1)** Use Wanninkhof (2014)'s γ = 0.251 with whatever wind reanalysis
   product you can obtain (ERA5 as a modern substitute for ERA-Interim, or ERA-Interim itself if
   available) and treat the ~20% uncertainty as expected/acceptable — the paper itself uses this
   uncertainty band in its own sensitivity analysis (§5, "Discussion").
2. **(More rigorous)** Reproduce the Sweeney (2007) calibration procedure yourself (Appendix A
   of that paper, formula reproduced above) against your specific wind product's `<U²>` and a
   documented bomb-¹⁴C inventory value, to derive your own γ. This is more work and only
   worthwhile if you need Foa to close within a few percent rather than ~20%.

### 4.4 Foa itself — one-way flux convention
Foa (ocean→atmosphere gross flux) and Fao (atmosphere→ocean gross flux) are the two one-way
components of the net air-sea CO2 exchange, following the standard isotope mass-balance
convention (Tans, P.P., 1980, "On calculating the transfer of carbon-13 in reservoir models of
the carbon cycle," *Tellus* 32(5)):
```
F_net = Foa − Fao = k × s × (pCO2_ocean − pCO2_atm)
Foa   = k × s × pCO2_ocean          # depends only on the ocean side
Fao   = k × s × pCO2_atm            # depends only on the atmosphere side
```
where `s` = CO2 solubility in seawater (mol m⁻³ atm⁻¹; temperature/salinity dependent — **not**
detailed in the papers read for this project; use the standard Weiss (1974) solubility formula,
which is the field standard cited implicitly throughout this literature). This is why Miller
2025 describes Foa as derived "from a monthly climatology of surface ocean pCO2" alone — it only
needs the ocean-side value, not the full ΔpCO2.

### 4.5 pCO2 ocean climatology and its secular drift
- Base monthly climatology: Takahashi et al. (2009), by ocean basin/latitude band (see their
  Fig. 10 for representative monthly ΔpCO2 by basin — Atlantic/Pacific/Indian/Southern).
- Miller (2025) updates this at a **fixed drift of +1.5 μatm/yr** for ocean pCO2 (vs. observed
  atmospheric pCO2 rising at **+2.0 ± 0.4 μatm/yr** over the analysis period) — i.e., the ocean
  only partially tracks the atmospheric CO2 rise, consistent with Δoce's slow relaxation in §3.
- Wind forcing (ERA-Interim, Dee et al. 2011) updated at native temporal resolution (paper
  states "every 90 min" — subsample/aggregate to monthly `<U²>` per grid cell as needed).

---

## 5. Assemble the isoflux

```
isoflux(lat, lon, t) = Foa(lat, lon, t) × Δodis(lat, lon, t)      # PgC‰/month, grid-cell level
```
1. Compute monthly, per grid cell.
2. Sum over months within a year → annual, per grid cell (Fig. 1b/Fig. 2 in the paper are the
   2006 annual maps — good intermediate visual checkpoint).
3. Sum over all ocean grid cells → **global annual FoaΔodis** (compare to Table 1).
4. Sum only over cells south of the 20‰ isoline (§3.2) → **Southern Ocean annual FoaΔodis**
   (compare to Table S2). Table S2's Foa/Δodis columns are **flux-weighted spatial+temporal
   means**, i.e. `Δodis_mean = Σ(Foa×Δodis) / Σ(Foa)` and `Foa_mean = Σ(Foa)/N_cells` — implement
   both a straight and flux-weighted aggregator and check which matches Table S2's numbers.

---

## 6. Full validation tables (type these into your test fixtures)

**Table 1 (main text) — Global FoaΔodis, PgC‰/yr:**
```
2000: -3426.6   2001: -3271.2   2002: -2916.0   2003: -2769.6   2004: -2631.3
2005: -2380.7   2006: -2278.8   2007: -1994.2   2008: -1916.4   2009: -1661.5
2010: -1696.6   2011: -1437.2   2012: -1303.3
SUM (2000-2012): -29683
```

**Table S2 (Supporting Information) — annual Foa (PgC/yr) and Δodis (‰), global and Southern Ocean:**
```
Year  Global_Foa  Global_Δodis  SOcean_Foa  SOcean_Δodis
2000     69.1       -51.4         21.3        -114.4
2001     69.2       -49.0         22.1        -111.8
2002     69.3       -43.5         20.7        -108.8
2003     69.4       -41.3         21.6        -104.8
2004     70.2       -38.8         21.8        -103.3
2005     70.0       -35.2         21.3        -100.2
2006     71.6       -32.9         22.3         -97.3
2007     72.7       -28.4         21.7         -92.4
2008     72.2       -27.5         22.5         -91.5
2009     71.7       -24.0         21.7         -88.9
2010     75.7       -23.2         23.8         -87.8
2011     75.1       -19.8         23.2         -84.9
2012     74.6       -18.1         23.1         -83.9
change:   +8%                    -64.8%
```

Note Global_Foa × Global_Δodis ≠ exactly the Table 1 FoaΔodis value directly (Table S2 values
are flux-weighted *means*, not the same as summing the gridded product — Table 1 comes from
summing the actual gridded monthly product, Table S2 is a diagnostic mean). Use Table 1 as the
primary target, Table S2 for validating your Foa and Δodis fields independently before
multiplying them.

---

## 7. External data you will need to source yourself (not included in the project PDFs)

The papers cite these datasets by DOI/URL; none are bundled here. You (the user) will need to
download them manually and place them under `data/raw/`, since this sandbox's network egress is
restricted to package registries (PyPI/npm/GitHub) and cannot reach NOAA/AGU/Wiley servers.

| Data | Where | Used for |
|---|---|---|
| GLODAP 1995 gridded Δ¹⁴C-DIC | https://glodap.info (modern portal; original CDIAC link in Key et al. 2004 is dead) | §3.1 starting field |
| **Final published ocean disequilibrium fluxes (ground truth!)** | https://doi.org/10.15138/8dhy-6k90 (Lindsay, Miller & Lehman 2025 dataset) | Best possible validation — this *is* the answer key, gridded monthly, 2000–2012 |
| All gridded input fluxes incl. box-model code | https://doi.org/10.15138/m6h9-wv02 | Cross-check / possibly contains the actual reference implementation |
| NOAA/INSTAAR atmospheric Δ14C station data | https://gml.noaa.gov/data/data.php | §3.4 Δatm construction |
| JFJ station data | CORSO data product, www.corso-project.eu | §3.4 (JFJ record) |
| SIO station data (SMO, PSA, SPO) | https://scrippsco2.ucsd.edu/data/atmospheric_co2/sampling_stations.html | §3.4 |
| Takahashi pCO2 climatology | Usually distributed via NOAA/OCADS alongside Takahashi et al. (2009) | §4.5 |
| ERA-Interim (or ERA5 as modern substitute) winds | Copernicus Climate Data Store | §4.1 wind speeds |
| CarbonTracker CT2013B (context only) | https://gml.noaa.gov/ccgg/carbontracker/CT2013B/ | Not required for v1 |

**Recommendation:** start by downloading the ground-truth gridded dataset
(doi:10.15138/8dhy-6k90) and the box-model code (doi:10.15138/m6h9-wv02) first — the latter may
contain the actual reference Python/Matlab implementation and answer several of the open
questions in §3.4 and §4.3 directly, making the rest of this document a cross-check rather than
a from-scratch derivation.

---

## 8. Suggested repository structure

```
ocean-14c-disequilibrium/
├── CLAUDE.md                     # this file, at repo root
├── README.md                     # short human-facing summary + how to run
├── pyproject.toml
├── data/
│   ├── raw/                      # user-supplied downloads (see §7); gitignored
│   └── processed/
├── src/ocean14c/
│   ├── __init__.py
│   ├── glodap.py                 # load 1995 GLODAP Δ14C-DIC grid; define region masks
│   ├── regression.py             # §3.2 two-population regression; rate(lat,lon) map
│   ├── surface_ocean.py          # §3.3 project Δoce(t) forward from 1995
│   ├── atmosphere.py             # §3.4 build Δatm(lat_band, t) incl. +10‰ correction band
│   ├── gas_exchange.py           # §4.1-4.3 Schmidt number, k(T,U), calibration constant
│   ├── foa.py                    # §4.4-4.5 Foa(t) from pCO2 climatology + drift
│   ├── isoflux.py                # §5 combine, aggregate global & Southern Ocean
│   ├── regions.py                # 20‰ isoline S.Ocean mask; N.Pacific 32°N/29°N split
│   └── validate.py               # §6 comparison against Table 1 / Table S2
├── notebooks/
│   └── 01_reconstruct_ocean_isoflux.ipynb   # end-to-end walkthrough + Fig.1b/Fig.2 style maps
├── tests/
│   ├── test_regression_coefficients.py      # checks §3.2 numbers are wired in correctly
│   ├── test_schmidt_number.py                # checks §4.2 polynomial against Sc(20°C)=668
│   └── test_against_table1.py                # end-to-end check against §6 targets
└── config/
    └── default.yaml               # toggle: Miller vs Basu tropical correction band (§3.4),
                                    # original vs GLODAP-v2 regression coeffs (§3.2),
                                    # gas-transfer γ constant choice (§4.3)
```

---

## 9. Build order (first Claude Code session)

1. Scaffold the repo structure above; `pyproject.toml` with numpy, scipy, xarray, pandas,
   matplotlib, pytest.
2. Implement `gas_exchange.py` first — it's fully self-contained and has no external data
   dependency (Schmidt number polynomial is just constants). Write `test_schmidt_number.py`
   immediately (Sc(20°C) should equal 668).
3. Implement `regression.py` with the coefficients in §3.2 as hardcoded constants (with a config
   flag for original vs. v2). This also has no external data dependency yet — write
   `test_regression_coefficients.py` to lock in the numbers before touching real grids.
4. Stub `glodap.py`, `atmosphere.py`, and `foa.py` with clear `NotImplementedError`/TODOs
   pointing at §7's data sources, so the pipeline is runnable end-to-end on synthetic/dummy
   grids before real data is wired in.
5. Implement `isoflux.py` + `validate.py` against the §6 numbers using synthetic inputs first
   (to validate the aggregation logic independent of data-fetching correctness).
6. Only after (2)–(5) are solid: wire in real GLODAP/Takahashi/ERA data as the user supplies it
   under `data/raw/`, and re-run `test_against_table1.py` for real.
7. Flag the two open issues explicitly in code comments and in `config/default.yaml`: the
   tropical correction band (10°N–40°S vs 20°N–20°S) and the gas-transfer γ calibration for
   ERA-Interim (§4.3) — these are genuine ambiguities in the literature, not implementation bugs.

---

## 10. Extending beyond v1 (not required now, for context only)

The ocean term is one of five in the full atmospheric ¹⁴C mass balance (Miller 2025, Eq. 2):
fossil (Ff), terrestrial disequilibrium (FbaΔbdis), cosmogenic production, nuclear production,
and radioactive decay. If this project grows to reproduce the full mass balance and the two-box
(troposphere/stratosphere) atmospheric transport model:
- Box model: `d(¹⁴I_trop)/dt = F_trop + k(¹⁴I_strat/M_strat − ¹⁴I_trop/M_trop)`, stratosphere
  residence time 3 years, stratosphere:troposphere mass ratio 1:4, cosmogenic production split
  1:1 between compartments.
- Preferred flux balance in the paper scales cosmogenic production ×0.8 and terrestrial
  disequilibrium ×0.7 relative to their process-model defaults, holding ocean and fossil
  isofluxes as empirically fixed. This is a separate optimization step layered on top of what
  this document covers.
- Basu et al. (2016) (`acp1656652016.pdf`) is the most directly relevant reference for a full
  inversion implementation, including their TRANSCOM-region error-covariance scheme (11 ocean
  regions, 3-month temporal correlation for the disequilibrium terms) — read Section 2.1.2 and
  Table 1 of that paper if you get to this stage.
