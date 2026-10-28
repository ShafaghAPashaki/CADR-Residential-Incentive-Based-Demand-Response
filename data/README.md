# Acquiring the input data

The repository does **not** redistribute household electricity records or household-level derived time series. Obtain authorized access to Pecan Street Dataport and obtain the 2018 electricity price series from the source described in the manuscript. Follow each provider’s licence.

The frozen configurations expect two local, untracked files:

- `data/load_hourly_2018.csv`, columns `dataid,time,air,car,clotheswasher,dishwasher,dry,non-shiftable,total`; dataids `661`, `3039`, `8565`. `time` is parsed by `utils/load_demand.py` (format below). Consumption values are in hourly kWh (numerically equal to mean kW over each one-hour bin). `non-shiftable` is total usage minus the listed flexible appliances, subject to consistent aggregation and treatment of missing values. **The exact private Dataport export and preprocessing provenance must be matched for exact numerical replication.**
- `data/ercot_hourly_price.csv`, columns `timestamp,Price`; verify the units and source against `utils/load_price.py` and the paper.

Place these files in `data/` locally. Never add them to a public commit. The data hashes of the published study were `c183ce46…` (`load_hourly_2018.csv`) and `56fdd5b4…` (`ercot_hourly_price.csv`); exact reproduction of the forecasts requires byte-identical inputs. To regenerate the forecast inputs used by the DDQN, follow `forecast_regeneration/REGENERATE.md`.

## Required file names (downloaded name vs. model name)

The downloaded filename can be arbitrary; the final paths below are **mandatory** for the frozen `config_seed_*.yaml` files. Run from the repository root.

| Download or source file | Prepare / rename to | Requirement |
|---|---|---|
| Pecan Street Austin 2018, 15-minute Dataport export (e.g. `15minute_data_austin.csv`) | `data/load_hourly_2018.csv` | **Convert with `data/prepare_load.py`; do not merely rename the raw file.** |
| EnergyOnline ERCOT 2018 hourly price export (Excel/CSV; arbitrary downloaded name) | `data/ercot_hourly_price.csv` | **Select the correct price series first**, produce `timestamp,Price` (USD/MWh), and validate with `data/prepare_price.py`. A raw Excel export with multiple zones cannot merely be renamed. |

The same processed CSVs must be copied to `forecast_regeneration/data/`
when running the preserved historical LSTM source scripts; see
`START_HERE_SINGLE_SEED.md`. Neither licensed data files nor downloaded
source archives should be committed to Git.

**Official acquisition pages:** [Pecan Street Dataport](https://dataport.pecanstreet.org/);
[EnergyOnline ERCOT Real-time Price](https://www.energyonline.com/Data/GenericData.aspx?DataId=4).
An alternative official historical market-data index is
[ERCOT Market Prices](https://www.ercot.com/mktinfo/prices).
The original EnergyOnline **price zone and extraction settings have not yet
been recovered**. Do not select a zone arbitrarily or claim an unrelated
series matches the study. Match a candidate against the locally retained
historical `ercot_hourly_price.csv` before accepting it as exact.

## Preparing the inputs yourself

The data directory includes tested scripts and **schema-only/synthetic examples**:

- [`examples/raw_15min_schema.csv`](examples/raw_15min_schema.csv) — **header only** from the actual 79-column Dataport export, no observations.
- [`examples/synthetic_raw_15min.csv`](examples/synthetic_raw_15min.csv) — eight **fabricated** rows for validating column alignment, UTC offset and appliance rules.
- [`examples/hourly_load_schema.csv`](examples/hourly_load_schema.csv) and [`examples/hourly_price_schema.csv`](examples/hourly_price_schema.csv) — headers only.
- [`examples/synthetic_hourly_price.csv`](examples/synthetic_hourly_price.csv) — deliberately fabricated prices, not a usable 2018 series.
- [`prepare_load.py`](prepare_load.py) and [`prepare_price.py`](prepare_price.py) — preparation/validation commands.

For a properly licensed `15minute_data_austin.csv` export of 2018, run from the project root:

```powershell
python data/prepare_load.py --input downloads/15minute_data_austin.csv --output data/load_hourly_2018.csv --year 2018
python data/prepare_price.py --input downloads/ercot_hourly_2018.csv --output data/ercot_hourly_price.csv --year 2018
python -m unittest tests.test_data_preparation -v
```

**Original load transformations recovered.** The independent preparation code follows the
MARL-iDR project's [`utils/pre_process.py`](https://github.com/ShafaghAPashaki/MARL-iDR-Multi-Agent-Reinforcement-Learning-for-Incentive-based-Residential-Demand-Response/blob/main/utils/pre_process.py):
offset-aware `local_15min` timestamps converted to UTC, 15-minute max resampling,
zero for missing values or readings below 0.1, `air1+air2+air3+airwindowunit1` capped
at 4 kW, car/washer/dishwasher/dryer discretised to 4/1/2/2 kW when active,
31 selected background channels summed and capped at 5 kW, and finally
**arithmetic averaging of 15-minute power profiles into hourly bins**.
This is *not* a naïve sum or average of the original appliance channels.

This method was tested against the locally held 2018 original processed file:
all **218,835 household-hours across 25 IDs** matched in every consumption column
within `1e-9`; maximum absolute difference `1.7763568394002505e-15`.
The generated CSV may have a different byte-level SHA-256 because of row ordering,
float-to-text formatting and column serialization, despite numerically matching.
The original Pecan Street source is not redistributed.

A fresh Dataport download must still cover the correct channels, household IDs and
time range. A different raw extract may contain different missing observations or
different access rules. The source's `local_15min` must include the original
UTC offsets (`-05`/`-06`); timestamps used by CADR are UTC-naive *after*
conversion, not local clock time. Zero-filling follows the historic method
and should be reported as an explicit preprocessing assumption.

**Price preparation is different.** `prepare_price.py` validates a *previously
selected hourly* `timestamp,Price` series in USD/MWh and checks full 2018
coverage, allowing the extra 1 January 2019 boundary hour. It **does not**
identify or download the original EnergyOnline/ ERCOT price zone, product, or
aggregation. Users must choose a series consistent with the manuscript and
check its provenance. On the historical 8,761-row price file, the validator
preserved the input SHA-256 exactly.

The separate `utils/load_demand.py` and `utils/load_price.py` modules
load these processed CSVs into the CADR simulator and do **not** themselves
perform raw Dataport or market-data acquisition.

## Exact parsing and units

- Residential `time`: strictly `dd/mm/yyyy HH:MM` (the loader uses `%d/%m/%Y %H:%M`); one row per house per hour.
- The study's file contained 25 Dataport households with 8,754 hourly rows each; only `661`, `3039` and `8565` are used (the loaders filter by `dataid`). Its SHA-256 (`c183ce46…`) therefore depends on the full 25-household export; a file containing only the three households will not match that hash (the code filters by `dataid`, but a three-household file has not been tested).
- In the study's file `total` equals the sum of the six other columns for every row of the three households (maximum difference 2e-15).
- `air`, `car`, `clotheswasher`, `dishwasher`, `dry`, `non-shiftable`, `total`: hourly energy in kWh. In hourly bins the numerical value equals mean kW.
- Appliances correspond to the like-named hourly Pecan Street Dataport channels after aggregation into hourly intervals. Determine the source export's original granularity, timezone and missing-value treatment under your own Dataport license.
- `non-shiftable = total - (air + car + clotheswasher + dishwasher + dry)` only when these are the exact channels included in `total` and the source readings are consistent; validate non-negativity, no gaps and per-house/hour totals. No source-data preparation script is supplied that proves the original export-to-CSV transformation, so byte-identical reproduction from arbitrary exports is **not guaranteed**.
- Electricity price `timestamp` is parsed day-first; `Price` is expressed in **USD/MWh**, converted by `utils/load_price.py` to **cent/kWh** via division by 10. Match source, timezone and any missing-hour handling to the study.
