# Acquiring the input data

The repository does **not** redistribute household electricity records or household-level derived time series. Obtain authorized access to Pecan Street Dataport and obtain the 2018 electricity price series from the source described in the manuscript. Follow each provider’s licence.

The frozen configurations expect two local, untracked files:

- `data/load_hourly_2018.csv`, columns `dataid,time,air,car,clotheswasher,dishwasher,dry,non-shiftable,total`; dataids `661`, `3039`, `8565`. `time` is parsed by `utils/load_demand.py` (format below). Consumption values are in hourly kWh (numerically equal to mean kW over each one-hour bin). `non-shiftable` is total usage minus the listed flexible appliances, subject to consistent aggregation and treatment of missing values. **The exact private Dataport export and preprocessing provenance must be matched for exact numerical replication.**
- `data/ercot_hourly_price.csv`, columns `timestamp,Price`; verify the units and source against `utils/load_price.py` and the paper.

Place these files in `data/` locally. Never add them to a public commit. The data hashes of the published study were `c183ce46…` (`load_hourly_2018.csv`) and `56fdd5b4…` (`ercot_hourly_price.csv`); exact reproduction of the forecasts requires byte-identical inputs. To regenerate the forecast inputs used by the DDQN, follow `forecast_regeneration/REGENERATE.md`.

## Exact parsing and units

- Residential `time`: strictly `dd/mm/yyyy HH:MM` (the loader uses `%d/%m/%Y %H:%M`); one row per house per hour.
- The study's file contained 25 Dataport households with 8,754 hourly rows each; only `661`, `3039` and `8565` are used (the loaders filter by `dataid`). Its SHA-256 (`c183ce46…`) therefore depends on the full 25-household export; a file containing only the three households will not match that hash (the code filters by `dataid`, but a three-household file has not been tested).
- In the study's file `total` equals the sum of the six other columns for every row of the three households (maximum difference 2e-15).
- `air`, `car`, `clotheswasher`, `dishwasher`, `dry`, `non-shiftable`, `total`: hourly energy in kWh. In hourly bins the numerical value equals mean kW.
- Appliances correspond to the like-named hourly Pecan Street Dataport channels after aggregation into hourly intervals. Determine the source export's original granularity, timezone and missing-value treatment under your own Dataport license.
- `non-shiftable = total - (air + car + clotheswasher + dishwasher + dry)` only when these are the exact channels included in `total` and the source readings are consistent; validate non-negativity, no gaps and per-house/hour totals. No source-data preparation script is supplied that proves the original export-to-CSV transformation, so byte-identical reproduction from arbitrary exports is **not guaranteed**.
- Electricity price `timestamp` is parsed day-first; `Price` is expressed in **USD/MWh**, converted by `utils/load_price.py` to **cent/kWh** via division by 10. Match source, timezone and any missing-hour handling to the study.
