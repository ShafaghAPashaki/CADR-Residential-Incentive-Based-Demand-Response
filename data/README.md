# Data access

The repository does not redistribute the residential load dataset used in the study.
The residential electricity data were derived from the **Pecan Street Dataport** dataset and must be obtained by users under the applicable Pecan Street access terms.

The code expects the processed hourly load file at:

```text
data/load_hourly_2018.csv
```

with the columns:

```text
dataid,time,air,car,clotheswasher,dishwasher,dry,non-shiftable,total
```

The experiments use households `661`, `3039`, and `8565`.

The code also expects the 2018 hourly ERCOT price series at:

```text
data/ercot_hourly_price.csv
```

with columns:

```text
timestamp,Price
```

The price data used in the study were obtained from the source cited in the manuscript. Users should obtain the source data independently and prepare the hourly file in the format above.

Because third-party data are not redistributed here, exact end-to-end reproduction requires legitimate access to the underlying datasets.
