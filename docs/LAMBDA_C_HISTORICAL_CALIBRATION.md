# Historical tracking-weight calibration: public reproduction boundary

The coefficient `lambda_c = 0.2216381800923157` was obtained in July 2026 from a historical training-only scenario/action audit. This is a **replay of the original calculation**, not a recalibration of the released final forecast-state environment or a tuning exercise based on test performance.

## Original method

- Twelve selected training-period scenario **labels** (not twelve necessarily distinct hours); two labels selected the same original hour and each was counted.
- Three participants, five historical candidate incentive intensities per participant, and 125 joint actions for each label: 1,500 evaluations in total.
- The economic series comprised the absolute value of the normalised economic/stakeholder reward component, before adding tracking penalties.
- The tracking series comprised the **unweighted** Smooth-L1 tracking loss, before multiplication by `lambda_c`.
- The independent 95th percentiles were computed using `pandas.Series.quantile(0.95)` with **linear interpolation**. The ratio `Q95(abs(economic))/Q95(unweighted tracking)` set the coefficient.

Expected original results:

| Statistic | Value |
| --- | ---: |
| Economic Q95 | `0.13883091059361805` |
| Tracking Q95 | `0.626385357142857` |
| Ratio / tracking coefficient | `0.22163818009231573` |

The historical configuration retained `0.2216381800923157`. These are reported historical values, not values newly computed from the public repository.

## Evidence and access

The original 12 scenario labels and 1,500 archived component observations were recovered from a private historical audit workbook, and the archived calculation was checked against a historical environment snapshot. The workbook, source data, dates/hour-level scenario definitions, household identifiers, and original environment/configuration files are **not distributed** here. The previously released `calibrate_lambda_c.py` samples *different* scenarios and is a diagnostic only; its output cannot be represented as a replay of the original coefficient.

An independent authorised reproduction requires obtaining access to the relevant Pecan Street Dataport household records, the original ERCOT price source, and the appropriate private historical input and source snapshots from the rights holders or study custodians subject to their permissions. Mere access to an arbitrary Dataport sample cannot guarantee reproduction of the original values. No promise is made that the private files can be supplied.

## Data use and publication boundary

The repository's MIT licence applies to the released project code, **not to Pecan Street data or material derived from licensed data**. Applicable rights depend on the actual agreement under which the study data were acquired, which has not been established for public redistribution of the historical workbook or observation rows. See [Pecan Street's licensing overview](https://www.pecanstreet.org/dataport/licenses/) and [its publicly posted university access agreement](https://www.pecanstreet.org/wp-content/uploads/2020/01/2020-0101-Dataport-Data-License-Agreement-UniFree.pdf) for context; they do not replace the study's signed agreement. ERCOT's [website terms](https://www.ercot.com/help/terms) likewise do not establish the provenance or redistribution rights of the study's processed prices.

Accordingly, **no historical observations or third-party data are included in this documentation update**. This documentation does not assert open-data reproducibility. No model parameter, checkpoint, or scientific result has been changed.
