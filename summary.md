# Flight Delay Predictor — Project Summary

## 1. The original problem

An audit of the project (`audit_report.md`) found the model always predicted "delayed," regardless of input. Root cause: the target variable `IsDelayed = (ArrDelay > 15)` was computed on a dataset where **`ArrDelay` had a minimum value of exactly 15** — every row was already a delayed flight. The source CSV (`Flight_delay.csv`, originally 96,379 rows) was a pre-filtered "delayed flights only" extract, not a real flight-outcomes dataset. No amount of threshold tuning could fix this — the on-time class simply didn't exist in the data.

Old class balance: **2.6% on-time / 97.4% delayed**. Both models (Logistic Regression 41.5% accuracy, XGBoost 53.1%) were worse than a constant "always delayed" baseline.

## 2. New data source

Replaced with real BTS (Bureau of Transportation Statistics) on-time performance data — the same government source the original dataset was silently filtered from.

| Month | Source |
|---|---|
| Jan 2024 | Kaggle: [`shubhamsingh42/flight-delay-dataset-2018-2024`](https://www.kaggle.com/datasets/shubhamsingh42/flight-delay-dataset-2018-2024) |
| Feb–Dec 2024 | Direct from BTS TranStats, verified byte-identical (119/119 columns, same order) to the Kaggle schema |

Direct BTS URL pattern (no login required, `{year}` and `{month}` 1–12):
```
https://transtats.bts.gov/PREZIP/On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{year}_{month}.zip
```
Carrier/airport full names were filled in from a second BTS-derived Kaggle dataset: [`daryaheyko/airline-on-time-statistics-and-delay-causes-bts`](https://www.kaggle.com/datasets/daryaheyko/airline-on-time-statistics-and-delay-causes-bts) (aggregated monthly delay-cause data — not used for training, only for code→name lookups).

Both are pulled and merged by [`build_dataset.py`](build_dataset.py), which also handles the column mapping and month-by-month stratified sampling.

## 3. Current dataset — `Flight_delay.csv`

| | |
|---|---|
| Rows | 699,996 |
| Columns | 29 |
| Date range | 2024-01-01 to 2024-12-31 (full calendar year) |
| Sampling | ~58,333 flights randomly sampled from each of the 12 months (equal weight per month, so seasonality isn't skewed by months with more total flights) |
| Unique carriers | 10 (major US airlines) |
| Unique airports | 359 origins / 359 destinations |
| Cancelled flights | 9,480 (excluded from modeling) |
| Diverted flights | 1,740 (excluded from modeling) |
| Class balance | 78.8% on-time / 19.6% delayed (>15 min) — realistic, matches real-world US domestic delay rates |

Note: this is a **sample**, not the full year of BTS data. The full year is ~7.5M flights; it was cut down to ~700K on request to keep local training fast and light on the machine. To rebuild larger or smaller, edit `TARGET_ROWS` in `build_dataset.py` and rerun (source files are cached locally, no re-download needed unless the cache was cleared).

### All 29 columns

| Column | Description |
|---|---|
| `DayOfWeek` | 1 = Monday … 7 = Sunday |
| `Date` | Scheduled flight date (DD-MM-YYYY) |
| `DepTime` | Actual departure time (local, HHMM) |
| `ArrTime` | Actual arrival time (local, HHMM) |
| `CRSArrTime` | Scheduled arrival time (local, HHMM) |
| `UniqueCarrier` | IATA carrier code |
| `Airline` | Full airline name |
| `FlightNum` | Flight number |
| `TailNum` | Aircraft tail number |
| `ActualElapsedTime` | Actual total flight time (minutes) |
| `CRSElapsedTime` | Scheduled total flight time (minutes) |
| `AirTime` | Actual time airborne (minutes) |
| `ArrDelay` | Arrival delay in minutes (**can be negative** = early; this is the fix — old data had min 15) |
| `DepDelay` | Departure delay in minutes |
| `Origin` / `Org_Airport` | Origin IATA code / full airport name |
| `Dest` / `Dest_Airport` | Destination IATA code / full airport name |
| `Distance` | Miles |
| `TaxiIn` / `TaxiOut` | Taxi time in minutes |
| `Cancelled` / `CancellationCode` | Cancellation flag / reason code |
| `Diverted` | Diversion flag |
| `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay` | Post-flight delay-cause breakdown (minutes) — known only after the flight, **never used as model input**, only for root-cause EDA |

### Backup / historical files kept for reference
- None currently in the folder — the pre-fix broken dataset and the Jan-2024-only intermediate build were deleted per your request. Both are reproducible from `build_dataset.py` if ever needed again (just change `MONTHS`/`TARGET_ROWS`).

## 4. Modeling pipeline (`final.ipynb` / `final.py`)

- Filters out cancelled/diverted flights, builds `IsDelayed = (ArrDelay > 15)` as target.
- **27 features** used (no data leakage — only information known before departure):

  **Numerical (23):** `DayOfWeek`, `Month`, `Day`, `Year`, `IsWeekend`, `CRSArr_hour`, `CRSElapsedTime`, `Distance`, `Carrier_DelayRate`, `Carrier_AvgDelay`, `Origin_DelayRate`, `Origin_AvgDelay`, `Dest_DelayRate`, `Dest_AvgDelay`, `IsHolidaySeason`, `IsSummer`, `IsRushHour`, `IsLateNight`, `Avg_Speed`, `IsShortFlight`, `IsLongFlight`, `Route_Frequency`, `IsPopularRoute`

  **Categorical, label-encoded (4):** `UniqueCarrier`, `Origin`, `Dest`, `TimeBlock`

  Explicitly **excluded** (would leak the answer): `DepTime`, `ArrTime`, `ActualElapsedTime`, `AirTime`, `TaxiIn`, `TaxiOut`, all 5 delay-cause columns, `ArrDelay` itself.

- Time-based 80/20 train/test split (trains on earlier dates, tests on later ones — no future leakage).
- Three models trained: Logistic Regression, Random Forest (EDA only, not saved), XGBoost.

## 5. Model results (current, on the 700K dataset)

688,776 flights after removing cancelled/diverted → 551,020 train / 137,756 test.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 72.5% | 24.7% | 32.6% | 0.281 | 0.613 |
| **XGBoost (best)** | **75.5%** | 26.2% | 26.6% | 0.264 | **0.618** |

**Top 10 most important features (XGBoost):** `TimeBlock_Encoded` (0.147), `CRSArr_hour` (0.126), `IsSummer` (0.110), `IsLateNight` (0.090), `Month` (0.048), `Carrier_DelayRate` (0.047), `IsRushHour` (0.036), `IsHolidaySeason` (0.033), `Origin_DelayRate` (0.032), `Day` (0.031).

**Business impact:** using cost weights ($10 per false positive, $100 per false negative — missing a real delay costs more), XGBoost saves ~19% in modeled cost versus a "predict always on-time" baseline.

**Honest read on quality:** ROC-AUC ~0.61–0.62 is real, modest skill — not a "solved" problem. Precision (~26%) means most flights flagged "delayed" won't actually be; recall (~27%) means most real delays still won't be caught. This is expected: the model only sees pre-flight information (no live weather, no ATC data, no aircraft-swap info), so it can only capture systematic risk (time of day, season, route/carrier history), not day-of, unpredictable causes. Worth knowing before treating the output as more than a relative risk signal.

**Seasonal/temporal findings:** July is the worst month for delays (29.0%); Friday is the worst day (22.0%).

## 6. The app (`app.py`)

A Streamlit dashboard ("FlightIQ") where a user picks a date, time, airline, origin/destination, distance, and duration, and gets:
- XGBoost delay probability with a LOW/MEDIUM/HIGH risk label (thresholds calibrated to this model's real output range: <30% low, 30–45% medium, ≥45% high — recalibrated from the old 40/60 cutoffs, which were tuned to the broken model and almost never showed anything but "medium").
- Logistic Regression's probability as a secondary "baseline" reading, plus a flag when the two models disagree (a rough confidence signal).
- An "About this model's accuracy" panel showing the real accuracy/precision/recall/ROC-AUC table and a plain-language caveat about what the model can't see.
- A risk-factors panel: this specific carrier/origin/destination's historical delay rate (colors recalibrated to the real 14–28% range, not the old 35–50% cutoffs), plus flags like holiday season, summer, rush hour, popular route.

Run it with:
```bash
./.venv/Scripts/streamlit.exe run app.py
```

## 7. File reference

| File | Purpose |
|---|---|
| `Flight_delay.csv` | Current working dataset (see §3) |
| `build_dataset.py` | Downloads/merges BTS+Kaggle source data into `Flight_delay.csv`. Rerun to change date range or sample size. |
| `prepare_lookup.py` | Builds `models/lookup_stats.json` (historical carrier/airport/route stats + UI dropdown options) from `Flight_delay.csv`. Rerun after rebuilding the dataset. |
| `final.ipynb` / `final.py` | Full EDA + modeling notebook (source of truth) and its plaintext mirror, kept in sync via `convert_nb.py`. Rerun (`jupyter nbconvert --execute`) after rebuilding the dataset to retrain. |
| `resave_model.py` | Converts the trained XGBoost `.pkl` into version-safe `.json` format for `app.py`. Rerun after retraining. |
| `convert_nb.py` | One-way sync: regenerates `final.py` from `final.ipynb`'s code cells. Rerun after editing the notebook. |
| `app.py` | The interactive Streamlit predictor (see §6). |
| `audit.py` | Original diagnostic script that produced the audit report identifying the broken target variable. |
| `models/` | All trained artifacts `app.py` loads: `xgboost_model.json`, `logistic_regression_model.pkl`, `scaler.pkl`, `label_encoders.pkl`, `feature_names.json`, `lookup_stats.json`. |
| `results/` | Notebook outputs: `model_comparison.csv`, `feature_importance.csv`, `test_predictions.csv`. |

## 8. If you want to go further

- **More data**: full year is ~7.5M flights available (vs. the 700K sample here) — rerun `build_dataset.py` with a higher `TARGET_ROWS`, or extend `MONTHS` to cover 2018–2023 too (same BTS source, same URL pattern, just older years).
- **Better accuracy**: current ROC-AUC (~0.62) is limited mainly by feature scope (no weather/ATC data) rather than data volume — more rows alone won't move it much further; new feature sources would.
- **Re-run order after any data change**: `build_dataset.py` → `prepare_lookup.py` → execute `final.ipynb` → `resave_model.py` → `convert_nb.py` (in that order — later steps depend on earlier ones' output).
