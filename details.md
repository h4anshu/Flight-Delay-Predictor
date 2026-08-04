# Flight Delay Predictor — Full Technical Details

This is the comprehensive reference for the project: what was broken, how it was fixed, where the data came from, what the models actually do, and how to extend any of it later. `README.md` is the short pitch; this is the whole story.

---

## 1. Goal

Predict whether a US domestic flight will arrive more than 15 minutes late, using only information known *before* departure (schedule, route, carrier, season) — no live weather, no actual departure/arrival times, nothing that would leak the answer.

---

## 2. The original bug

An audit (`audit_report.md`, `audit.py`) of the inherited project found the model always predicted "delayed," regardless of input.

**Root cause:** the target variable `IsDelayed = (ArrDelay > 15)` was computed on a dataset where `ArrDelay`'s minimum value was **exactly 15**. Verified directly against the raw CSV:

```
ArrDelay min:  15
ArrDelay max:  1707
count of ArrDelay < 15:   0
```

Every row was already a delayed flight — the source `Flight_delay.csv` (96,379 rows) was a pre-filtered "delayed flights only" extract, not a real flight-outcomes dataset. The on-time class didn't exist anywhere in the data, so no amount of threshold tuning could fix it.

**Old class balance:** 2.6% on-time / 97.4% delayed.
**Old model performance:** Logistic Regression 41.5% accuracy, XGBoost 53.1% — both *worse* than the trivial baseline of always guessing "delayed" (97.4% accurate by definition, since that's what 97.4% of the data was).

This was a **data acquisition problem, not a modeling problem**. No feature engineering, hyperparameter tuning, or threshold adjustment could have fixed it — the fix had to be a new dataset with a genuine on-time population.

---

## 3. Data sourcing

### 3.1 Source datasets

| Period | Source | Why |
|---|---|---|
| Jan 2024 | Kaggle: [`shubhamsingh42/flight-delay-dataset-2018-2024`](https://www.kaggle.com/datasets/shubhamsingh42/flight-delay-dataset-2018-2024) | Confirmed to include genuine on-time and early-arrival flights (negative `ArrDelay` values), unlike the original dataset |
| Feb–Dec 2024 | Direct from BTS TranStats "Marketing Carrier On-Time Performance" table | Same underlying government source; downloaded directly since the Kaggle dataset itself only actually contained January despite its "2018-2024" name |

### 3.2 Schema verification

Before merging BTS-direct data with the Kaggle file, the columns were diffed programmatically to confirm they were interchangeable:

- BTS's "Reporting Carrier On-Time Performance (1987-present)" table (**109 columns**, `IATA_CODE_Reporting_Airline` naming) was checked first and found to **not** match — different carrier-code field names, no marketing/operating carrier split.
- BTS's "Marketing Carrier On-Time Performance (Beginning January 2018)" table was checked next: **119/119 columns, identical names, identical order** to the Kaggle file. This is the one used.

Direct download URL pattern (no login/API key required):
```
https://transtats.bts.gov/PREZIP/On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{year}_{month}.zip
```
`{month}` is `1`–`12`, `{year}` from 2018 onward — so this same pipeline can pull additional years if ever needed.

### 3.3 Carrier / airport name lookup

Full carrier and airport names (e.g. `AA` → "American Airlines Inc.") aren't in the flight-level files — they were sourced from a second, separate BTS-derived Kaggle dataset: [`daryaheyko/airline-on-time-statistics-and-delay-causes-bts`](https://www.kaggle.com/datasets/daryaheyko/airline-on-time-statistics-and-delay-causes-bts) (an aggregated monthly delay-cause table). This dataset is **not** used for model training — only as a code → full-name lookup, verified to have 100% coverage of the carriers/airports appearing in the flight data.

### 3.4 Build script

All of the above is automated in [`build_dataset.py`](build_dataset.py):
1. Downloads/caches both Kaggle datasets and the 11 BTS monthly zip files.
2. Maps every source's columns onto one consistent 29-column schema (see §4.2).
3. Takes a random, equal-sized sample from each month (default: ~58,333/month) and concatenates them — this keeps the sample seasonally balanced rather than skewed toward months with more raw flights.
4. Writes the result to `Flight_delay.csv`.

Rerun it anytime to change the date range or sample size — the download cache means re-running with different `TARGET_ROWS` doesn't require re-downloading anything.

---

## 4. Current dataset

### 4.1 Summary stats

| | |
|---|---|
| Rows | 699,996 |
| Columns | 29 |
| Date range | 2024-01-01 to 2024-12-31 (full calendar year) |
| Sampling | ~58,333 flights per month, equal weight (not proportional to each month's raw volume) |
| Unique carriers | 10 major US airlines |
| Unique airports | 359 origins / 359 destinations |
| Cancelled | 9,480 (excluded from modeling) |
| Diverted | 1,740 (excluded from modeling) |
| Class balance | 78.8% on-time / 19.6% delayed (>15 min) — matches real-world US domestic delay rates |

This is a **sample**, not the full available year — the full year is ~7.5M flights (built once, confirmed working, then reduced to ~700K on request to keep local training fast and light on the machine it runs on). To go back to more data, increase `TARGET_ROWS` in `build_dataset.py`.

### 4.2 Full column reference

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
| `ArrDelay` | Arrival delay in minutes — **can be negative** (early arrival); this is the fix, the old data's minimum was 15 |
| `DepDelay` | Departure delay in minutes |
| `Origin` / `Org_Airport` | Origin IATA code / full airport name |
| `Dest` / `Dest_Airport` | Destination IATA code / full airport name |
| `Distance` | Miles |
| `TaxiIn` / `TaxiOut` | Taxi time in minutes |
| `Cancelled` / `CancellationCode` | Cancellation flag / reason code |
| `Diverted` | Diversion flag |
| `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay` | Post-flight delay-cause breakdown, in minutes — known only *after* the flight, used only for root-cause EDA, **never** as a model input |

---

## 5. Modeling pipeline (`final.ipynb` / `final.py`)

### 5.1 Steps

1. Load data, parse dates, filter out cancelled/diverted flights.
2. Build target: `IsDelayed = (ArrDelay > 15)`.
3. Exploratory analysis: delay rate by day of week, hour, carrier, origin/destination airport, distance, route, month/season, and delay-cause breakdown.
4. Feature engineering: historical delay-rate encodings per carrier/origin/destination, time-of-day and seasonal flags, route popularity, derived speed.
5. Encode categoricals (label encoding), scale numerics (StandardScaler fit on train only).
6. **Time-based** 80/20 train/test split — trains on earlier dates, tests on later ones, so the evaluation reflects predicting genuinely future flights rather than a random shuffle that could leak nearby-in-time information.
7. Train Logistic Regression, Random Forest (EDA/comparison only, not saved), and XGBoost, all with class-weight balancing (since the true classes are ~80/20, not 50/50).
8. Evaluate, compare, save the two deployed models (LR + XGBoost) and preprocessing artifacts.

### 5.2 Features (27 total)

**Numerical (23):** `DayOfWeek`, `Month`, `Day`, `Year`, `IsWeekend`, `CRSArr_hour`, `CRSElapsedTime`, `Distance`, `Carrier_DelayRate`, `Carrier_AvgDelay`, `Origin_DelayRate`, `Origin_AvgDelay`, `Dest_DelayRate`, `Dest_AvgDelay`, `IsHolidaySeason`, `IsSummer`, `IsRushHour`, `IsLateNight`, `Avg_Speed`, `IsShortFlight`, `IsLongFlight`, `Route_Frequency`, `IsPopularRoute`

**Categorical, label-encoded (4):** `UniqueCarrier`, `Origin`, `Dest`, `TimeBlock`

### 5.3 Explicitly excluded (data leakage)

`DepTime`, `ArrTime`, `ActualElapsedTime`, `AirTime`, `TaxiIn`, `TaxiOut`, all 5 post-flight delay-cause columns, and `ArrDelay` itself — all of these are only known *after* the flight happens, so including them would let the model "cheat" by seeing the answer (or something correlated with it) in its own inputs. This mirrors a real deployment constraint: at prediction time (someone checking before booking or before a flight), none of this exists yet.

---

## 6. Model results

688,776 flights after removing cancelled/diverted → 551,020 train / 137,756 test (time-based split).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 72.5% | 24.7% | 32.6% | 0.281 | 0.613 |
| **XGBoost (best)** | **75.5%** | 26.2% | 26.6% | 0.264 | **0.618** |

### Top 10 features by importance (XGBoost)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `TimeBlock_Encoded` | 0.147 |
| 2 | `CRSArr_hour` | 0.126 |
| 3 | `IsSummer` | 0.110 |
| 4 | `IsLateNight` | 0.090 |
| 5 | `Month` | 0.048 |
| 6 | `Carrier_DelayRate` | 0.047 |
| 7 | `IsRushHour` | 0.036 |
| 8 | `IsHolidaySeason` | 0.033 |
| 9 | `Origin_DelayRate` | 0.032 |
| 10 | `Day` | 0.031 |

Time-of-day and season dominate — makes intuitive sense: congestion and weather patterns are much more predictable pre-flight than any individual flight's specific risk.

### Business impact

Using a cost-sensitive framing ($10 per false positive — unnecessarily flagged delay; $100 per false negative — missed a real delay, since that's the more costly mistake in practice), XGBoost reduces total modeled cost by **19.1%** versus a "always predict on-time" baseline.

### Seasonal findings

- **Worst month:** July (29.0% delayed) — consistent with summer thunderstorm season.
- **Worst day:** Friday (22.0% delayed).

### Honest assessment

ROC-AUC of ~0.61–0.62 is real, modest predictive skill — **not** a solved problem. With precision ~26% and recall ~27%, most flights flagged "delayed" won't actually be, and most real delays won't be caught. This is the expected ceiling given the feature set: the model only has pre-flight, systematic information (time, season, route/carrier history) — no live weather, no ATC data, no aircraft-swap/maintenance info, all of which drive the *unpredictable* component of any specific flight's delay. More historical rows would not meaningfully move this number; new feature sources would.

---

## 7. The application (`app.py`)

A Streamlit app ("FlightIQ Dashboard") where a user enters a flight date, arrival time, airline, origin, destination, distance, and duration, and receives:

- **XGBoost prediction** with a LOW / MEDIUM / HIGH risk label. Thresholds (30% / 45%) are calibrated against the model's *actual* predicted-probability distribution on the real test set (median ~36%, 75th percentile ~46%) — not arbitrary round numbers. The true base delay rate is ~16-20%, so the model's outputs cluster well below 50%; a naive 50%-centered threshold would make "HIGH RISK" nearly unreachable.
- **Logistic Regression's** probability shown as a secondary "baseline" reading.
- **Model agreement indicator** — flags when the two models land on opposite sides of the decision boundary, a rough but real confidence signal.
- **"About this model's accuracy"** panel — shows the actual accuracy/precision/recall/ROC-AUC table live from `results/model_comparison.csv`, plus a plain-language caveat about what the model can and can't see. This exists specifically because the *original* bug was a model that projected false confidence; the fix isn't complete if the new app does the same thing in a different way.
- **Risk factors panel** — this specific carrier/origin/destination's historical delay rate (colors calibrated to the real 14-28% range), plus contextual flags (holiday season, summer, rush hour, popular route, long/short flight).

Run with:
```bash
./.venv/Scripts/streamlit.exe run app.py
```
(Always invoke tools through `.venv/Scripts/`, not a bare global command — running via a different/global Python previously produced `InconsistentVersionWarning`s from scikit-learn, since the models are pickled with the exact version pinned in `requirements.txt`.)

---

## 8. Verification performed

Every claim in this document was checked against the live system, not assumed:

- Diagnosed the original bug by loading the raw CSV and computing `ArrDelay`'s actual min/max/distribution directly — not by trusting the reported accuracy numbers.
- Verified the two BTS table schemas by diffing column lists programmatically (109 vs. 119 columns; confirmed the 119-column match was exact, same names, same order).
- After each dataset rebuild, re-ran the full notebook and confirmed **zero error cells** in the executed output.
- After each model retrain, exercised the *actual* `app.py` prediction code path (not a simplified stand-in) against contrasting low/medium/high-risk test cases and confirmed predictions varied sensibly and monotonically with real risk factors.
- Launched the actual Streamlit app in a browser, clicked through a real prediction, and confirmed no server errors and no version-mismatch warnings.
- Regenerated `final.py` from `final.ipynb` and diffed it against the committed version to confirm they're in sync (one cosmetic blank-line difference, zero logic drift).
- Confirmed git history has no accidentally-committed large files (`.venv` and `Flight_delay.csv` both correctly gitignored — the latter exceeds GitHub's 100MB per-file limit and would have failed to push).

---

## 9. File reference

| File | Purpose |
|---|---|
| `Flight_delay.csv` | Current working dataset (gitignored — too large for GitHub; rebuild with `build_dataset.py`) |
| `build_dataset.py` | Downloads/merges BTS + Kaggle source data into `Flight_delay.csv` |
| `prepare_lookup.py` | Builds `models/lookup_stats.json` (historical stats + UI dropdown options) from `Flight_delay.csv` |
| `final.ipynb` / `final.py` | Full EDA + modeling notebook (source of truth) and its plaintext mirror |
| `convert_nb.py` | One-way sync: regenerates `final.py` from `final.ipynb`'s code cells |
| `resave_model.py` | Converts the trained XGBoost `.pkl` into version-safe `.json` for `app.py` |
| `app.py` | The interactive Streamlit predictor |
| `audit.py` | Original diagnostic script that identified the broken target variable |
| `models/` | All artifacts `app.py` loads: `xgboost_model.json`, `logistic_regression_model.pkl`, `scaler.pkl`, `label_encoders.pkl`, `feature_names.json`, `lookup_stats.json` |
| `results/` | Notebook outputs: `model_comparison.csv`, `feature_importance.csv`, `test_predictions.csv` |
| `requirements.txt` | Pinned dependency versions, including the exact scikit-learn version models are pickled with |

## 10. Reproduction order

If the data is ever rebuilt (different date range, sample size, or more months added), rerun in this order — each step depends on the previous one's output:

```
build_dataset.py          → rebuilds Flight_delay.csv
prepare_lookup.py         → rebuilds models/lookup_stats.json
jupyter nbconvert --execute final.ipynb   → retrains models, saves models/*.pkl
resave_model.py           → converts XGBoost model to .json for the app
convert_nb.py             → re-syncs final.py with final.ipynb
```

## 11. Known limitations

- **Single year (2024) sample**, not the full available history — more years are available from the same BTS source if needed.
- **~700K-row sample**, not the full ~7.5M flights available for 2024 — traded off deliberately for local training speed.
- **No live/external data** — weather, ATC conditions, and aircraft-specific factors aren't available pre-flight in this pipeline, capping achievable accuracy regardless of row count.
- **Random Forest** was trained during EDA for comparison but not saved/deployed — only Logistic Regression and XGBoost are in the app.
