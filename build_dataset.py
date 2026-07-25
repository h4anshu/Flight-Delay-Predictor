# Rebuilds Flight_delay.csv from real BTS on-time performance data (full
# 2024 calendar year), replacing the old dataset which only contained
# pre-filtered delayed flights (ArrDelay min was 15 - zero on-time flights
# existed in it).
#
# Jan 2024 comes from Kaggle (shubhamsingh42/flight-delay-dataset-2018-2024).
# Feb-Dec 2024 come directly from BTS TranStats "Marketing Carrier On-Time
# Performance" table (identical 119-column schema, verified by diffing
# against the Kaggle file). URL pattern:
#   https://transtats.bts.gov/PREZIP/On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{year}_{month}.zip
#
# Carrier/airport full names come from the aggregated BTS "Airline_Delay_Cause"
# dataset (daryaheyko/airline-on-time-statistics-and-delay-causes-bts), which
# has the same codes with human-readable names attached.

import os
import shutil
import zipfile
import tempfile
import pandas as pd
import requests
import kagglehub

FLIGHTS_DS = "shubhamsingh42/flight-delay-dataset-2018-2024"
NAMES_DS = "daryaheyko/airline-on-time-statistics-and-delay-causes-bts"
BTS_URL = ("https://transtats.bts.gov/PREZIP/"
           "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{year}_{month}.zip")

MONTHS = [(2024, m) for m in range(1, 13)]
TARGET_ROWS = 700_000
ROWS_PER_MONTH = TARGET_ROWS // len(MONTHS)

NEEDED_RAW_COLS = [
    'DayOfWeek', 'FlightDate', 'DepTime', 'ArrTime', 'CRSArrTime',
    'IATA_Code_Marketing_Airline', 'Flight_Number_Marketing_Airline', 'Tail_Number',
    'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', 'DepDelay',
    'Origin', 'Dest', 'Distance', 'TaxiIn', 'TaxiOut', 'Cancelled',
    'CancellationCode', 'Diverted', 'CarrierDelay', 'WeatherDelay', 'NASDelay',
    'SecurityDelay', 'LateAircraftDelay',
]

ORIGINAL_COLUMNS = [
    'DayOfWeek', 'Date', 'DepTime', 'ArrTime', 'CRSArrTime', 'UniqueCarrier',
    'Airline', 'FlightNum', 'TailNum', 'ActualElapsedTime', 'CRSElapsedTime',
    'AirTime', 'ArrDelay', 'DepDelay', 'Origin', 'Org_Airport', 'Dest',
    'Dest_Airport', 'Distance', 'TaxiIn', 'TaxiOut', 'Cancelled',
    'CancellationCode', 'Diverted', 'CarrierDelay', 'WeatherDelay',
    'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
]


def hhmm_to_int(series):
    return series.astype('Int64')


def map_columns(df, carrier_name, airport_name):
    return pd.DataFrame({
        'DayOfWeek': df['DayOfWeek'],
        'Date': pd.to_datetime(df['FlightDate']).dt.strftime('%d-%m-%Y'),
        'DepTime': hhmm_to_int(df['DepTime']),
        'ArrTime': hhmm_to_int(df['ArrTime']),
        'CRSArrTime': hhmm_to_int(df['CRSArrTime']),
        'UniqueCarrier': df['IATA_Code_Marketing_Airline'],
        'Airline': df['IATA_Code_Marketing_Airline'].map(carrier_name),
        'FlightNum': df['Flight_Number_Marketing_Airline'],
        'TailNum': df['Tail_Number'],
        'ActualElapsedTime': df['ActualElapsedTime'],
        'CRSElapsedTime': df['CRSElapsedTime'],
        'AirTime': df['AirTime'],
        'ArrDelay': df['ArrDelay'],
        'DepDelay': df['DepDelay'],
        'Origin': df['Origin'],
        'Org_Airport': df['Origin'].map(airport_name),
        'Dest': df['Dest'],
        'Dest_Airport': df['Dest'].map(airport_name),
        'Distance': df['Distance'],
        'TaxiIn': df['TaxiIn'],
        'TaxiOut': df['TaxiOut'],
        'Cancelled': df['Cancelled'].astype(int),
        'CancellationCode': df['CancellationCode'],
        'Diverted': df['Diverted'].astype(int),
        'CarrierDelay': df['CarrierDelay'],
        'WeatherDelay': df['WeatherDelay'],
        'NASDelay': df['NASDelay'],
        'SecurityDelay': df['SecurityDelay'],
        'LateAircraftDelay': df['LateAircraftDelay'],
    })[ORIGINAL_COLUMNS]


def download_bts_month(year, month, cache_dir):
    zpath = os.path.join(cache_dir, f'{year}_{month}.zip')
    if not os.path.exists(zpath):
        url = BTS_URL.format(year=year, month=month)
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        with open(zpath, 'wb') as f:
            f.write(r.content)
    with zipfile.ZipFile(zpath) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith('.csv'))
        with z.open(csv_name) as f:
            return pd.read_csv(f, usecols=NEEDED_RAW_COLS)


def main():
    cache_dir = os.path.join(tempfile.gettempdir(), 'bts_months')
    os.makedirs(cache_dir, exist_ok=True)

    names_path = kagglehub.dataset_download(NAMES_DS)
    names = pd.read_csv(
        os.path.join(names_path, 'Airline_Delay_Cause.csv'),
        usecols=['carrier', 'carrier_name', 'airport', 'airport_name'],
    )
    carrier_name = dict(zip(names['carrier'], names['carrier_name']))
    airport_name = dict(zip(names['airport'], names['airport_name']))

    def sample_month(mapped):
        n = min(ROWS_PER_MONTH, len(mapped))
        return mapped.sample(n=n, random_state=42)

    flights_path = kagglehub.dataset_download(FLIGHTS_DS)
    jan_df = pd.read_parquet(os.path.join(flights_path, 'flight_data.parquet'), columns=NEEDED_RAW_COLS)
    jan_mapped = sample_month(map_columns(jan_df, carrier_name, airport_name))
    frames = [jan_mapped]
    print(f'2024-01 (kaggle): {len(jan_df):,} rows -> sampled {len(jan_mapped):,}')

    for year, month in MONTHS:
        if (year, month) == (2024, 1):
            continue
        raw = download_bts_month(year, month, cache_dir)
        mapped = sample_month(map_columns(raw, carrier_name, airport_name))
        frames.append(mapped)
        print(f'{year}-{month:02d} (BTS): {len(raw):,} rows -> sampled {len(mapped):,}')

    out = pd.concat(frames, ignore_index=True)

    if os.path.exists('Flight_delay.csv'):
        os.remove('Flight_delay.csv')

    out.to_csv('Flight_delay.csv', index=False)
    print(f'Wrote Flight_delay.csv: {out.shape[0]:,} rows, {out.shape[1]} columns')
    print(f"On-time (ArrDelay<=15): {(out['ArrDelay']<=15).mean()*100:.1f}%")
    print(f"Delayed  (ArrDelay>15): {(out['ArrDelay']>15).mean()*100:.1f}%")


if __name__ == '__main__':
    main()
