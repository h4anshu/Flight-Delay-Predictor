import pandas as pd
import json
import os
import numpy as np

def prepare_lookup_data():
    print("Loading dataset...")
    df = pd.read_csv("Flight_delay.csv")
    
    # 1. Clean Data
    df = df.dropna(subset=['Org_Airport', 'Dest_Airport'])
    df = df[(df['Cancelled'] == 0) & (df['Diverted'] == 0)]
    df['IsDelayed'] = (df['ArrDelay'] > 15).astype(int)
    
    if 'Route' not in df.columns:
        df['Route'] = df['Origin'] + '-' + df['Dest']
        
    print("Calculating historical performance stats...")
    
    # 2. Carrier Stats
    carrier_stats = df.groupby('UniqueCarrier').agg({
        'IsDelayed': 'mean',
        'ArrDelay': 'mean'
    }).reset_index()
    
    carrier_dict = {}
    for _, row in carrier_stats.iterrows():
        carrier_dict[row['UniqueCarrier']] = {
            'Carrier_DelayRate': row['IsDelayed'],
            'Carrier_AvgDelay': row['ArrDelay']
        }
        
    # 3. Origin Stats
    origin_stats = df.groupby('Origin').agg({
        'IsDelayed': 'mean',
        'ArrDelay': 'mean'
    }).reset_index()
    
    origin_dict = {}
    for _, row in origin_stats.iterrows():
        origin_dict[row['Origin']] = {
            'Origin_DelayRate': row['IsDelayed'],
            'Origin_AvgDelay': row['ArrDelay']
        }
        
    # 4. Dest Stats
    dest_stats = df.groupby('Dest').agg({
        'IsDelayed': 'mean',
        'ArrDelay': 'mean'
    }).reset_index()
    
    dest_dict = {}
    for _, row in dest_stats.iterrows():
        dest_dict[row['Dest']] = {
            'Dest_DelayRate': row['IsDelayed'],
            'Dest_AvgDelay': row['ArrDelay']
        }
        
    # 5. Route Frequency
    route_counts = df.groupby('Route').size().reset_index(name='Route_Frequency')
    route_dict = dict(zip(route_counts['Route'], route_counts['Route_Frequency']))
    median_route_frequency = float(route_counts['Route_Frequency'].median())
    
    # Defaults in case a carrier/airport is completely new
    global_carrier_delay_rate = float(carrier_stats['IsDelayed'].median())
    global_carrier_avg_delay = float(carrier_stats['ArrDelay'].median())
    global_origin_delay_rate = float(origin_stats['IsDelayed'].median())
    global_origin_avg_delay = float(origin_stats['ArrDelay'].median())
    global_dest_delay_rate = float(dest_stats['IsDelayed'].median())
    global_dest_avg_delay = float(dest_stats['ArrDelay'].median())
    
    # Extract unique valid options for the UI
    unique_carriers = sorted(df['UniqueCarrier'].dropna().unique().tolist())
    unique_origins = sorted(df['Origin'].dropna().unique().tolist())
    unique_dests = sorted(df['Dest'].dropna().unique().tolist())
    
    # Also fetch the full names if possible
    # Airline names
    airline_mapping = dict(zip(df['UniqueCarrier'], df['Airline']))
    # Airport names
    origin_mapping = dict(zip(df['Origin'], df['Org_Airport']))
    dest_mapping = dict(zip(df['Dest'], df['Dest_Airport']))
    
    airport_mapping = {**origin_mapping, **dest_mapping}

    # Final dictionary
    lookup_data = {
        'carriers': carrier_dict,
        'origins': origin_dict,
        'dests': dest_dict,
        'routes': route_dict,
        'median_route_frequency': median_route_frequency,
        'defaults': {
            'Carrier_DelayRate': global_carrier_delay_rate,
            'Carrier_AvgDelay': global_carrier_avg_delay,
            'Origin_DelayRate': global_origin_delay_rate,
            'Origin_AvgDelay': global_origin_avg_delay,
            'Dest_DelayRate': global_dest_delay_rate,
            'Dest_AvgDelay': global_dest_avg_delay,
            'Route_Frequency': 1
        },
        'ui_options': {
            'carriers': unique_carriers,
            'origins': unique_origins,
            'dests': unique_dests,
            'airline_names': airline_mapping,
            'airport_names': airport_mapping
        }
    }
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Clean up numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open('models/lookup_stats.json', 'w') as f:
        json.dump(lookup_data, f, indent=4, default=convert_numpy)
        
    print("Successfully saved models/lookup_stats.json")

if __name__ == "__main__":
    prepare_lookup_data()
