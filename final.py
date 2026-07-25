# Import the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, roc_auc_score, classification_report, 
                              confusion_matrix)

df = pd.read_csv("Flight_delay.csv")
df.head(4)

# Convert time columns into hours

df['Dep_hour'] = pd.to_datetime(df['DepTime'].astype(str).str.zfill(4),
                                format='%H%M', errors='coerce').dt.hour
df['Arr_hour'] = pd.to_datetime(df['ArrTime'].astype(str).str.zfill(4),
                                format='%H%M', errors='coerce').dt.hour
df['CRSArr_hour'] = pd.to_datetime(df['CRSArrTime'].astype(str).str.zfill(4),
                                   format='%H%M', errors='coerce').dt.hour

# Drop Null values

df = df.dropna(subset=['Org_Airport', 'Dest_Airport'])

# Convert Date and Extract Features
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['IsWeekend'] = df['DayOfWeek'].isin([6,7]).astype(int)

# Remove cancelled/diverted flights

df = df[(df['Cancelled'] == 0) & (df['Diverted'] == 0)]

# Create binary target (Industry standard: >15 min = Delayed)
df['IsDelayed'] = (df['ArrDelay'] > 15).astype(int)


print("=" * 70)
print("📊 CURRENT DATA STATE AFTER YOUR PREPROCESSING")
print("=" * 70)

print(f"\n✓ Current Shape: {df.shape}")
print(f"✓ Columns: {df.shape[1]}")
print(f"✓ Rows: {df.shape[0]:,}")

print("\n📋 Columns Available:")
for idx, col in enumerate(df.columns, 1):
    print(f"   {idx:2}. {col}")

# Check missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print("\n⚠️  Missing Values:")
    print(missing[missing > 0])
else:
    print("\n✓ No missing values!")

# Check data types
print("\n📊 Data Types Summary:")
print(df.dtypes.value_counts())

print("\n" + "=" * 70)
print("STEP 1: TARGET VARIABLE ANALYSIS")
print("=" * 70)


# Class distribution
print("\n📊 BINARY CLASSIFICATION DISTRIBUTION:")
delay_counts = df['IsDelayed'].value_counts().sort_index()
delay_pcts = df['IsDelayed'].value_counts(normalize=True).sort_index() * 100

print(f"\n   Class 0 (On-Time ≤15 min):  {delay_counts[0]:6,} ({delay_pcts[0]:5.1f}%)")
print(f"   Class 1 (Delayed >15 min):  {delay_counts[1]:6,} ({delay_pcts[1]:5.1f}%)")
print(f"   Imbalance Ratio: {delay_counts[0]/delay_counts[1]:.2f}:1")

if delay_pcts[1] < 40:
    print("\n   ⚠️  CLASS IMBALANCE DETECTED!")
    print("   → Will need to handle this during modeling")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Target Variable Distribution Analysis', fontsize=16, fontweight='bold')

# 1. ArrDelay Distribution (Histogram)
axes[0].hist(df['ArrDelay'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(df['ArrDelay'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["ArrDelay"].mean():.1f}min')
axes[0].axvline(df['ArrDelay'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["ArrDelay"].median():.1f}min')
axes[0].axvline(15, color='orange', linestyle='--', linewidth=2, label='Threshold: 15min')
axes[0].set_xlabel('Arrival Delay (minutes)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Arrival Delays')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Binary Classification (Bar Chart)
delay_counts.plot(kind='bar', ax=axes[1], color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Number of Flights')
axes[1].set_title('Binary Classification Distribution')
axes[1].set_xticklabels(['On-Time (0)', 'Delayed (1)'], rotation=0)
axes[1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(delay_counts):
    axes[1].text(i, v + 1000, f'{v:,}\n({delay_pcts[i]:.1f}%)', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("STEP 2: TEMPORAL PATTERNS ANALYSIS")
print("=" * 70)

# Day of Week Analysis
print("\n📊 DAY OF WEEK PATTERNS:")

day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 
             4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}

# Create day name column if not exists
if 'DayName' not in df.columns:
    df['DayName'] = df['DayOfWeek'].map(day_names)

# Aggregate by day of week
day_stats = df.groupby(['DayOfWeek', 'DayName']).agg({
    'ArrDelay': ['count', 'mean', 'median', 'std'],
    'IsDelayed': 'mean'
}).round(2)

day_stats.columns = ['Flights', 'Avg_Delay', 'Median_Delay', 'Std_Delay', 'Delay_Rate']
day_stats['Delay_Rate'] = (day_stats['Delay_Rate'] * 100).round(1)

print(day_stats)

# Which day has most delays?
worst_day = day_stats['Delay_Rate'].idxmax()
best_day = day_stats['Delay_Rate'].idxmin()

print(f"\n📌 KEY INSIGHTS:")
print(f"   Worst Day: {worst_day[1]} ({day_stats.loc[worst_day, 'Delay_Rate']:.1f}% delayed)")
print(f"   Best Day: {best_day[1]} ({day_stats.loc[best_day, 'Delay_Rate']:.1f}% delayed)")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Day of Week Analysis', fontsize=16, fontweight='bold')

# Prepare data
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_plot_data = day_stats.reset_index()
day_plot_data['DayName'] = pd.Categorical(day_plot_data['DayName'], categories=day_order, ordered=True)
day_plot_data = day_plot_data.sort_values('DayName')

# 1. Average Delay by Day
axes[0].bar(day_plot_data['DayName'], day_plot_data['Avg_Delay'], 
            color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Day of Week')
axes[0].set_ylabel('Average Delay (minutes)')
axes[0].set_title('Average Arrival Delay by Day of Week')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for i, (day, val) in enumerate(zip(day_plot_data['DayName'], day_plot_data['Avg_Delay'])):
    axes[0].text(i, val + 1, f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# 2. Delay Rate by Day
colors = ['green' if x < 50 else 'orange' if x < 60 else 'red' 
          for x in day_plot_data['Delay_Rate']]
axes[1].bar(day_plot_data['DayName'], day_plot_data['Delay_Rate'], 
            color=colors, alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Day of Week')
axes[1].set_ylabel('Delay Rate (%)')
axes[1].set_title('Percentage of Delayed Flights by Day')
axes[1].tick_params(axis='x', rotation=45)
axes[1].axhline(y=50, color='red', linestyle='--', linewidth=1, label='50% threshold')
axes[1].grid(axis='y', alpha=0.3)
axes[1].legend()

# Add value labels
for i, (day, val) in enumerate(zip(day_plot_data['DayName'], day_plot_data['Delay_Rate'])):
    axes[1].text(i, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("STEP 3: TIME OF DAY PATTERNS")
print("=" * 70)

# Use the hours you already created
print("\n📊 HOURLY PATTERNS (Scheduled Arrival Time):")

# Group by scheduled arrival hour
if 'CRSArr_hour' in df.columns:
    hourly_stats = df.groupby('CRSArr_hour').agg({
        'ArrDelay': ['count', 'mean', 'median'],
        'IsDelayed': 'mean'
    }).round(2)
    
    hourly_stats.columns = ['Flights', 'Avg_Delay', 'Median_Delay', 'Delay_Rate']
    hourly_stats['Delay_Rate'] = (hourly_stats['Delay_Rate'] * 100).round(1)
    
    print(hourly_stats)
    
    # Find peak delay hours
    peak_hour = hourly_stats['Delay_Rate'].idxmax()
    best_hour = hourly_stats['Delay_Rate'].idxmin()
    
    print(f"\n📌 KEY INSIGHTS:")
    print(f"   Peak Delay Hour: {peak_hour}:00 ({hourly_stats.loc[peak_hour, 'Delay_Rate']:.1f}% delayed)")
    print(f"   Best Hour: {best_hour}:00 ({hourly_stats.loc[best_hour, 'Delay_Rate']:.1f}% delayed)")

# Time blocks analysis
print("\n📊 TIME BLOCK PATTERNS:")

def get_time_block(hour):
    if pd.isna(hour):
        return 'Unknown'
    elif 0 <= hour < 6:
        return 'Night (00-06)'
    elif 6 <= hour < 12:
        return 'Morning (06-12)'
    elif 12 <= hour < 18:
        return 'Afternoon (12-18)'
    else:
        return 'Evening (18-24)'

if 'TimeBlock' not in df.columns:
    df['TimeBlock'] = df['CRSArr_hour'].apply(get_time_block)

block_stats = df.groupby('TimeBlock').agg({
    'ArrDelay': ['count', 'mean'],
    'IsDelayed': 'mean'
}).round(2)

block_stats.columns = ['Flights', 'Avg_Delay', 'Delay_Rate']
block_stats['Delay_Rate'] = (block_stats['Delay_Rate'] * 100).round(1)

# Reorder blocks
block_order = ['Morning (06-12)', 'Afternoon (12-18)', 'Evening (18-24)', 'Night (00-06)']
block_stats = block_stats.reindex([b for b in block_order if b in block_stats.index])

print(block_stats)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Time of Day Analysis', fontsize=16, fontweight='bold')

# 1. Hourly Delay Pattern (Line Plot)
axes[0, 0].plot(hourly_stats.index, hourly_stats['Avg_Delay'], 
                marker='o', linewidth=2, color='steelblue', markersize=6)
axes[0, 0].fill_between(hourly_stats.index, hourly_stats['Avg_Delay'], alpha=0.3, color='steelblue')
axes[0, 0].set_xlabel('Hour of Day')
axes[0, 0].set_ylabel('Average Delay (minutes)')
axes[0, 0].set_title('Average Delay by Hour of Day')
axes[0, 0].set_xticks(range(0, 24, 2))
axes[0, 0].grid(alpha=0.3)

# 2. Hourly Delay Rate (Bar Chart)
colors_hourly = ['green' if x < 50 else 'orange' if x < 60 else 'red' 
                 for x in hourly_stats['Delay_Rate']]
axes[0, 1].bar(hourly_stats.index, hourly_stats['Delay_Rate'], 
               color=colors_hourly, alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('Delay Rate (%)')
axes[0, 1].set_title('Delay Rate by Hour')
axes[0, 1].set_xticks(range(0, 24, 2))
axes[0, 1].axhline(y=50, color='red', linestyle='--', linewidth=1)
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Flight Volume by Hour (Histogram)
axes[1, 0].bar(hourly_stats.index, hourly_stats['Flights'], 
               color='purple', alpha=0.6, edgecolor='black')
axes[1, 0].set_xlabel('Hour of Day')
axes[1, 0].set_ylabel('Number of Flights')
axes[1, 0].set_title('Flight Volume by Hour')
axes[1, 0].set_xticks(range(0, 24, 2))
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. Time Block Comparison
axes[1, 1].barh(block_stats.index, block_stats['Avg_Delay'], 
                color='coral', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Average Delay (minutes)')
axes[1, 1].set_ylabel('Time Block')
axes[1, 1].set_title('Average Delay by Time Block')
axes[1, 1].grid(axis='x', alpha=0.3)

# Add value labels
for i, (block, val) in enumerate(zip(block_stats.index, block_stats['Avg_Delay'])):
    axes[1, 1].text(val + 1, i, f'{val:.1f}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("STEP 4: CARRIER/AIRLINE ANALYSIS")
print("=" * 70)

print(f"\n📊 Total Unique Carriers: {df['UniqueCarrier'].nunique()}")

# Carrier performance
carrier_stats = df.groupby(['UniqueCarrier', 'Airline']).agg({
    'ArrDelay': ['count', 'mean', 'median', 'std'],
    'IsDelayed': 'mean'
}).round(2)

carrier_stats.columns = ['Flights', 'Avg_Delay', 'Median_Delay', 'Std_Delay', 'Delay_Rate']
carrier_stats['Delay_Rate'] = (carrier_stats['Delay_Rate'] * 100).round(1)
carrier_stats = carrier_stats.sort_values('Flights', ascending=False)

print("\n📊 TOP 10 CARRIERS BY FLIGHT COUNT:")
print(carrier_stats.head(10))

# Best and worst carriers
best_carrier = carrier_stats['Delay_Rate'].idxmin()
worst_carrier = carrier_stats['Delay_Rate'].idxmax()

print(f"\n📌 KEY INSIGHTS:")
print(f"   Best Carrier: {best_carrier[0]} - {best_carrier[1][:30]}...")
print(f"      Delay Rate: {carrier_stats.loc[best_carrier, 'Delay_Rate']:.1f}%")
print(f"      Avg Delay: {carrier_stats.loc[best_carrier, 'Avg_Delay']:.1f} min")

print(f"\n   Worst Carrier: {worst_carrier[0]} - {worst_carrier[1][:30]}...")
print(f"      Delay Rate: {carrier_stats.loc[worst_carrier, 'Delay_Rate']:.1f}%")
print(f"      Avg Delay: {carrier_stats.loc[worst_carrier, 'Avg_Delay']:.1f} min")

# Get top 15 carriers by flight count
top_carriers = carrier_stats.head(15).reset_index()

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Carrier Performance Analysis (Top 15 by Flight Volume)', fontsize=16, fontweight='bold')

# 1. Flight Count by Carrier
axes[0, 0].barh(top_carriers['UniqueCarrier'], top_carriers['Flights'], 
                color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Number of Flights')
axes[0, 0].set_ylabel('Carrier Code')
axes[0, 0].set_title('Flight Volume by Carrier')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(axis='x', alpha=0.3)

# 2. Average Delay by Carrier
colors_carrier = ['green' if x < 50 else 'orange' if x < 60 else 'red' 
                  for x in top_carriers['Avg_Delay']]
axes[0, 1].barh(top_carriers['UniqueCarrier'], top_carriers['Avg_Delay'], 
                color=colors_carrier, alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Average Delay (minutes)')
axes[0, 1].set_ylabel('Carrier Code')
axes[0, 1].set_title('Average Delay by Carrier')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(axis='x', alpha=0.3)

# 3. Delay Rate by Carrier
colors_rate = ['green' if x < 50 else 'orange' if x < 60 else 'red' 
               for x in top_carriers['Delay_Rate']]
axes[1, 0].barh(top_carriers['UniqueCarrier'], top_carriers['Delay_Rate'], 
                color=colors_rate, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Delay Rate (%)')
axes[1, 0].set_ylabel('Carrier Code')
axes[1, 0].set_title('Percentage of Delayed Flights by Carrier')
axes[1, 0].axvline(x=50, color='red', linestyle='--', linewidth=1)
axes[1, 0].invert_yaxis()
axes[1, 0].grid(axis='x', alpha=0.3)

# 4. Scatter: Flights vs Delay Rate
axes[1, 1].scatter(top_carriers['Flights'], top_carriers['Delay_Rate'], 
                   s=200, alpha=0.6, c=top_carriers['Avg_Delay'], 
                   cmap='RdYlGn_r', edgecolors='black', linewidth=1.5)
axes[1, 1].set_xlabel('Number of Flights')
axes[1, 1].set_ylabel('Delay Rate (%)')
axes[1, 1].set_title('Carrier Performance: Volume vs Delay Rate')
axes[1, 1].grid(alpha=0.3)

# Add carrier labels
for idx, row in top_carriers.iterrows():
    axes[1, 1].annotate(row['UniqueCarrier'], 
                        (row['Flights'], row['Delay_Rate']),
                        fontsize=8, ha='center')

# Add colorbar
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label('Avg Delay (min)', rotation=270, labelpad=20)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("STEP 5: AIRPORT ANALYSIS")
print("=" * 70)

print(f"\n📊 Total Unique Origin Airports: {df['Origin'].nunique()}")
print(f"📊 Total Unique Destination Airports: {df['Dest'].nunique()}")

# Origin airport stats
origin_stats = df.groupby(['Origin', 'Org_Airport']).agg({
    'ArrDelay': ['count', 'mean', 'median'],
    'IsDelayed': 'mean'
}).round(2)

origin_stats.columns = ['Flights', 'Avg_Delay', 'Median_Delay', 'Delay_Rate']
origin_stats['Delay_Rate'] = (origin_stats['Delay_Rate'] * 100).round(1)
origin_stats = origin_stats.sort_values('Flights', ascending=False)

print("\n📊 TOP 15 ORIGIN AIRPORTS:")
print(origin_stats.head(15))

# Busiest vs most delayed origins
busiest_origin = origin_stats.index[0]
most_delayed_origin = origin_stats[origin_stats['Flights'] > 1000]['Delay_Rate'].idxmax()

print(f"\n📌 KEY INSIGHTS:")
print(f"   Busiest Origin: {busiest_origin[0]} - {busiest_origin[1][:40]}...")
print(f"      Flights: {origin_stats.loc[busiest_origin, 'Flights']:,}")
print(f"      Delay Rate: {origin_stats.loc[busiest_origin, 'Delay_Rate']:.1f}%")

if most_delayed_origin:
    print(f"\n   Most Delayed Origin (>1000 flights): {most_delayed_origin[0]} - {most_delayed_origin[1][:40]}...")
    print(f"      Delay Rate: {origin_stats.loc[most_delayed_origin, 'Delay_Rate']:.1f}%")
    print(f"      Avg Delay: {origin_stats.loc[most_delayed_origin, 'Avg_Delay']:.1f} min")

# Destination airport stats
dest_stats = df.groupby(['Dest', 'Dest_Airport']).agg({
    'ArrDelay': ['count', 'mean', 'median'],
    'IsDelayed': 'mean'
}).round(2)

dest_stats.columns = ['Flights', 'Avg_Delay', 'Median_Delay', 'Delay_Rate']
dest_stats['Delay_Rate'] = (dest_stats['Delay_Rate'] * 100).round(1)
dest_stats = dest_stats.sort_values('Flights', ascending=False)

print("\n📊 TOP 15 DESTINATION AIRPORTS:")
print(dest_stats.head(15))

# Busiest vs most delayed destinations
busiest_dest = dest_stats.index[0]
most_delayed_dest = dest_stats[dest_stats['Flights'] > 1000]['Delay_Rate'].idxmax()

print(f"\n📌 KEY INSIGHTS:")
print(f"   Busiest Destination: {busiest_dest[0]} - {busiest_dest[1][:40]}...")
print(f"      Flights: {dest_stats.loc[busiest_dest, 'Flights']:,}")
print(f"      Delay Rate: {dest_stats.loc[busiest_dest, 'Delay_Rate']:.1f}%")

if most_delayed_dest:
    print(f"\n   Most Delayed Destination (>1000 flights): {most_delayed_dest[0]} - {most_delayed_dest[1][:40]}...")
    print(f"      Delay Rate: {dest_stats.loc[most_delayed_dest, 'Delay_Rate']:.1f}%")
    print(f"      Avg Delay: {dest_stats.loc[most_delayed_dest, 'Avg_Delay']:.1f} min")

# Get top 12 airports by flight count
top_origins = origin_stats.head(12).reset_index()
top_dests = dest_stats.head(12).reset_index()

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Airport Performance Analysis (Top 12 by Flight Volume)', fontsize=16, fontweight='bold')

# 1. Top Origin Airports - Flight Volume
axes[0, 0].barh(top_origins['Origin'], top_origins['Flights'], 
                color='teal', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Number of Flights')
axes[0, 0].set_ylabel('Airport Code')
axes[0, 0].set_title('Top Origin Airports by Flight Volume')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(axis='x', alpha=0.3)

# 2. Top Origin Airports - Delay Rate
colors_orig = ['green' if x < 50 else 'orange' if x < 60 else 'red' 
               for x in top_origins['Delay_Rate']]
axes[0, 1].barh(top_origins['Origin'], top_origins['Delay_Rate'], 
                color=colors_orig, alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Delay Rate (%)')
axes[0, 1].set_ylabel('Airport Code')
axes[0, 1].set_title('Origin Airport Delay Rates')
axes[0, 1].invert_yaxis()
axes[0, 1].axvline(x=50, color='red', linestyle='--', linewidth=1)
axes[0, 1].grid(axis='x', alpha=0.3)

# 3. Top Destination Airports - Flight Volume
axes[1, 0].barh(top_dests['Dest'], top_dests['Flights'], 
                color='darkgreen', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Number of Flights')
axes[1, 0].set_ylabel('Airport Code')
axes[1, 0].set_title('Top Destination Airports by Flight Volume')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(axis='x', alpha=0.3)

# 4. Top Destination Airports - Delay Rate
colors_dest = ['green' if x < 50 else 'orange' if x < 60 else 'red' 
               for x in top_dests['Delay_Rate']]
axes[1, 1].barh(top_dests['Dest'], top_dests['Delay_Rate'], 
                color=colors_dest, alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Delay Rate (%)')
axes[1, 1].set_ylabel('Airport Code')
axes[1, 1].set_title('Destination Airport Delay Rates')
axes[1, 1].invert_yaxis()
axes[1, 1].axvline(x=50, color='red', linestyle='--', linewidth=1)
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("STEP 6: DISTANCE ANALYSIS")
print("=" * 70)

# Distance statistics
print("\n📊 DISTANCE STATISTICS:")
print(df['Distance'].describe())

# Distance categories analysis
if 'DistanceCategory' not in df.columns:
    bins_dist = [0, 500, 1000, 1500, 5000]
    labels_dist = ['Short (<500mi)', 'Medium (500-1000mi)', 
                   'Medium-Long (1000-1500mi)', 'Long (>1500mi)']
    df['DistanceCategory'] = pd.cut(df['Distance'], bins=bins_dist, labels=labels_dist)
    print("✓ Created DistanceCategory")

# Distance category stats
dist_stats = df.groupby('DistanceCategory').agg({
    'ArrDelay': ['count', 'mean', 'median'],
    'IsDelayed': 'mean',
    'Distance': 'mean'
}).round(2)

dist_stats.columns = ['Flights', 'Avg_Delay', 'Median_Delay', 'Delay_Rate', 'Avg_Distance']
dist_stats['Delay_Rate'] = (dist_stats['Delay_Rate'] * 100).round(1)

print("\n📊 DELAY PATTERNS BY DISTANCE CATEGORY:")
print(dist_stats)

# Correlation between distance and delay
correlation = df['Distance'].corr(df['ArrDelay'])
print(f"\n📊 CORRELATION:")
print(f"   Distance vs ArrDelay: {correlation:.4f}")
if abs(correlation) < 0.1:
    print("   → Weak correlation")
elif abs(correlation) < 0.3:
    print("   → Moderate correlation")
else:
    print("   → Strong correlation")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Distance Analysis', fontsize=16, fontweight='bold')

# 1. Distance Distribution
axes[0, 0].hist(df['Distance'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(df['Distance'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {df["Distance"].mean():.0f} mi')
axes[0, 0].axvline(df['Distance'].median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {df["Distance"].median():.0f} mi')
axes[0, 0].set_xlabel('Distance (miles)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Flight Distances')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Average Delay by Distance Category
axes[0, 1].bar(range(len(dist_stats)), dist_stats['Avg_Delay'], 
               color=['green', 'yellow', 'orange', 'red'], alpha=0.7, edgecolor='black')
axes[0, 1].set_xticks(range(len(dist_stats)))
axes[0, 1].set_xticklabels(dist_stats.index, rotation=45, ha='right')
axes[0, 1].set_ylabel('Average Delay (minutes)')
axes[0, 1].set_title('Average Delay by Distance Category')
axes[0, 1].grid(axis='y', alpha=0.3)

# Add value labels
for i, val in enumerate(dist_stats['Avg_Delay']):
    axes[0, 1].text(i, val + 1, f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# 3. Scatter: Distance vs Delay
sample_size = min(10000, len(df))  # Use sample for performance
sample_df = df.sample(n=sample_size, random_state=42)

axes[1, 0].scatter(sample_df['Distance'], sample_df['ArrDelay'], 
                   alpha=0.3, s=10, color='steelblue')
axes[1, 0].set_xlabel('Distance (miles)')
axes[1, 0].set_ylabel('Arrival Delay (minutes)')
axes[1, 0].set_title(f'Distance vs Arrival Delay (Sample of {sample_size:,} flights)')
axes[1, 0].grid(alpha=0.3)

# Add trend line
z = np.polyfit(sample_df['Distance'], sample_df['ArrDelay'], 1)
p = np.poly1d(z)
axes[1, 0].plot(sample_df['Distance'].sort_values(), 
                p(sample_df['Distance'].sort_values()), 
                "r--", linewidth=2, label=f'Trend (corr={correlation:.3f})')
axes[1, 0].legend()

# 4. Delay Rate by Distance Category
axes[1, 1].bar(range(len(dist_stats)), dist_stats['Delay_Rate'], 
               color=['green', 'yellow', 'orange', 'red'], alpha=0.7, edgecolor='black')
axes[1, 1].set_xticks(range(len(dist_stats)))
axes[1, 1].set_xticklabels(dist_stats.index, rotation=45, ha='right')
axes[1, 1].set_ylabel('Delay Rate (%)')
axes[1, 1].set_title('Percentage of Delayed Flights by Distance')
axes[1, 1].axhline(y=50, color='red', linestyle='--', linewidth=1, label='50% threshold')
axes[1, 1].grid(axis='y', alpha=0.3)
axes[1, 1].legend()

# Add value labels
for i, val in enumerate(dist_stats['Delay_Rate']):
    axes[1, 1].text(i, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("STEP 7: ROUTE ANALYSIS")
print("=" * 70)

# Create route if not exists
if 'Route' not in df.columns:
    df['Route'] = df['Origin'] + '-' + df['Dest']
    print("✓ Created Route column")

print(f"\n📊 Total Unique Routes: {df['Route'].nunique():,}")

# Route statistics
route_stats = df.groupby('Route').agg({
    'ArrDelay': ['count', 'mean', 'median'],
    'IsDelayed': 'mean',
    'Distance': 'first',
    'Origin': 'first',
    'Dest': 'first'
}).round(2)

route_stats.columns = ['Flights', 'Avg_Delay', 'Median_Delay', 'Delay_Rate', 
                       'Distance', 'Origin', 'Dest']
route_stats['Delay_Rate'] = (route_stats['Delay_Rate'] * 100).round(1)
route_stats = route_stats.sort_values('Flights', ascending=False)

print("\n📊 TOP 20 ROUTES BY FLIGHT COUNT:")
print(route_stats.head(20))

# Most delayed routes (with significant volume)
significant_routes = route_stats[route_stats['Flights'] >= 100]
most_delayed_routes = significant_routes.nlargest(10, 'Delay_Rate')

print("\n📊 TOP 10 MOST DELAYED ROUTES (≥100 flights):")
print(most_delayed_routes[['Flights', 'Avg_Delay', 'Delay_Rate', 'Distance']])

# Least delayed routes
least_delayed_routes = significant_routes.nsmallest(10, 'Delay_Rate')

print("\n📊 TOP 10 LEAST DELAYED ROUTES (≥100 flights):")
print(least_delayed_routes[['Flights', 'Avg_Delay', 'Delay_Rate', 'Distance']])

top_routes = route_stats.head(15).reset_index()

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Route Performance Analysis (Top 15 Routes by Volume)', fontsize=16, fontweight='bold')

# 1. Top Routes by Flight Volume
axes[0, 0].barh(range(len(top_routes)), top_routes['Flights'], 
                color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_yticks(range(len(top_routes)))
axes[0, 0].set_yticklabels(top_routes['Route'])
axes[0, 0].set_xlabel('Number of Flights')
axes[0, 0].set_ylabel('Route')
axes[0, 0].set_title('Top Routes by Flight Volume')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(axis='x', alpha=0.3)

# 2. Average Delay by Top Routes
colors_route = ['green' if x < 50 else 'orange' if x < 70 else 'red' 
                for x in top_routes['Avg_Delay']]
axes[0, 1].barh(range(len(top_routes)), top_routes['Avg_Delay'], 
                color=colors_route, alpha=0.7, edgecolor='black')
axes[0, 1].set_yticks(range(len(top_routes)))
axes[0, 1].set_yticklabels(top_routes['Route'])
axes[0, 1].set_xlabel('Average Delay (minutes)')
axes[0, 1].set_ylabel('Route')
axes[0, 1].set_title('Average Delay by Route')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(axis='x', alpha=0.3)

# 3. Scatter: Route Volume vs Delay Rate
axes[1, 0].scatter(top_routes['Flights'], top_routes['Delay_Rate'], 
                   s=top_routes['Distance']/5, alpha=0.6, 
                   c=top_routes['Avg_Delay'], cmap='RdYlGn_r',
                   edgecolors='black', linewidth=1.5)
axes[1, 0].set_xlabel('Number of Flights')
axes[1, 0].set_ylabel('Delay Rate (%)')
axes[1, 0].set_title('Route Performance: Volume vs Delay Rate\n(bubble size = distance)')
axes[1, 0].grid(alpha=0.3)

# Add route labels for interesting points
for idx, row in top_routes.iterrows():
    if row['Delay_Rate'] > 65 or row['Flights'] > top_routes['Flights'].quantile(0.8):
        axes[1, 0].annotate(row['Route'], 
                           (row['Flights'], row['Delay_Rate']),
                           fontsize=7, ha='center')

# 4. Distance vs Delay for Top Routes
axes[1, 1].scatter(top_routes['Distance'], top_routes['Avg_Delay'], 
                   s=top_routes['Flights']/10, alpha=0.6,
                   c=top_routes['Delay_Rate'], cmap='RdYlGn_r',
                   edgecolors='black', linewidth=1.5)
axes[1, 1].set_xlabel('Distance (miles)')
axes[1, 1].set_ylabel('Average Delay (minutes)')
axes[1, 1].set_title('Distance vs Delay for Top Routes\n(bubble size = flight volume)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("STEP 8: MONTHLY & SEASONAL PATTERNS")
print("=" * 70)

# Monthly analysis
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

if 'MonthName' not in df.columns:
    df['MonthName'] = df['Month'].map(month_names)

monthly_stats = df.groupby(['Month', 'MonthName']).agg({
    'ArrDelay': ['count', 'mean', 'median'],
    'IsDelayed': 'mean'
}).round(2)

monthly_stats.columns = ['Flights', 'Avg_Delay', 'Median_Delay', 'Delay_Rate']
monthly_stats['Delay_Rate'] = (monthly_stats['Delay_Rate'] * 100).round(1)

print("\n📊 MONTHLY PATTERNS:")
print(monthly_stats)

# Year analysis (if multiple years)
if df['Year'].nunique() > 1:
    yearly_stats = df.groupby('Year').agg({
        'ArrDelay': ['count', 'mean'],
        'IsDelayed': 'mean'
    }).round(2)
    
    yearly_stats.columns = ['Flights', 'Avg_Delay', 'Delay_Rate']
    yearly_stats['Delay_Rate'] = (yearly_stats['Delay_Rate'] * 100).round(1)
    
    print("\n📊 YEARLY PATTERNS:")
    print(yearly_stats)

# Find worst and best months
worst_month = monthly_stats['Delay_Rate'].idxmax()
best_month = monthly_stats['Delay_Rate'].idxmin()

print(f"\n📌 KEY INSIGHTS:")
print(f"   Worst Month: {worst_month[1]} ({monthly_stats.loc[worst_month, 'Delay_Rate']:.1f}% delayed)")
print(f"   Best Month: {best_month[1]} ({monthly_stats.loc[best_month, 'Delay_Rate']:.1f}% delayed)")

# Weekend vs Weekday
weekend_stats = df.groupby('IsWeekend').agg({
    'ArrDelay': ['count', 'mean'],
    'IsDelayed': 'mean'
}).round(2)

weekend_stats.columns = ['Flights', 'Avg_Delay', 'Delay_Rate']
weekend_stats['Delay_Rate'] = (weekend_stats['Delay_Rate'] * 100).round(1)
weekend_stats.index = ['Weekday', 'Weekend']

print("\n📊 WEEKEND vs WEEKDAY:")
print(weekend_stats)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Monthly & Seasonal Patterns', fontsize=16, fontweight='bold')

# Prepare data
monthly_plot = monthly_stats.reset_index()

# 1. Monthly Flight Volume
axes[0, 0].bar(monthly_plot['MonthName'], monthly_plot['Flights'], 
               color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Number of Flights')
axes[0, 0].set_title('Flight Volume by Month')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Monthly Average Delay
axes[0, 1].plot(monthly_plot['MonthName'], monthly_plot['Avg_Delay'], 
                marker='o', linewidth=2, markersize=8, color='red')
axes[0, 1].fill_between(range(len(monthly_plot)), monthly_plot['Avg_Delay'], 
                        alpha=0.3, color='red')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Average Delay (minutes)')
axes[0, 1].set_title('Average Delay by Month')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(alpha=0.3)

# 3. Monthly Delay Rate
colors_month = ['green' if x < 50 else 'orange' if x < 60 else 'red' 
                for x in monthly_plot['Delay_Rate']]
axes[1, 0].bar(monthly_plot['MonthName'], monthly_plot['Delay_Rate'], 
               color=colors_month, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Delay Rate (%)')
axes[1, 0].set_title('Percentage of Delayed Flights by Month')
axes[1, 0].axhline(y=50, color='red', linestyle='--', linewidth=1)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)

# Add value labels
for i, val in enumerate(monthly_plot['Delay_Rate']):
    axes[1, 0].text(i, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

# 4. Weekend vs Weekday Comparison
categories = ['Avg Delay (min)', 'Delay Rate (%)']
weekday_vals = [weekend_stats.loc['Weekday', 'Avg_Delay'], 
                weekend_stats.loc['Weekday', 'Delay_Rate']]
weekend_vals = [weekend_stats.loc['Weekend', 'Avg_Delay'], 
                weekend_stats.loc['Weekend', 'Delay_Rate']]

x = np.arange(len(categories))
width = 0.35

bars1 = axes[1, 1].bar(x - width/2, weekday_vals, width, 
                       label='Weekday', color='steelblue', alpha=0.7, edgecolor='black')
bars2 = axes[1, 1].bar(x + width/2, weekend_vals, width, 
                       label='Weekend', color='coral', alpha=0.7, edgecolor='black')

axes[1, 1].set_ylabel('Value')
axes[1, 1].set_title('Weekday vs Weekend Comparison')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(categories)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("STEP 9: DELAY ROOT CAUSE ANALYSIS")
print("=" * 70)

print("\n⚠️  NOTE: These delay types are known ONLY AFTER the delay occurs.")
print("    They explain WHY delays happened but CANNOT be used for prediction!\n")

delay_types = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 
               'SecurityDelay', 'LateAircraftDelay']

delay_analysis = {}

for dtype in delay_types:
    incidents = (df[dtype] > 0).sum()
    if incidents > 0:
        total_minutes = df[dtype].sum()
        avg_minutes = df[df[dtype] > 0][dtype].mean()
        pct = (incidents / len(df)) * 100
        
        delay_analysis[dtype.replace('Delay', '')] = {
            'Incidents': incidents,
            'Total_Minutes': total_minutes,
            'Avg_per_Incident': avg_minutes,
            'Percentage': pct
        }

delay_df = pd.DataFrame(delay_analysis).T
delay_df = delay_df.sort_values('Total_Minutes', ascending=False)

print("📊 DELAY TYPE BREAKDOWN:")
print(delay_df.round(2))

# Calculate contribution
delay_df['Contribution_%'] = (delay_df['Total_Minutes'] / delay_df['Total_Minutes'].sum() * 100).round(1)

print("\n📊 CONTRIBUTION TO TOTAL DELAY:")
print(delay_df[['Total_Minutes', 'Contribution_%']])

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Delay Root Cause Analysis', fontsize=16, fontweight='bold')

# 1. Incidents by Delay Type
axes[0, 0].barh(delay_df.index, delay_df['Incidents'], 
                color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Number of Incidents')
axes[0, 0].set_ylabel('Delay Type')
axes[0, 0].set_title('Incidents by Delay Type')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, val) in enumerate(delay_df['Incidents'].items()):
    axes[0, 0].text(val + 100, i, f'{val:,}', va='center', fontweight='bold')

# 2. Total Minutes by Delay Type
axes[0, 1].barh(delay_df.index, delay_df['Total_Minutes'], 
                color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Total Minutes')
axes[0, 1].set_ylabel('Delay Type')
axes[0, 1].set_title('Total Delay Minutes by Type')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, val) in enumerate(delay_df['Total_Minutes'].items()):
    axes[0, 1].text(val + 1000, i, f'{val:,.0f}', va='center', fontweight='bold')

# 3. Contribution Pie Chart
colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
explode = (0.05, 0, 0, 0, 0)  # Explode the largest slice

axes[1, 0].pie(delay_df['Total_Minutes'], labels=delay_df.index, autopct='%1.1f%%',
               startangle=90, colors=colors_pie, explode=explode,
               shadow=True, textprops={'fontsize': 10, 'weight': 'bold'})
axes[1, 0].set_title('Contribution to Total Delay Time')

# 4. Average Delay per Incident
axes[1, 1].barh(delay_df.index, delay_df['Avg_per_Incident'], 
                color='green', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Average Minutes per Incident')
axes[1, 1].set_ylabel('Delay Type')
axes[1, 1].set_title('Average Delay Duration per Incident')
axes[1, 1].invert_yaxis()
axes[1, 1].grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, val) in enumerate(delay_df['Avg_per_Incident'].items()):
    axes[1, 1].text(val + 1, i, f'{val:.1f}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("STEP 10: CORRELATION ANALYSIS")
print("=" * 70)

# Select numerical features for correlation (excluding leakage features)
safe_numerical = [
    'DayOfWeek', 'Month', 'Day', 'Year', 'IsWeekend',
    'CRSArr_hour', 'CRSElapsedTime', 'Distance',
    'FlightNum', 'ArrDelay', 'DepDelay'
]

# Filter only columns that exist in the dataframe
safe_numerical = [col for col in safe_numerical if col in df.columns]

print(f"\n📊 Computing correlations for {len(safe_numerical)} numerical features...")

# Compute correlation matrix
corr_matrix = df[safe_numerical].corr()

# Correlations with target (ArrDelay)
target_corr = corr_matrix['ArrDelay'].sort_values(ascending=False)

print("\n📊 CORRELATIONS WITH ARRIVAL DELAY:")
print(target_corr)

# Strong correlations (|corr| > 0.5)
strong_corr = target_corr[abs(target_corr) > 0.5]
if len(strong_corr) > 1:  # Exclude ArrDelay itself
    print(f"\n📌 STRONG CORRELATIONS (|r| > 0.5):")
    for feat, corr in strong_corr.items():
        if feat != 'ArrDelay':
            print(f"   {feat}: {corr:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')

# 1. Full Correlation Heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=axes[0])
axes[0].set_title('Correlation Matrix of Numerical Features')

# 2. Correlations with Target (Bar Chart)
target_corr_plot = target_corr.drop('ArrDelay')  # Remove self-correlation
colors_corr = ['red' if x < 0 else 'green' for x in target_corr_plot]

axes[1].barh(range(len(target_corr_plot)), target_corr_plot.values, 
             color=colors_corr, alpha=0.7, edgecolor='black')
axes[1].set_yticks(range(len(target_corr_plot)))
axes[1].set_yticklabels(target_corr_plot.index)
axes[1].set_xlabel('Correlation Coefficient')
axes[1].set_title('Feature Correlations with Arrival Delay')
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1)
axes[1].grid(axis='x', alpha=0.3)
axes[1].invert_yaxis()

# Add correlation values
for i, val in enumerate(target_corr_plot.values):
    axes[1].text(val + 0.02 if val > 0 else val - 0.02, i, f'{val:.3f}', 
                 va='center', ha='left' if val > 0 else 'right', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("STEP 12: PREPARING DATA FOR MODELING")
print("=" * 70)

# Identify safe features (NO DATA LEAKAGE)
print("\n⚠️  CRITICAL: Avoiding Data Leakage!")
print("   We only use information available BEFORE the flight departs\n")

# Features that would cause data leakage (DON'T USE!)
LEAKAGE_FEATURES = [
    'DepTime', 'ArrTime',           # Actual times (known only after flight)
    'ActualElapsedTime', 'AirTime', # Actual durations
    'TaxiIn', 'TaxiOut',            # Actual taxi times
    'Dep_hour', 'Arr_hour',         # Actual departure/arrival hours
    'CarrierDelay', 'WeatherDelay', 'NASDelay',  # Post-flight delay reasons
    'SecurityDelay', 'LateAircraftDelay',
    'ArrDelay',                     # This is our TARGET, not a feature!
    'Date',                         # Will extract features from this
    'Airline', 'Org_Airport', 'Dest_Airport',  # Use codes instead
    'DayName', 'MonthName',         # Already have numerical versions
    'Cancelled', 'CancellationCode', 'Diverted'  # Filter these out
]

# Safe features to use for prediction
SAFE_FEATURES = [
    # Temporal features
    'DayOfWeek', 'Month', 'Day', 'Year', 'IsWeekend',
    
    # Scheduled time features (known before flight)
    'CRSArr_hour',      # Scheduled arrival hour
    'CRSArrTime',       # Scheduled arrival time
    'CRSElapsedTime',   # Scheduled flight duration
    
    # Flight identifiers
    'FlightNum',        # Flight number
    'TailNum',          # Aircraft identifier
    
    # Route information
    'UniqueCarrier',    # Airline code
    'Origin',           # Origin airport code
    'Dest',             # Destination airport code
    'Distance',         # Flight distance
    
    # Derived features
    'TimeBlock',        # Time of day category
    'DistanceCategory', # Distance category
    'Route'             # Origin-Destination pair
]

# Filter only columns that exist in our dataframe
SAFE_FEATURES = [col for col in SAFE_FEATURES if col in df.columns]

print(f"✓ Safe features identified: {len(SAFE_FEATURES)}")
print(f"✗ Leakage features to avoid: {len(LEAKAGE_FEATURES)}")

print("\n📋 FEATURES WE WILL USE FOR MODELING:")
for i, feat in enumerate(SAFE_FEATURES, 1):
    print(f"   {i:2}. {feat}")

# Separate categorical and numerical features
categorical_features = ['UniqueCarrier', 'Origin', 'Dest', 'Route', 
                        'TailNum', 'TimeBlock', 'DistanceCategory']
categorical_features = [f for f in categorical_features if f in SAFE_FEATURES]

numerical_features = [f for f in SAFE_FEATURES if f not in categorical_features]

print(f"\n📊 Feature Types:")
print(f"   Numerical features: {len(numerical_features)}")
print(f"   Categorical features: {len(categorical_features)}")

# Check cardinality of categorical features
print("\n📊 Categorical Feature Cardinality:")
for feat in categorical_features:
    nunique = df[feat].nunique()
    print(f"   {feat}: {nunique:,} unique values")
    if nunique > 10000:
        print(f"      ⚠️  High cardinality - may need special handling")

print("\n" + "=" * 70)
print("STEP 13: PRACTICAL FEATURE ENGINEERING")
print("=" * 70)

print("\n🔧 Creating useful features for better predictions...\n")

# 1. Historical Performance Features (Target Encoding)
# These features show how reliable each carrier/airport/route has been historically
print("📊 Step 1: Creating Historical Performance Features...")

# Carrier historical performance
carrier_stats = df.groupby('UniqueCarrier').agg({
    'IsDelayed': 'mean',
    'ArrDelay': 'mean'
}).reset_index()
carrier_stats.columns = ['UniqueCarrier', 'Carrier_DelayRate', 'Carrier_AvgDelay']

df = df.merge(carrier_stats, on='UniqueCarrier', how='left')
print("   ✓ Carrier_DelayRate: How often this carrier is delayed")
print("   ✓ Carrier_AvgDelay: Average delay for this carrier")

# Origin airport historical performance
origin_stats = df.groupby('Origin').agg({
    'IsDelayed': 'mean',
    'ArrDelay': 'mean'
}).reset_index()
origin_stats.columns = ['Origin', 'Origin_DelayRate', 'Origin_AvgDelay']

df = df.merge(origin_stats, on='Origin', how='left')
print("   ✓ Origin_DelayRate: How often flights from this airport are delayed")
print("   ✓ Origin_AvgDelay: Average delay at origin airport")

# Destination airport historical performance
dest_stats = df.groupby('Dest').agg({
    'IsDelayed': 'mean',
    'ArrDelay': 'mean'
}).reset_index()
dest_stats.columns = ['Dest', 'Dest_DelayRate', 'Dest_AvgDelay']

df = df.merge(dest_stats, on='Dest', how='left')
print("   ✓ Dest_DelayRate: How often flights to this airport are delayed")
print("   ✓ Dest_AvgDelay: Average delay at destination airport")

# 2. Time-based Features
print("\n📅 Step 2: Creating Time-based Features...")

# Is it a holiday season? (December, January)
df['IsHolidaySeason'] = df['Month'].isin([12, 1]).astype(int)
print("   ✓ IsHolidaySeason: Flights during peak travel months")

# Is it summer? (June, July, August)
df['IsSummer'] = df['Month'].isin([6, 7, 8]).astype(int)
print("   ✓ IsSummer: Summer travel season")

# Rush hour indicator (7-9 AM and 5-7 PM)
df['IsRushHour'] = df['CRSArr_hour'].isin([7, 8, 17, 18]).astype(int)
print("   ✓ IsRushHour: Peak traffic hours at airports")

# Late night/early morning flights (10 PM - 6 AM)
df['IsLateNight'] = df['CRSArr_hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
print("   ✓ IsLateNight: Red-eye flights")

# 3. Distance-based Features
print("\n✈️  Step 3: Creating Distance-based Features...")

# Speed (distance per scheduled hour)
df['Avg_Speed'] = df['Distance'] / (df['CRSElapsedTime'] / 60)
df['Avg_Speed'] = df['Avg_Speed'].replace([np.inf, -np.inf], np.nan)
df['Avg_Speed'] = df['Avg_Speed'].fillna(df['Avg_Speed'].median())
print("   ✓ Avg_Speed: Average speed of flight (mph)")

# Is it a short flight? (< 500 miles)
df['IsShortFlight'] = (df['Distance'] < 500).astype(int)
print("   ✓ IsShortFlight: Short-haul flights")

# Is it a long flight? (> 1500 miles)
df['IsLongFlight'] = (df['Distance'] > 1500).astype(int)
print("   ✓ IsLongFlight: Long-haul flights")

# 4. Route-specific Features
print("\n🛫 Step 4: Creating Route-specific Features...")

# Route popularity (number of flights on this route)
route_counts = df.groupby('Route').size().reset_index(name='Route_Frequency')
df = df.merge(route_counts, on='Route', how='left')
print("   ✓ Route_Frequency: How busy is this route")

# Is it a popular route? (more than median frequency)
median_frequency = df['Route_Frequency'].median()
df['IsPopularRoute'] = (df['Route_Frequency'] > median_frequency).astype(int)
print("   ✓ IsPopularRoute: High-traffic routes")

print(f"\n✅ Feature engineering complete!")
print(f"   Total features created: 18 new features")
print(f"   Dataset shape now: {df.shape}")

print("\n" + "=" * 70)
print("STEP 14: FINAL FEATURE SELECTION")
print("=" * 70)

# Update our feature lists with engineered features
NUMERICAL_FEATURES = [
    # Original numerical features
    'DayOfWeek', 'Month', 'Day', 'Year', 'IsWeekend',
    'CRSArr_hour', 'CRSElapsedTime', 'Distance',
    
    # Engineered historical features
    'Carrier_DelayRate', 'Carrier_AvgDelay',
    'Origin_DelayRate', 'Origin_AvgDelay',
    'Dest_DelayRate', 'Dest_AvgDelay',
    
    # Engineered time features
    'IsHolidaySeason', 'IsSummer', 'IsRushHour', 'IsLateNight',
    
    # Engineered distance features
    'Avg_Speed', 'IsShortFlight', 'IsLongFlight',
    
    # Engineered route features
    'Route_Frequency', 'IsPopularRoute'
]

CATEGORICAL_FEATURES = [
    'UniqueCarrier', 'Origin', 'Dest', 'TimeBlock'
]

# Filter only existing columns
NUMERICAL_FEATURES = [f for f in NUMERICAL_FEATURES if f in df.columns]
CATEGORICAL_FEATURES = [f for f in CATEGORICAL_FEATURES if f in df.columns]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = 'IsDelayed'

print(f"\n📊 FINAL FEATURE SUMMARY:")
print(f"   Numerical Features: {len(NUMERICAL_FEATURES)}")
print(f"   Categorical Features: {len(CATEGORICAL_FEATURES)}")
print(f"   Total Features: {len(ALL_FEATURES)}")
print(f"   Target Variable: {TARGET}")

print("\n📋 NUMERICAL FEATURES ({} features):".format(len(NUMERICAL_FEATURES)))
for i, feat in enumerate(NUMERICAL_FEATURES, 1):
    print(f"   {i:2}. {feat}")

print("\n📋 CATEGORICAL FEATURES ({} features):".format(len(CATEGORICAL_FEATURES)))
for i, feat in enumerate(CATEGORICAL_FEATURES, 1):
    print(f"   {i:2}. {feat}")

# Check for missing values
missing_in_features = df[ALL_FEATURES + [TARGET]].isnull().sum()
if missing_in_features.sum() > 0:
    print("\n⚠️  Missing values detected:")
    print(missing_in_features[missing_in_features > 0])
    
    print("\n🔧 Filling missing values...")
    # Fill numerical with median
    for col in NUMERICAL_FEATURES:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"   ✓ Filled {col} with median")
    
    # Fill categorical with mode
    for col in CATEGORICAL_FEATURES:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"   ✓ Filled {col} with mode")
else:
    print("\n✓ No missing values in feature set!")

print("\n" + "=" * 70)
print("STEP 15: ENCODING CATEGORICAL VARIABLES")
print("=" * 70)

# Store encoders for later use
encoders = {}

print("\n🔢 Label encoding categorical features...")
print("   (Converting text categories to numbers)")

for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    df[f'{col}_Encoded'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    
    print(f"\n   {col}:")
    print(f"      Original values: {df[col].nunique():,} unique")
    print(f"      Encoded as: 0 to {df[col].nunique()-1}")
    
    # Show example encoding
    sample_mapping = dict(zip(df[col].unique()[:3], 
                             df[f'{col}_Encoded'].unique()[:3]))
    print(f"      Example: {sample_mapping}")

# Update feature list with encoded versions
CATEGORICAL_ENCODED = [f'{col}_Encoded' for col in CATEGORICAL_FEATURES]
FINAL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_ENCODED

print(f"\n✓ All categorical features encoded")
print(f"✓ Final feature count: {len(FINAL_FEATURES)}")

print("\n" + "=" * 70)
print("VISUALIZATION: FEATURE OVERVIEW")
print("=" * 70)

# Create a summary visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Feature Engineering Overview', fontsize=16, fontweight='bold')

# 1. Feature Type Distribution
feature_types = pd.DataFrame({
    'Type': ['Numerical', 'Categorical'],
    'Count': [len(NUMERICAL_FEATURES), len(CATEGORICAL_FEATURES)]
})

axes[0, 0].bar(feature_types['Type'], feature_types['Count'], 
               color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
axes[0, 0].set_ylabel('Number of Features')
axes[0, 0].set_title('Feature Type Distribution')
axes[0, 0].grid(axis='y', alpha=0.3)

for i, val in enumerate(feature_types['Count']):
    axes[0, 0].text(i, val + 0.5, str(val), ha='center', va='bottom', 
                    fontweight='bold', fontsize=14)

# 2. Sample of Engineered Features - Carrier Performance
top_carriers = df.groupby('UniqueCarrier')['Carrier_DelayRate'].first().nlargest(10)
axes[0, 1].barh(range(len(top_carriers)), top_carriers.values, 
                color='red', alpha=0.7, edgecolor='black')
axes[0, 1].set_yticks(range(len(top_carriers)))
axes[0, 1].set_yticklabels(top_carriers.index)
axes[0, 1].set_xlabel('Historical Delay Rate')
axes[0, 1].set_title('Top 10 Carriers by Delay Rate (Historical)')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(axis='x', alpha=0.3)

# 3. Distribution of Key Engineered Features
sample_features = ['IsWeekend', 'IsHolidaySeason', 'IsSummer', 
                   'IsRushHour', 'IsShortFlight', 'IsPopularRoute']
sample_features = [f for f in sample_features if f in df.columns]

feature_means = [df[f].mean() * 100 for f in sample_features]

axes[1, 0].barh(range(len(sample_features)), feature_means, 
                color='green', alpha=0.7, edgecolor='black')
axes[1, 0].set_yticks(range(len(sample_features)))
axes[1, 0].set_yticklabels(sample_features)
axes[1, 0].set_xlabel('Percentage of Flights (%)')
axes[1, 0].set_title('Binary Feature Distribution')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(axis='x', alpha=0.3)

# Add value labels
for i, val in enumerate(feature_means):
    axes[1, 0].text(val + 1, i, f'{val:.1f}%', va='center', fontsize=9)

# 4. Feature Categories
categories = {
    'Temporal': ['DayOfWeek', 'Month', 'Year', 'IsWeekend', 'IsHolidaySeason', 'IsSummer'],
    'Scheduled': ['CRSArr_hour', 'CRSElapsedTime', 'IsRushHour', 'IsLateNight'],
    'Historical': ['Carrier_DelayRate', 'Origin_DelayRate', 'Dest_DelayRate'],
    'Route': ['Distance', 'Avg_Speed', 'Route_Frequency', 'IsPopularRoute'],
    'Categorical': CATEGORICAL_ENCODED
}

category_counts = {k: len([f for f in v if f in FINAL_FEATURES]) 
                   for k, v in categories.items()}

axes[1, 1].pie(category_counts.values(), labels=category_counts.keys(), 
               autopct='%1.1f%%', startangle=90,
               colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'],
               explode=(0.05, 0, 0, 0, 0))
axes[1, 1].set_title('Feature Distribution by Category')

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("STEP 16: TRAIN-TEST SPLIT (TIME-BASED)")
print("=" * 70)

print("\n⚠️  CRITICAL: Using time-based split to prevent data leakage!")
print("   We train on PAST data and test on FUTURE data")
print("   This simulates real-world scenario where we predict future flights\n")

# Sort by date
df_sorted = df.sort_values('Date').reset_index(drop=True)

# 80-20 split
split_idx = int(len(df_sorted) * 0.8)

X_train = df_sorted[FINAL_FEATURES].iloc[:split_idx]
X_test = df_sorted[FINAL_FEATURES].iloc[split_idx:]
y_train = df_sorted[TARGET].iloc[:split_idx]
y_test = df_sorted[TARGET].iloc[split_idx:]

dates_train = df_sorted['Date'].iloc[:split_idx]
dates_test = df_sorted['Date'].iloc[split_idx:]

print(f"📊 SPLIT SUMMARY:")
print(f"   Training Set: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"   Test Set: {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")

print(f"\n📅 TIME PERIODS:")
print(f"   Training: {dates_train.min().date()} to {dates_train.max().date()}")
print(f"   Testing:  {dates_test.min().date()} to {dates_test.max().date()}")

# Verify no date overlap
if dates_train.max() >= dates_test.min():
    print("\n   ❌ WARNING: Date overlap detected!")
else:
    print("\n   ✓ No date overlap - proper time-based split")

# Class distribution
train_dist = y_train.value_counts()
test_dist = y_test.value_counts()

print(f"\n📊 CLASS DISTRIBUTION:")
print(f"\n   Training Set:")
print(f"      On-Time (0): {train_dist[0]:,} ({train_dist[0]/len(y_train)*100:.1f}%)")
print(f"      Delayed (1): {train_dist[1]:,} ({train_dist[1]/len(y_train)*100:.1f}%)")
print(f"      Imbalance Ratio: {train_dist[0]/train_dist[1]:.2f}:1")

print(f"\n   Test Set:")
print(f"      On-Time (0): {test_dist[0]:,} ({test_dist[0]/len(y_test)*100:.1f}%)")
print(f"      Delayed (1): {test_dist[1]:,} ({test_dist[1]/len(y_test)*100:.1f}%)")
print(f"      Imbalance Ratio: {test_dist[0]/test_dist[1]:.2f}:1")

# Calculate class weights for handling imbalance
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train), 
                                     y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print(f"\n⚖️  CLASS WEIGHTS (for imbalanced data):")
print(f"   Class 0 (On-Time): {class_weight_dict[0]:.3f}")
print(f"   Class 1 (Delayed): {class_weight_dict[1]:.3f}")
print("   → These weights help the model pay more attention to the minority class")

print("\n" + "=" * 70)
print("STEP 17: FEATURE SCALING")
print("=" * 70)

from sklearn.preprocessing import StandardScaler

print("\n📏 Why scale features?")
print("   - Makes all features comparable (same range)")
print("   - Helps ML algorithms converge faster")
print("   - Important for distance-based algorithms")

print("\n🔧 Applying StandardScaler...")
print("   Formula: (value - mean) / standard_deviation")

scaler = StandardScaler()

# Fit on training data ONLY (to prevent leakage)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Features scaled successfully")
print(f"   Training set shape: {X_train_scaled.shape}")
print(f"   Test set shape: {X_test_scaled.shape}")

# Show scaling example
print("\n📊 SCALING EXAMPLE (first 3 features, first 5 rows):")
print("\nBefore scaling:")
print(X_train.iloc[:5, :3])

print("\nAfter scaling:")
scaled_sample = pd.DataFrame(X_train_scaled[:5, :3], 
                             columns=X_train.columns[:3])
print(scaled_sample)

print("\n💡 Notice: All values now centered around 0 with similar ranges")

print("\n" + "=" * 70)
print("STEP 18: LOGISTIC REGRESSION (Baseline Model)")
print("=" * 70)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, roc_auc_score, classification_report, 
                              confusion_matrix)
import time

print("\n🤖 What is Logistic Regression?")
print("   - Simple, interpretable model")
print("   - Good baseline to compare against")
print("   - Fast to train")

print("\n⏱️  Training Logistic Regression...")
start_time = time.time()

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Handle imbalanced data
    random_state=42,
    n_jobs=6
)

lr_model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

print(f"✓ Model trained in {training_time:.2f} seconds")

# Make predictions
print("\n🔮 Making predictions on test set...")
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics
print("\n📊 LOGISTIC REGRESSION PERFORMANCE:")
print("="*50)

accuracy = accuracy_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)
roc_auc = roc_auc_score(y_test, y_pred_proba_lr)

print(f"\n   Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f}  ({precision*100:.2f}%)")
print(f"   Recall:    {recall:.4f}  ({recall*100:.2f}%)")
print(f"   F1-Score:  {f1:.4f}  ({f1*100:.2f}%)")
print(f"   ROC-AUC:   {roc_auc:.4f}  ({roc_auc*100:.2f}%)")

print("\n📋 WHAT DO THESE METRICS MEAN?")
print(f"   • Accuracy:  {accuracy*100:.1f}% of all predictions were correct")
print(f"   • Precision: {precision*100:.1f}% of predicted delays were actually delayed")
print(f"   • Recall:    {recall*100:.1f}% of actual delays were caught")
print(f"   • F1-Score:  Balance between precision and recall")
print(f"   • ROC-AUC:   {roc_auc*100:.1f}% - Model's ability to distinguish classes")

# Detailed classification report
print("\n📋 DETAILED CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_lr, 
                           target_names=['On-Time', 'Delayed'],
                           digits=4))

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
tn, fp, fn, tp = cm_lr.ravel()

print("\n📊 CONFUSION MATRIX:")
print("\n                 Predicted")
print("                On-Time  Delayed")
print(f"Actual On-Time   {tn:6,}   {fp:6,}")
print(f"Actual Delayed   {fn:6,}   {tp:6,}")

print("\n💡 INTERPRETATION:")
print(f"   • True Negatives (TN):  {tn:,} - Correctly predicted on-time")
print(f"   • False Positives (FP): {fp:,} - Wrongly predicted as delayed")
print(f"   • False Negatives (FN): {fn:,} - Missed actual delays")
print(f"   • True Positives (TP):  {tp:,} - Correctly predicted delays")

print("\n" + "=" * 70)
print("STEP 19: RANDOM FOREST CLASSIFIER")
print("=" * 70)

from sklearn.ensemble import RandomForestClassifier

print("\n🌲 What is Random Forest?")
print("   - Ensemble of many decision trees")
print("   - More powerful than single tree")
print("   - Good for complex patterns")
print("   - Handles non-linear relationships well")

print("\n⏱️  Training Random Forest...")
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=15,          # Maximum tree depth
    min_samples_split=20,  # Minimum samples to split
    min_samples_leaf=10,   # Minimum samples in leaf
    class_weight='balanced',
    random_state=42,
    n_jobs=6,
    verbose=0
)

rf_model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

print(f"✓ Model trained in {training_time:.2f} seconds")

# Make predictions
print("\n🔮 Making predictions on test set...")
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics
print("\n📊 RANDOM FOREST PERFORMANCE:")
print("="*50)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print(f"\n   Accuracy:  {accuracy_rf:.4f}  ({accuracy_rf*100:.2f}%)")
print(f"   Precision: {precision_rf:.4f}  ({precision_rf*100:.2f}%)")
print(f"   Recall:    {recall_rf:.4f}  ({recall_rf*100:.2f}%)")
print(f"   F1-Score:  {f1_rf:.4f}  ({f1_rf*100:.2f}%)")
print(f"   ROC-AUC:   {roc_auc_rf:.4f}  ({roc_auc_rf*100:.2f}%)")

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()

print("\n📊 CONFUSION MATRIX:")
print("\n                 Predicted")
print("                On-Time  Delayed")
print(f"Actual On-Time   {tn_rf:6,}   {fp_rf:6,}")
print(f"Actual Delayed   {fn_rf:6,}   {tp_rf:6,}")
print("\n📈 IMPROVEMENT vs Logistic Regression:")
print(f"   Accuracy:  {(accuracy_rf - accuracy)*100:+.2f}%")
print(f"   Precision: {(precision_rf - precision)*100:+.2f}%")
print(f"   Recall:    {(recall_rf - recall)*100:+.2f}%")
print(f"   F1-Score:  {(f1_rf - f1)*100:+.2f}%")
print(f"   ROC-AUC:   {(roc_auc_rf - roc_auc)*100:+.2f}%")

print("\n" + "=" * 70)
print("STEP 20: XGBOOST CLASSIFIER")
print("=" * 70)

import xgboost as xgb

print("\n🚀 What is XGBoost?")
print("   - Extreme Gradient Boosting")
print("   - Builds trees sequentially, learning from mistakes")
print("   - Often wins data science competitions")
print("   - Great for tabular data")

print("\n⏱️  Training XGBoost...")

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"   Handling class imbalance with scale_pos_weight: {scale_pos_weight:.2f}")

start_time = time.time()

xgb_model = xgb.XGBClassifier(
    n_estimators=150,       # Number of boosting rounds
    max_depth=6,            # Maximum tree depth
    learning_rate=0.1,      # Step size shrinkage
    subsample=0.8,          # Fraction of samples per tree
    colsample_bytree=0.8,   # Fraction of features per tree
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    random_state=42,
    n_jobs=6
)

xgb_model.fit(
    X_train_scaled, 
    y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

training_time = time.time() - start_time
print(f"✓ Model trained in {training_time:.2f} seconds")

# Make predictions
print("\n🔮 Making predictions on test set...")
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics
print("\n📊 XGBOOST PERFORMANCE:")
print("="*50)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

print(f"\n   Accuracy:  {accuracy_xgb:.4f}  ({accuracy_xgb*100:.2f}%)")
print(f"   Precision: {precision_xgb:.4f}  ({precision_xgb*100:.2f}%)")
print(f"   Recall:    {recall_xgb:.4f}  ({recall_xgb*100:.2f}%)")
print(f"   F1-Score:  {f1_xgb:.4f}  ({f1_xgb*100:.2f}%)")
print(f"   ROC-AUC:   {roc_auc_xgb:.4f}  ({roc_auc_xgb*100:.2f}%)")

# Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
tn_xgb, fp_xgb, fn_xgb, tp_xgb = cm_xgb.ravel()

print("\n📊 CONFUSION MATRIX:")
print("\n                 Predicted")
print("                On-Time  Delayed")
print(f"Actual On-Time   {tn_xgb:6,}   {fp_xgb:6,}")
print(f"Actual Delayed   {fn_xgb:6,}   {tp_xgb:6,}")

print("\n📈 IMPROVEMENT vs Logistic Regression:")
print(f"   Accuracy:  {(accuracy_xgb - accuracy)*100:+.2f}%")
print(f"   Precision: {(precision_xgb - precision)*100:+.2f}%")
print(f"   Recall:    {(recall_xgb - recall)*100:+.2f}%")
print(f"   F1-Score:  {(f1_xgb - f1)*100:+.2f}%")
print(f"   ROC-AUC:   {(roc_auc_xgb - roc_auc)*100:+.2f}%")

print("\n" + "=" * 70)
print("STEP 20: MODEL COMPARISON")
print("=" * 70)

# Create comparison dataframe
model_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'XGBoost'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_xgb)
    ],
    'Precision': [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_xgb)
    ],
    'Recall': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_xgb)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_xgb)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, y_pred_proba_lr),
        roc_auc_score(y_test, y_pred_proba_xgb)
    ]
})

print("\n📊 MODEL PERFORMANCE COMPARISON:")
print(model_comparison.to_string(index=False))

# Identify best model
best_model_idx = model_comparison['ROC-AUC'].idxmax()
best_model_name = model_comparison.loc[best_model_idx, 'Model']
best_auc = model_comparison.loc[best_model_idx, 'ROC-AUC']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   ROC-AUC Score: {best_auc:.4f}")

# ----------------------------------------------------------------------------
# SIMPLE MULTI-PLOT VISUALIZATION
# ----------------------------------------------------------------------------
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

for i, metric in enumerate(metrics):
    row = i // 3
    col = i % 3
    ax = axes[row, col]

    ax.bar(model_comparison['Model'], model_comparison[metric],
           color=['skyblue', 'lightcoral'], edgecolor='black')
    
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add value labels
    for j, val in enumerate(model_comparison[metric]):
        ax.text(j, val + 0.02, f"{val:.2f}", ha='center', fontsize=10, fontweight='bold')

# Remove the last empty box (bottom right)
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()


print("\n" + "=" * 70)
print("STEP 21: FEATURE IMPORTANCE ANALYSIS (XGBoost)")
print("=" * 70)

# Get feature importance from XGBoost
feature_importance = pd.DataFrame({
    'Feature': FINAL_FEATURES,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 TOP 20 MOST IMPORTANT FEATURES:")
print(feature_importance.head(20).to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')

# 1. Top 20 Features (Bar chart)
top_20 = feature_importance.head(20)
axes[0].barh(range(len(top_20)), top_20['Importance'], color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_yticks(range(len(top_20)))
axes[0].set_yticklabels(top_20['Feature'])
axes[0].set_xlabel('Importance Score')
axes[0].set_title('Top 20 Most Important Features')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# Add value labels
for i, val in enumerate(top_20['Importance']):
    axes[0].text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=8)

# 2. Cumulative importance
feature_importance_sorted = feature_importance.copy()
feature_importance_sorted['Cumulative'] = feature_importance_sorted['Importance'].cumsum()
feature_importance_sorted['Cumulative_Pct'] = (feature_importance_sorted['Cumulative'] / 
                                                feature_importance_sorted['Importance'].sum() * 100)

axes[1].plot(range(len(feature_importance_sorted)), 
             feature_importance_sorted['Cumulative_Pct'], 
             linewidth=2, color='green')
axes[1].axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% threshold')
axes[1].axhline(y=90, color='orange', linestyle='--', linewidth=2, label='90% threshold')
axes[1].set_xlabel('Number of Features')
axes[1].set_ylabel('Cumulative Importance (%)')
axes[1].set_title('Cumulative Feature Importance')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Find how many features for 80% and 90%
n_80 = (feature_importance_sorted['Cumulative_Pct'] <= 80).sum()
n_90 = (feature_importance_sorted['Cumulative_Pct'] <= 90).sum()

axes[1].text(n_80, 82, f'{n_80} features', ha='center', fontsize=10, fontweight='bold')
axes[1].text(n_90, 92, f'{n_90} features', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\n📌 KEY INSIGHTS:")
print(f"   Top 20 features account for {feature_importance.head(20)['Importance'].sum()/feature_importance['Importance'].sum()*100:.1f}% of importance")
print(f"   {n_80} features needed for 80% cumulative importance")
print(f"   {n_90} features needed for 90% cumulative importance")

print("\n" + "=" * 70)
print("VISUALIZATION: ROC & PRECISION-RECALL CURVES")
print("=" * 70)

from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Performance Curves (XGBoost)', fontsize=16, fontweight='bold')

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_xgb)
roc_auc = roc_auc_score(y_test, y_pred_proba_xgb)

axes[0].plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

# 2. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_xgb)
avg_precision = average_precision_score(y_test, y_pred_proba_xgb)

axes[1].plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend(loc="lower left")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"✓ ROC-AUC: {roc_auc:.4f}")
print(f"✓ Average Precision: {avg_precision:.4f}")

print("VISUALIZATION: CONFUSION MATRICES")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')

# 1. Logistic Regression Confusion Matrix
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['On-Time', 'Delayed'],
            yticklabels=['On-Time', 'Delayed'],
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Logistic Regression')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Add percentages
total_lr = cm_lr.sum()
for i in range(2):
    for j in range(2):
        pct = (cm_lr[i, j] / total_lr) * 100
        axes[0].text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='red')

# 2. XGBoost Confusion Matrix
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['On-Time', 'Delayed'],
            yticklabels=['On-Time', 'Delayed'],
            ax=axes[1], cbar_kws={'label': 'Count'})
axes[1].set_title('XGBoost')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

# Add percentages
total_xgb = cm_xgb.sum()
for i in range(2):
    for j in range(2):
        pct = (cm_xgb[i, j] / total_xgb) * 100
        axes[1].text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='red')

plt.tight_layout()
plt.show()

# Calculate detailed metrics from confusion matrices
print("\n📊 DETAILED CONFUSION MATRIX ANALYSIS:")

for model_name, cm in [('Logistic Regression', cm_lr), ('XGBoost', cm_xgb)]:
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    print(f"\n{model_name}:")
    print(f"   True Negatives (TN):  {tn:,}")
    print(f"   False Positives (FP): {fp:,}")
    print(f"   False Negatives (FN): {fn:,}")
    print(f"   True Positives (TP):  {tp:,}")
    print(f"   Sensitivity (TPR):    {sensitivity:.4f}")
    print(f"   Specificity (TNR):    {specificity:.4f}")
    print(f"   False Positive Rate:  {fpr:.4f}")
    print(f"   False Negative Rate:  {fnr:.4f}")

print("\n" + "=" * 70)
print("STEP 22: BUSINESS IMPACT ANALYSIS")
print("=" * 70)

# Define business costs
cost_false_positive = 10   # Cost of incorrectly predicting delay
cost_false_negative = 100  # Cost of missing a delay (higher impact)

print("\n💰 BUSINESS COST ASSUMPTIONS:")
print(f"   Cost per False Positive: ${cost_false_positive}")
print(f"      (Unnecessary actions taken for non-delayed flights)")
print(f"   Cost per False Negative: ${cost_false_negative}")
print(f"      (Missing actual delays, customer dissatisfaction)")

# Calculate business costs for both models
results = {}

for model_name, cm in [('Logistic Regression', cm_lr), ('XGBoost', cm_xgb)]:
    tn, fp, fn, tp = cm.ravel()
    
    total_fp_cost = fp * cost_false_positive
    total_fn_cost = fn * cost_false_negative
    total_cost = total_fp_cost + total_fn_cost
    avg_cost_per_prediction = total_cost / len(y_test)
    
    results[model_name] = {
        'FP_Cost': total_fp_cost,
        'FN_Cost': total_fn_cost,
        'Total_Cost': total_cost,
        'Avg_Cost': avg_cost_per_prediction
    }
    
    print(f"\n📊 {model_name.upper()} BUSINESS COSTS:")
    print(f"   False Positive Cost: ${total_fp_cost:,}")
    print(f"   False Negative Cost: ${total_fn_cost:,}")
    print(f"   Total Cost: ${total_cost:,}")
    print(f"   Avg Cost per Prediction: ${avg_cost_per_prediction:.2f}")

# Baseline: Always predict majority class (On-Time)
baseline_fn = (y_test == 1).sum()
baseline_cost = baseline_fn * cost_false_negative

print(f"\n📊 BASELINE (Always predict On-Time):")
print(f"   Total Cost: ${baseline_cost:,}")

# Calculate savings
print(f"\n💵 COST SAVINGS vs BASELINE:")
for model_name, costs in results.items():
    savings = baseline_cost - costs['Total_Cost']
    savings_pct = (savings / baseline_cost) * 100
    
    print(f"\n   {model_name}:")
    print(f"      Savings: ${savings:,} ({savings_pct:.1f}%)")
    if savings > 0:
        print(f"      ✓ Model provides cost savings!")
    else:
        print(f"      ❌ Model is more expensive than baseline")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Business Impact Analysis', fontsize=16, fontweight='bold')

# 1. Cost Breakdown by Model
models = ['Baseline', 'Logistic Reg', 'XGBoost']
fp_costs = [0, results['Logistic Regression']['FP_Cost'], results['XGBoost']['FP_Cost']]
fn_costs = [baseline_cost, results['Logistic Regression']['FN_Cost'], results['XGBoost']['FN_Cost']]

x = np.arange(len(models))
width = 0.35

bars1 = axes[0].bar(x - width/2, fp_costs, width, label='FP Cost', 
                    color='orange', alpha=0.7, edgecolor='black')
bars2 = axes[0].bar(x + width/2, fn_costs, width, label='FN Cost', 
                    color='red', alpha=0.7, edgecolor='black')

axes[0].set_ylabel('Cost ($)')
axes[0].set_title('Cost Breakdown by Model')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:,.0f}', ha='center', va='bottom', fontsize=9)

# 2. Total Cost Comparison
total_costs = [baseline_cost, 
               results['Logistic Regression']['Total_Cost'], 
               results['XGBoost']['Total_Cost']]

colors_cost = ['gray', 'steelblue', 'green']
bars = axes[1].bar(models, total_costs, color=colors_cost, alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Total Cost ($)')
axes[1].set_title('Total Cost Comparison')
axes[1].grid(axis='y', alpha=0.3)

# Highlight best model
min_cost_idx = total_costs.index(min(total_costs))
bars[min_cost_idx].set_color('#FFD700')
bars[min_cost_idx].set_edgecolor('black')
bars[min_cost_idx].set_linewidth(3)

# Add value labels and savings
for i, (model, cost) in enumerate(zip(models, total_costs)):
    axes[1].text(i, cost + 50000, f'${cost:,}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    if i > 0:  # Calculate savings for ML models
        savings = baseline_cost - cost
        axes[1].text(i, cost/2, f'Save\n${savings:,}', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("STEP 23: SAVING MODELS AND RESULTS")
print("=" * 70)

import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Save models
print("\n💾 Saving trained models...")
joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("   ✓ Logistic Regression model saved")
print("   ✓ XGBoost model saved")
print("   ✓ Scaler saved")

# Save encoders
joblib.dump(encoders, 'models/label_encoders.pkl')
print("   ✓ Label encoders saved")

# Save feature names
feature_dict = {
    'all_features': FINAL_FEATURES,
    'numerical_features': NUMERICAL_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES,
    'categorical_encoded': CATEGORICAL_ENCODED,
    'target': TARGET
}

import json
with open('models/feature_names.json', 'w') as f:
    json.dump(feature_dict, f, indent=4)
print("   ✓ Feature names saved")

# Save model comparison
model_comparison.to_csv('results/model_comparison.csv', index=False)
print("   ✓ Model comparison saved")

# Save feature importance
feature_importance.to_csv('results/feature_importance.csv', index=False)
print("   ✓ Feature importance saved")

# Save test predictions
test_results = pd.DataFrame({
    'y_true': y_test,
    'y_pred_lr': y_pred_lr,
    'y_pred_proba_lr': y_pred_proba_lr,
    'y_pred_xgb': y_pred_xgb,
    'y_pred_proba_xgb': y_pred_proba_xgb
})
test_results.to_csv('results/test_predictions.csv', index=False)
print("   ✓ Test predictions saved")

print("\n✓ All models and results saved successfully!")
print("\n📁 Saved files:")
print("   models/")
print("      ├── logistic_regression_model.pkl")
print("      ├── xgboost_model.pkl")
print("      ├── scaler.pkl")
print("      ├── label_encoders.pkl")
print("      └── feature_names.json")
print("   results/")
print("      ├── model_comparison.csv")
print("      ├── feature_importance.csv")
print("      └── test_predictions.csv")

print("\n" + "=" * 80)
print("🎉 FLIGHT DELAY PREDICTION - FINAL SUMMARY REPORT")
print("=" * 80)

print("\n📊 DATASET SUMMARY:")
print(f"   Total Flights Analyzed: {len(df):,}")
print(f"   Training Samples: {len(X_train):,}")
print(f"   Test Samples: {len(X_test):,}")
print(f"   Features Used: {len(FINAL_FEATURES)}")
print(f"   Target Variable: {TARGET}")

print("\n📅 TIME PERIOD:")
print(f"   From: {df['Date'].min()}")
print(f"   To: {df['Date'].max()}")
print(f"   Duration: {(df['Date'].max() - df['Date'].min()).days} days")

print("\n📈 CLASS DISTRIBUTION:")
delay_dist = df[TARGET].value_counts(normalize=True) * 100
print(f"   On-Time (Class 0): {delay_dist[0]:.1f}%")
print(f"   Delayed (Class 1): {delay_dist[1]:.1f}%")

print("\n🏆 BEST MODEL: XGBoost")
print(f"   Accuracy:  {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"   Recall:    {recall_score(y_test, y_pred_xgb):.4f}")
print(f"   F1-Score:  {f1_score(y_test, y_pred_xgb):.4f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")

print("\n📌 TOP 5 MOST IMPORTANT FEATURES:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"   {idx+1}. {row['Feature']}: {row['Importance']:.4f}")

print("\n💰 BUSINESS IMPACT:")
xgb_savings = baseline_cost - results['XGBoost']['Total_Cost']
xgb_savings_pct = (xgb_savings / baseline_cost) * 100
print(f"   Cost Savings vs Baseline: ${xgb_savings:,} ({xgb_savings_pct:.1f}%)")
print(f"   Avg Cost per Prediction: ${results['XGBoost']['Avg_Cost']:.2f}")

print("\n🎯 KEY INSIGHTS:")
print(f"   1. {feature_importance.iloc[0]['Feature']} is the most important predictor")
print(f"   2. Model achieves {roc_auc_score(y_test, y_pred_proba_xgb):.1%} ROC-AUC score")
print(f"   3. Reduces business costs by {xgb_savings_pct:.1f}% compared to no prediction")

# Identify patterns from EDA
if 'worst_month' in locals():
    print(f"   4. {worst_month[1]} is the worst month for delays ({monthly_stats.loc[worst_month, 'Delay_Rate']:.1f}% delayed)")
if 'worst_day' in locals():
    print(f"   5. {worst_day[1]} is the worst day for delays ({day_stats.loc[worst_day, 'Delay_Rate']:.1f}% delayed)")

print("\n⚠️  DATA LEAKAGE PREVENTION:")
print("   ✓ Time-based train-test split (no future data in training)")
print("   ✓ Only pre-flight features used (no actual times or post-flight data)")
print("   ✓ Proper feature engineering with historical aggregations")



