import json
import pandas as pd
import numpy as np

print("=" * 70)
print("FULL PROJECT AUDIT")
print("=" * 70)

# 1. Check lookup stats
with open('models/lookup_stats.json', 'r') as f:
    d = json.load(f)

print("\n--- CARRIER DELAY RATES (sample) ---")
for k, v in list(d['carriers'].items())[:10]:
    print(f"  {k}: DelayRate={v['Carrier_DelayRate']:.4f}  AvgDelay={v['Carrier_AvgDelay']:.1f} min")

print("\n--- ORIGIN DELAY RATES (sample) ---")
for k, v in list(d['origins'].items())[:10]:
    print(f"  {k}: DelayRate={v['Origin_DelayRate']:.4f}  AvgDelay={v['Origin_AvgDelay']:.1f} min")

print("\n--- DEST DELAY RATES (sample) ---")
for k, v in list(d['dests'].items())[:10]:
    print(f"  {k}: DelayRate={v['Dest_DelayRate']:.4f}  AvgDelay={v['Dest_AvgDelay']:.1f} min")

print("\n--- DEFAULTS ---")
print(d['defaults'])

print("\n--- ROUTE FREQUENCY STATS ---")
routes = d['routes']
freqs = list(routes.values())
print(f"  Total routes: {len(routes)}")
print(f"  Median freq: {d['median_route_frequency']}")
print(f"  Min freq: {min(freqs)}, Max freq: {max(freqs)}")

# 2. Check test predictions
print("\n\n--- TEST PREDICTIONS ANALYSIS ---")
preds = pd.read_csv('results/test_predictions.csv')
print(f"  Total test samples: {len(preds)}")
print(f"\n  Actual distribution:")
print(f"    On-Time (0): {(preds['y_true']==0).sum()} ({(preds['y_true']==0).mean()*100:.1f}%)")
print(f"    Delayed (1): {(preds['y_true']==1).sum()} ({(preds['y_true']==1).mean()*100:.1f}%)")

print(f"\n  LR Prediction distribution:")
print(f"    Predicted On-Time: {(preds['y_pred_lr']==0).sum()} ({(preds['y_pred_lr']==0).mean()*100:.1f}%)")
print(f"    Predicted Delayed: {(preds['y_pred_lr']==1).sum()} ({(preds['y_pred_lr']==1).mean()*100:.1f}%)")

print(f"\n  XGBoost Prediction distribution:")
print(f"    Predicted On-Time: {(preds['y_pred_xgb']==0).sum()} ({(preds['y_pred_xgb']==0).mean()*100:.1f}%)")
print(f"    Predicted Delayed: {(preds['y_pred_xgb']==1).sum()} ({(preds['y_pred_xgb']==1).mean()*100:.1f}%)")

print(f"\n  LR Probability stats:")
print(f"    Mean: {preds['y_pred_proba_lr'].mean():.4f}")
print(f"    Median: {preds['y_pred_proba_lr'].median():.4f}")
print(f"    Min: {preds['y_pred_proba_lr'].min():.4f}")
print(f"    Max: {preds['y_pred_proba_lr'].max():.4f}")
print(f"    Std: {preds['y_pred_proba_lr'].std():.4f}")

print(f"\n  XGBoost Probability stats:")
print(f"    Mean: {preds['y_pred_proba_xgb'].mean():.4f}")
print(f"    Median: {preds['y_pred_proba_xgb'].median():.4f}")
print(f"    Min: {preds['y_pred_proba_xgb'].min():.4f}")
print(f"    Max: {preds['y_pred_proba_xgb'].max():.4f}")
print(f"    Std: {preds['y_pred_proba_xgb'].std():.4f}")

# Distribution of XGBoost probabilities
print(f"\n  XGBoost probability distribution:")
bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
labels = ['<30%', '30-40%', '40-50%', '50-60%', '60-70%', '>70%']
preds['prob_bin'] = pd.cut(preds['y_pred_proba_xgb'], bins=bins, labels=labels)
print(preds['prob_bin'].value_counts().sort_index().to_string())

# 3. Check model comparison
print("\n\n--- MODEL COMPARISON ---")
comp = pd.read_csv('results/model_comparison.csv')
print(comp.to_string(index=False))

# 4. Check feature importance
print("\n\n--- TOP 10 FEATURE IMPORTANCE ---")
fi = pd.read_csv('results/feature_importance.csv')
print(fi.head(10).to_string(index=False))

# 5. Quick test: run a manual prediction
print("\n\n--- MANUAL PREDICTION TEST ---")
import joblib
import xgboost as xgb

xgb_model = xgb.XGBClassifier()
xgb_model.load_model('models/xgboost_model.json')
lr_model = joblib.load('models/logistic_regression_model.pkl')
scaler = joblib.load('models/scaler.pkl')
encoders = joblib.load('models/label_encoders.pkl')

with open('models/feature_names.json', 'r') as f:
    feat_dict = json.load(f)

# Test case 1: Short flight, morning, small carrier
test1 = {
    'DayOfWeek': 2, 'Month': 3, 'Day': 15, 'Year': 2025, 'IsWeekend': 0,
    'CRSArr_hour': 10, 'CRSElapsedTime': 120, 'Distance': 400,
    'Carrier_DelayRate': 0.3, 'Carrier_AvgDelay': 5.0,
    'Origin_DelayRate': 0.25, 'Origin_AvgDelay': 4.0,
    'Dest_DelayRate': 0.28, 'Dest_AvgDelay': 4.5,
    'IsHolidaySeason': 0, 'IsSummer': 0, 'IsRushHour': 0, 'IsLateNight': 0,
    'Avg_Speed': 200, 'IsShortFlight': 1, 'IsLongFlight': 0,
    'Route_Frequency': 500, 'IsPopularRoute': 1,
}

# Test case 2: Long flight, rush hour, busy airport
test2 = {
    'DayOfWeek': 5, 'Month': 7, 'Day': 4, 'Year': 2025, 'IsWeekend': 0,
    'CRSArr_hour': 17, 'CRSElapsedTime': 360, 'Distance': 2500,
    'Carrier_DelayRate': 0.65, 'Carrier_AvgDelay': 20.0,
    'Origin_DelayRate': 0.60, 'Origin_AvgDelay': 18.0,
    'Dest_DelayRate': 0.55, 'Dest_AvgDelay': 15.0,
    'IsHolidaySeason': 0, 'IsSummer': 1, 'IsRushHour': 1, 'IsLateNight': 0,
    'Avg_Speed': 420, 'IsShortFlight': 0, 'IsLongFlight': 1,
    'Route_Frequency': 2000, 'IsPopularRoute': 1,
}

for name, test in [("LOW RISK (short, morning, good stats)", test1), 
                    ("HIGH RISK (long, rush hour, bad stats)", test2)]:
    # Add encoded categoricals
    test['UniqueCarrier_Encoded'] = 0
    test['Origin_Encoded'] = 0
    test['Dest_Encoded'] = 0
    test['TimeBlock_Encoded'] = 0
    
    df = pd.DataFrame([test])
    df = df[feat_dict['all_features']]
    df_scaled = scaler.transform(df)
    
    xgb_prob = xgb_model.predict_proba(df_scaled)[0, 1]
    lr_prob = lr_model.predict_proba(df_scaled)[0, 1]
    
    print(f"\n  {name}:")
    print(f"    XGBoost delay prob: {xgb_prob:.4f} ({xgb_prob*100:.1f}%)")
    print(f"    LR delay prob: {lr_prob:.4f} ({lr_prob*100:.1f}%)")
