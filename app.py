import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import datetime

# Page Config
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

# Simple Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    header { visibility: hidden; }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 1200px;
    }

    h1, h2, h3 { color: #f8fafc !important; }

    .glass {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 16px;
    }

    .title-gradient {
        background: linear-gradient(90deg, #3b82f6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    .big-number {
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
        margin: 8px 0;
    }

    .label-text {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
    }

    .green { color: #10b981; }
    .red { color: #ef4444; }
    .amber { color: #f59e0b; }
    .blue { color: #38bdf8; }

    .insight-item {
        padding: 6px 0;
        font-size: 0.95rem;
    }

    .risk-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(255,255,255,0.1);
        margin: 6px 0 12px 0;
        overflow: hidden;
    }
    .risk-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    .model-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #6366f1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(59,130,246,0.3);
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('models/xgboost_model.json')
    lr_model = joblib.load('models/logistic_regression_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoders = joblib.load('models/label_encoders.pkl')
    with open('models/feature_names.json', 'r') as f:
        features_dict = json.load(f)
    with open('models/lookup_stats.json', 'r') as f:
        lookup_stats = json.load(f)
    try:
        model_comparison = pd.read_csv('results/model_comparison.csv')
    except FileNotFoundError:
        model_comparison = None
    return xgb_model, lr_model, scaler, encoders, features_dict, lookup_stats, model_comparison

try:
    xgb_model, lr_model, scaler, encoders, features_dict, lookup_stats, model_comparison = load_assets()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

ui = lookup_stats['ui_options']

# ─── Helper Functions ───────────────────────────────────────────────────────
def get_time_block(hour):
    if 0 <= hour < 6: return 'Night (00-06)'
    elif 6 <= hour < 12: return 'Morning (06-12)'
    elif 12 <= hour < 18: return 'Afternoon (12-18)'
    else: return 'Evening (18-24)'

def build_features(date, arr_hour, carrier, origin, dest, distance, elapsed):
    route = f"{origin}-{dest}"
    c = lookup_stats['carriers'].get(carrier, lookup_stats['defaults'])
    o = lookup_stats['origins'].get(origin, lookup_stats['defaults'])
    d = lookup_stats['dests'].get(dest, lookup_stats['defaults'])
    dfl = lookup_stats['defaults']
    rf = lookup_stats['routes'].get(route, dfl['Route_Frequency'])
    speed = distance / (elapsed / 60) if elapsed > 0 else 400

    features = {
        'DayOfWeek': date.isoweekday(),
        'Month': date.month,
        'Day': date.day,
        'Year': date.year,
        'IsWeekend': 1 if date.isoweekday() in [6, 7] else 0,
        'CRSArr_hour': arr_hour,
        'CRSElapsedTime': elapsed,
        'Distance': distance,
        'Carrier_DelayRate': c.get('Carrier_DelayRate', dfl['Carrier_DelayRate']),
        'Carrier_AvgDelay': c.get('Carrier_AvgDelay', dfl['Carrier_AvgDelay']),
        'Origin_DelayRate': o.get('Origin_DelayRate', dfl['Origin_DelayRate']),
        'Origin_AvgDelay': o.get('Origin_AvgDelay', dfl['Origin_AvgDelay']),
        'Dest_DelayRate': d.get('Dest_DelayRate', dfl['Dest_DelayRate']),
        'Dest_AvgDelay': d.get('Dest_AvgDelay', dfl['Dest_AvgDelay']),
        'IsHolidaySeason': 1 if date.month in [12, 1] else 0,
        'IsSummer': 1 if date.month in [6, 7, 8] else 0,
        'IsRushHour': 1 if arr_hour in [7, 8, 17, 18] else 0,
        'IsLateNight': 1 if arr_hour in [22, 23, 0, 1, 2, 3, 4, 5] else 0,
        'Avg_Speed': speed if 0 < speed < 1000 else 400,
        'IsShortFlight': 1 if distance < 500 else 0,
        'IsLongFlight': 1 if distance > 1500 else 0,
        'Route_Frequency': rf,
        'IsPopularRoute': 1 if rf > lookup_stats['median_route_frequency'] else 0,
    }

    cat_raw = {
        'UniqueCarrier': carrier,
        'Origin': origin,
        'Dest': dest,
        'TimeBlock': get_time_block(arr_hour)
    }
    return features, cat_raw

def run_prediction(features, cat_raw, model):
    df = pd.DataFrame([features])
    for col in features_dict['categorical_features']:
        le = encoders[col]
        val = str(cat_raw[col])
        df[f"{col}_Encoded"] = le.transform([val])[0] if val in le.classes_ else 0
    df = df[features_dict['all_features']]
    df_scaled = scaler.transform(df)
    prob = model.predict_proba(df_scaled)[0, 1]
    pred = model.predict(df_scaled)[0]
    return pred, prob, df.iloc[0].to_dict()

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("## ✈️ <span class='title-gradient'>FlightIQ</span> Dashboard", unsafe_allow_html=True)
st.markdown("<span style='color:#94a3b8;'>Predict flight delays using XGBoost & Logistic Regression · 27 features · Trained on 688K real US flights (BTS, full year 2024)</span>", unsafe_allow_html=True)
st.markdown("---")

# ─── INPUT FORM ─────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6, c7 = st.columns([1.2, 1, 1.5, 1.5, 1.5, 1, 1])

with c1:
    date_input = st.date_input("📅 Flight Date", min_value=datetime.date(2020, 1, 1))
with c2:
    arr_time = st.time_input("🕐 Arrival Time", datetime.time(14, 30))
with c3:
    carrier_code = st.selectbox("🛩️ Airline", ui['carriers'],
        format_func=lambda x: f"{x} – {ui['airline_names'].get(x, '')[:25]}")
with c4:
    origin = st.selectbox("🛫 Origin", ui['origins'],
        index=ui['origins'].index('JFK') if 'JFK' in ui['origins'] else 0,
        format_func=lambda x: f"{x} – {ui['airport_names'].get(x, '')[:20]}")
with c5:
    dest = st.selectbox("🛬 Destination", ui['dests'],
        index=ui['dests'].index('LAX') if 'LAX' in ui['dests'] else 1,
        format_func=lambda x: f"{x} – {ui['airport_names'].get(x, '')[:20]}")
with c6:
    distance = st.number_input("📏 Distance (mi)", 10, 8000, 2500)
with c7:
    elapsed = st.number_input("⏱️ Duration (min)", 30, 900, 300)

st.markdown("")
predict_btn = st.button("🔍  PREDICT DELAY", width='stretch')

# ─── RESULTS ────────────────────────────────────────────────────────────────
if predict_btn:
    if origin == dest:
        st.error("Origin and Destination cannot be the same airport.")
        st.stop()

    features, cat_raw = build_features(date_input, arr_time.hour, carrier_code, origin, dest, distance, elapsed)

    # Run both models
    xgb_pred, xgb_prob, final = run_prediction(features, cat_raw, xgb_model)
    lr_pred, lr_prob, _ = run_prediction(features, cat_raw, lr_model)

    st.markdown("---")

    left, right = st.columns([1, 1.5])

    # ── Left: Main Prediction (XGBoost) ──
    # Thresholds are calibrated to this model's actual predicted-probability
    # distribution on the real test set (median ~36%, 75th pct ~46%), not to
    # raw 50%/round numbers - the model never outputs anything close to 0% or
    # 100% since the true base delay rate is only ~16%.
    with left:
        delay_prob = xgb_prob * 100

        if delay_prob >= 45:
            risk_label, risk_color, risk_cls = "HIGH RISK", "#ef4444", "red"
        elif delay_prob >= 30:
            risk_label, risk_color, risk_cls = "MEDIUM RISK", "#f59e0b", "amber"
        else:
            risk_label, risk_color, risk_cls = "LOW RISK", "#10b981", "green"

        st.markdown(f"""
        <div class='glass' style='text-align:center;'>
            <div class='label-text'>XGBoost Prediction</div>
            <h2 class='{risk_cls}' style='margin:4px 0;'>{risk_label}</h2>
            <div class='big-number {risk_cls}'>{delay_prob:.1f}%</div>
            <p style='color:#94a3b8; margin:4px 0 12px 0;'>chance of delay (>15 min)</p>
            <div class='risk-bar'>
                <div class='risk-fill' style='width:{delay_prob}%; background:{risk_color};'></div>
            </div>
            <div style='display:flex; justify-content:space-between; font-size:0.75rem; color:#64748b;'>
                <span>On-Time</span><span>Delayed</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Logistic Regression comparison
        lr_delay = lr_prob * 100
        lr_color = "#ef4444" if lr_delay >= 45 else "#10b981"
        st.markdown(f"""
        <div class='glass' style='padding:14px 24px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <div class='label-text'>Logistic Regression</div>
                    <span style='font-size:1.2rem; font-weight:700; color:{lr_color};'>{lr_delay:.1f}% delay chance</span>
                </div>
                <div class='model-tag' style='background:rgba(255,255,255,0.08); color:#94a3b8;'>Baseline</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Agreement indicator - both models landing on the same side of the
        # decision boundary is a real (if rough) signal of confidence
        agree = (xgb_prob >= 0.45) == (lr_prob >= 0.45)
        agree_html = (
            "<span style='color:#10b981;'>✅ Both models agree</span>" if agree
            else "<span style='color:#f59e0b;'>⚠️ Models disagree — treat this one with extra caution</span>"
        )
        st.markdown(f"<div style='padding:4px 4px; font-size:0.85rem;'>{agree_html}</div>", unsafe_allow_html=True)

        # Honest performance disclosure - this model has real, modest skill
        # (ROC-AUC ~0.62), not "solved" accuracy. Say so plainly.
        with st.expander("ℹ️ About this model's accuracy"):
            if model_comparison is not None:
                st.dataframe(model_comparison.set_index('Model').style.format("{:.1%}"), width='stretch')
            st.markdown(
                "<span style='color:#94a3b8; font-size:0.85rem;'>"
                "Trained on pre-flight information only (no live weather/ATC feeds), so it "
                "cannot catch delays caused by conditions unknown until the day of travel. "
                "Treat the percentage as a relative risk signal for planning, not a precise forecast."
                "</span>", unsafe_allow_html=True)

    # ── Right: Risk Factors ──
    with right:
        cr = final['Carrier_DelayRate'] * 100
        orr = final['Origin_DelayRate'] * 100
        dr = final['Dest_DelayRate'] * 100

        # Real carrier/airport delay rates in this data run ~14-28% (median
        # ~20%), not the 0-100% spread these thresholds implied before.
        def rate_color(v):
            return "red" if v > 24 else ("amber" if v > 18 else "green")

        # Build insights
        flags = []
        if final['IsHolidaySeason']: flags.append("⚠️ <b>Holiday Season</b> — Peak travel traffic")
        if final['IsSummer']: flags.append("☀️ <b>Summer</b> — Higher volume & weather risk")
        if final['IsRushHour']: flags.append("🚦 <b>Rush Hour</b> — Peak airport congestion")
        if final['IsLateNight']: flags.append("🌙 <b>Late Night</b> — Fewer delays typical")
        if final['IsPopularRoute']: flags.append("🔀 <b>Popular Route</b> — High traffic")
        if final['IsLongFlight']: flags.append("🌐 <b>Long Flight</b> — Greater distance, more variables")
        if final['IsShortFlight']: flags.append("📍 <b>Short Flight</b> — Regional hop")
        if not flags: flags.append("✅ No major risk factors detected")

        insight_html = ""
        for f in flags:
            insight_html += f"<div class='insight-item'>{f}</div>"

        st.markdown(f"""
        <div class='glass'>
            <h3 style='margin-top:0;'>🔍 Risk Factors</h3>
            <div style='display:flex; gap:20px; margin:16px 0;'>
                <div style='flex:1; text-align:center;'>
                    <div class='label-text'>Airline Delay Rate</div>
                    <div class='stat-value {rate_color(cr)}'>{cr:.1f}%</div>
                </div>
                <div style='flex:1; text-align:center;'>
                    <div class='label-text'>Origin Delay Rate</div>
                    <div class='stat-value {rate_color(orr)}'>{orr:.1f}%</div>
                </div>
                <div style='flex:1; text-align:center;'>
                    <div class='label-text'>Dest Delay Rate</div>
                    <div class='stat-value {rate_color(dr)}'>{dr:.1f}%</div>
                </div>
            </div>
            <hr style='border-color:rgba(255,255,255,0.08); margin:12px 0;'>
            {insight_html}
        </div>
        """, unsafe_allow_html=True)

        # Flight summary
        route = f"{origin} → {dest}"
        day_name = date_input.strftime('%A')
        time_block = get_time_block(arr_time.hour)

        st.markdown(f"""
        <div class='glass' style='padding:16px 24px;'>
            <div style='display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px;'>
                <div><span class='label-text'>Route</span><br><b>{route}</b></div>
                <div><span class='label-text'>Day</span><br><b>{day_name}</b></div>
                <div><span class='label-text'>Time Block</span><br><b>{time_block}</b></div>
                <div><span class='label-text'>Distance</span><br><b>{distance} mi</b></div>
                <div><span class='label-text'>Duration</span><br><b>{elapsed} min</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
