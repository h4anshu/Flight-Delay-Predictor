import warnings
warnings.filterwarnings("ignore")
import joblib

# Load old pickle model (suppressing the warning)
xgb_model = joblib.load('models/xgboost_model.pkl')

# Re-save using XGBoost's native JSON format (version-safe)
xgb_model.save_model('models/xgboost_model.json')
print("Re-saved XGBoost model as models/xgboost_model.json")
