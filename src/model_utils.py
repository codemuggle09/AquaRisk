from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_model(name):
    name = name.lower()
    if name in ["logistic", "logisticregression"]:
        return LogisticRegression(max_iter=1000)
    elif name in ["rf", "randomforest", "randomforestclassifier"]:
        return RandomForestClassifier(n_estimators=300, random_state=42)
    elif name in ["xgboost", "xgb"]:
        return XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
    elif name in ["lightgbm", "lgbm"]:
        return LGBMClassifier()
    elif name in ["svm", "svc"]:
        return SVC(probability=True)
    elif name in ["linearregression"]:
        return LinearRegression()
    elif name in ["rf_regressor", "randomforestregressor"]:
        return RandomForestRegressor(n_estimators=300, random_state=42)
    elif name in ["svr"]:
        return SVR(kernel='rbf')
    else:
        raise ValueError(f"Unknown model name: {name}")
