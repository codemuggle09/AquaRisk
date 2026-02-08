from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import numpy as np
import pandas as pd

def train_regression_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RF_Regressor": RandomForestRegressor(n_estimators=300, random_state=42),
        "SVR": SVR(kernel='rbf')
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        results[name] = {"R2": r2, "RMSE": rmse}
        print(f"{name}: R² = {r2:.3f}, RMSE = {rmse:.3f}")

    return results
