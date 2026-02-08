import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

# Optional imports
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except ImportError:
    Sequential = None


# === Helper: compute sample weights for training ===
def get_sample_weights(y_train):
    """Return per-sample weights balanced by class frequencies."""
    return compute_sample_weight(class_weight='balanced', y=y_train)


# === Directory for saving models ===
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# === Build Deep Neural Network (Keras) ===
def build_dnn(input_dim, output_dim):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# === Define All Models (with balanced weighting) ===
def get_models():
    models = {
        "LR": LogisticRegression(max_iter=2000, solver="lbfgs", class_weight='balanced'),
        "SVM": SVC(kernel="rbf", probability=True, class_weight='balanced'),
        "RF": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight='balanced'
        ),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=10,
            verbose=False
        ),

        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, class_weight='balanced'),
            n_estimators=150,
            random_state=42
        ),
    }

    if XGBClassifier:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, use_label_encoder=False, eval_metric='mlogloss'
        )

    if LGBMClassifier:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=42, class_weight='balanced'
        )

    return models


# === Train and Save Models ===
def train_and_save_models(X, y, scaler=None, encoder=None, model_dir="models"):
    """
    Trains multiple classifiers, saves models and reports accuracy.
    """

    os.makedirs(model_dir, exist_ok=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Compute per-sample weights to handle imbalance
    sample_weights = get_sample_weights(y_train)

    models = get_models()
    results = {}

    # === Train each model ===
    for name, model in models.items():
        print(f"\n🚀 Training: {name}")

        # Fit with sample weights if supported
        try:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        except TypeError:
            model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc

        # Save trained model
        joblib.dump(model, os.path.join(model_dir, f"{name}.pkl"))
        print(f"{name} Accuracy: {acc:.4f}")

    # === DNN (Keras) ===
    if Sequential:
        print("\n🚀 Training: DNN")
        num_classes = len(np.unique(y))
        dnn = build_dnn(X_train.shape[1], num_classes)

        # Compute class_weight dict for DNN
        classes = np.unique(y_train)
        class_weights_vals = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(classes, class_weights_vals)}

        dnn.fit(np.array(X_train), np.array(y_train),
                epochs=30, batch_size=32, verbose=0, class_weight=class_weight)

        dnn.save(os.path.join(model_dir, "DNN.h5"))

        preds = dnn.predict(np.array(X_test)).argmax(axis=1)
        acc = accuracy_score(y_test, preds)
        results["DNN"] = acc
        print(f"DNN Accuracy: {acc:.4f}")

    # Save preprocessing objects
    if scaler is not None:
        joblib.dump(scaler, os.path.join(model_dir, "minmax_scaler.pkl"))

    if encoder is not None:
        joblib.dump(encoder, os.path.join(model_dir, "onehot_encoder.pkl"))

    # Save results as CSV
    pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": list(results.values())
    }).to_csv(os.path.join(model_dir, "accuracy_results.csv"), index=False)

    return results, (X_train, X_test, y_train, y_test)
