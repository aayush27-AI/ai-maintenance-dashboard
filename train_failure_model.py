import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---------- LOAD DATA ----------
df = pd.read_csv('ai4i2020_cleaned.csv')

# ---------- FEATURES & TARGET ----------
features = ['air_temperature', 'process_temperature', 'rpm', 'torque', 'tool_wear']
X = df[features]
y = df['machine_failure']

# ---------- TRAIN-TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- TRAIN MODEL ----------
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',   # handles the ~3% imbalance we saw earlier
    random_state=42
)
clf.fit(X_train, y_train)

# ---------- EVALUATE ----------
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------- SAVE MODEL ----------
joblib.dump(clf, 'failure_prediction_model.pkl')
print("\nModel saved as failure_prediction_model.pkl")