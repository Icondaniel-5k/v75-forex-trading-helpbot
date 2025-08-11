import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load the processed data
df = pd.read_csv("processed_v75_data.csv")

# Features and label
features = df[['rsi', 'ema20', 'ema50', 'ema200', 'slope', 'close',
               'ema20_above_ema50', 'price_vs_ema50', 'volatility', 'rsi_overbought']]

labels = df['signal']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

print(" Test set class distribution:")
print(y_test.value_counts(), '\n')
# Train model
# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
# Handle imbalance using scale_pos_weight
buy_count = y_train.value_counts()[1]
sell_count = y_train.value_counts()[0]
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
  # ratio to help with imbalance

# Now train on the resampled data
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model.fit(X_train, y_train)
# Evaluate
# Predict probabilities and apply custom threshold
# Get predicted probabilities
y_probs = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (BUY)

# Apply your custom threshold (e.g., 0.35)
custom_threshold = 0.35
y_pred_custom = (y_probs > custom_threshold).astype(int)

# Evaluate
print("Custom threshold evaluation (0.35):")
print(classification_report(y_test, y_pred_custom, target_names=["SELL", "BUY"]))

y_pred = (y_probs > 0.4).astype(int)  # Tune threshold as needed
accuracy = accuracy_score(y_test, y_pred)
print(f" Model trained with accuracy: {accuracy:.2f}")

print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=["SELL", "BUY"]))

# Save model
joblib.dump(model, "ml_model.pkl")
print(" Model saved as ml_model.pkl")
