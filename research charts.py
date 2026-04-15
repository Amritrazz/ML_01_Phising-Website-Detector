import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os

# 1. Load Model and Data
current_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(current_dir, 'phishing_rf_model.pkl'))
df = pd.read_csv(os.path.join(current_dir, 'phishing_final_features.csv'))

# 2. Re-create the Balanced Test Set (to ensure fair evaluation)
df_legit = df[df['label'] == 0]
df_phish = df[df['label'] == 1].sample(n=len(df_legit), random_state=42)
df_balanced = pd.concat([df_legit, df_phish]).sample(frac=1, random_state=42)

X = df_balanced.drop(['url', 'label'], axis=1)
y = df_balanced['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Generate Predictions
y_pred = model.predict(X_test)

# --- CHART 1: CONFUSION MATRIX ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
# Note: Using your label logic (0=Phish, 1=Safe)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Phishing', 'Safe'],
            yticklabels=['Phishing', 'Safe'])
plt.title('Confusion Matrix: Predictive Accuracy')
plt.ylabel('Actual Category')
plt.xlabel('Model Prediction')
plt.savefig('confusion_matrix.png')
print("✅ Saved: confusion_matrix.png")

# --- CHART 2: FEATURE IMPORTANCE ---
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
feature_names = X.columns
feat_importances = pd.Series(importances, index=feature_names).sort_values(ascending=True)
feat_importances.plot(kind='barh', color='teal')
plt.title('Feature Importance: What drives the detection?')
plt.xlabel('Relative Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("✅ Saved: feature_importance.png")

# 4. Print the Scientific Report
print("\n" + "="*30)
print("FINAL SCIENTIFIC REPORT")
print("="*30)
print(classification_report(y_test, y_pred, target_names=['Phishing', 'Safe']))