import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. Load your featurized data
# Ensure this file name matches your Phase 2 output exactly
data_path = 'phishing_final_features.csv'
df = pd.read_csv(data_path)

# 2. Balance the data (50/50 split to avoid bias)
df_legit = df[df['label'] == 0]
df_phish = df[df['label'] == 1].sample(n=len(df_legit), random_state=42)
df_balanced = pd.concat([df_legit, df_phish]).sample(frac=1, random_state=42)

# 3. Define Features (X) and Target (y)
X = df_balanced.drop(['url', 'label'], axis=1)
y = df_balanced['label']

# 4. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize and TRAIN the model (This creates 'rf_model')
print("Training the Random Forest model... This may take a moment.")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 6. SAVE the model to your folder
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, 'phishing_rf_model.pkl')

joblib.dump(rf_model, save_path)

print("\n" + "="*40)
print(f"SUCCESS: Model trained and saved!")
print(f"LOCATION: {save_path}")
print("="*40)