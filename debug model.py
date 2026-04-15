import pandas as pd
import joblib
import math
import os
from urllib.parse import urlparse

# --- 1. FEATURE EXTRACTION ---
def get_entropy(text):
    if not text: return 0
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    return - sum([p * math.log(p) / math.log(2.0) for p in prob])

def extract_features(url):
    parsed = urlparse(url)
    return {
        'url_len': len(url),
        'dot_count': url.count('.'),
        'hyphen_count': url.count('-'),
        'special_char_count': sum(not c.isalnum() for c in url),
        'digit_count': sum(c.isdigit() for c in url),
        'has_ip': 1 if any(char.isdigit() for char in parsed.netloc) and parsed.netloc.replace('.', '').isdigit() else 0,
        'entropy': get_entropy(url),
        'num_subdirs': url.count('/'),
        'has_https': 1 if url.startswith('https') else 0
    }

# --- 2. THE DEBUGGER LOGIC ---
def run_debug():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'phishing_rf_model.pkl')

    if not os.path.exists(model_path):
        print("❌ Model file not found. Run 'Model training.py' first!")
        return

    # Load model and define columns
    model = joblib.load(model_path)
    feature_cols = ['url_len', 'dot_count', 'hyphen_count', 'special_char_count', 'digit_count', 'has_ip', 'entropy', 'num_subdirs', 'has_https']

    print("🔍 DEBUG START: Testing with corrected label logic...")

    test_urls = [
        "https://web.whatsapp.com",  # Complex but safe
        "http://amzon-security-update.in",  # Looks safe but is phishing
        "https://github.com/login",  # Secure login
        "http://123.45.67.89/pay"  # Raw IP (High risk)
    ]

    for url in test_urls:
        feats = extract_features(url)
        feat_df = pd.DataFrame([feats])[feature_cols]

        # [Index 0 = Phishing probability, Index 1 = Safe probability]
        probs = model.predict_proba(feat_df)[0]

        print(f"\nURL: {url}")
        print(f"📊 Features: {feats}")
        print(f"📈 Raw Scores -> [Phishing (0): {probs[0]:.4f}] | [Safe (1): {probs[1]:.4f}]")

        # --- THE FIXED LINE ---
        if probs[1] > 0.5:
            print("Result: ✅ SAFE")
        else:
            print("Result: ⚠️ PHISHING")

if __name__ == "__main__":
    run_debug()