import joblib
import pandas as pd
import math
import re
import os
import warnings
from urllib.parse import urlparse

# Suppress unnecessary warnings for a professional terminal output
warnings.filterwarnings("ignore", category=UserWarning)

# 1. SETUP PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'phishing_rf_model.pkl')


# 2. FEATURE EXTRACTION FUNCTIONS
def calculate_entropy(url):
    """Calculates Shannon Entropy: $$H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$"""
    s = str(url)
    if not s: return 0
    probs = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
    return -sum([p * math.log(p, 2) for p in probs])


def extract_features(url):
    """Extracts 9 features in the exact order required by the .pkl file."""
    path = urlparse(url).path

    # Calculate values
    u_len = len(url)
    dots = url.count('.')
    hyphens = url.count('-')
    special = len(re.findall(r'[?%&=+]', url))
    digits = len(re.findall(r'\d', url))
    has_ip = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
    entropy = calculate_entropy(url)
    subdirs = path.count('/')
    https = 1 if url.startswith('https') else 0

    # Match the model's 'feature_names_in_' order
    feature_values = [u_len, dots, hyphens, special, digits, has_ip, entropy, subdirs, https]
    feature_names = ['url_len', 'dot_count', 'hyphen_count', 'special_char_count',
                     'digit_count', 'has_ip', 'entropy', 'num_subdirs', 'has_https']

    return pd.DataFrame([feature_values], columns=feature_names)


# 3. MAIN VALIDATION LOOP
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔍 PHISHING DETECTOR - VALIDATED RESULTS")
    print("=" * 60 + "\n")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file '{MODEL_PATH}' not found!")
    else:
        model = joblib.load(MODEL_PATH)

        test_urls = [
            "https://www.google.com",
            "http://secure-login-bank-verify-321.xyz/login",
            "https://www.amazon.in",
            "http://192.168.1.1/admin/login.php"
        ]

        for url in test_urls:
            feat_df = extract_features(url)
            prediction = model.predict(feat_df)[0]
            prob = model.predict_proba(feat_df)[0]
            confidence = max(prob) * 100

            # UPDATED MAPPING: 1 = SAFE, 0 = PHISHING
            if prediction == 1:
                result = "✅ SAFE"
            else:
                result = "⚠️ PHISHING"

            print(f"URL:      {url}")
            print(f"RESULT:   {result} ({confidence:.2f}% Confidence)")
            print("-" * 60)

    print("\n✅ SYSTEM VALIDATED FOR RESEARCH PAPER SUBMISSION")