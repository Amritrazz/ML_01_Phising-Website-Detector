import gradio as gr
import pandas as pd
import joblib
import math
import os
from urllib.parse import urlparse

# --- 1. Load the "Brain" ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'phishing_rf_model.pkl')
model = joblib.load(model_path)


# --- 2. Feature Extraction Logic ---
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
        'has_ip': 1 if any(char.isdigit() for char in parsed.netloc) and parsed.netloc.replace('.',
                                                                                               '').isdigit() else 0,
        'entropy': get_entropy(url),
        'num_subdirs': url.count('/'),
        'has_https': 1 if url.startswith('https') else 0
    }


# --- 3. The Prediction Function ---
def analyze_url(url):
    if not url.startswith(('http://', 'https://')):
        return "❌ Invalid URL", "Please include http:// or https://", 0

    # Extract and align columns
    feats = extract_features(url)
    feature_cols = ['url_len', 'dot_count', 'hyphen_count', 'special_char_count', 'digit_count', 'has_ip', 'entropy',
                    'num_subdirs', 'has_https']
    feat_df = pd.DataFrame([feats])[feature_cols]

    # Predict probabilities
    # Index 0 = Phishing, Index 1 = Safe
    probs = model.predict_proba(feat_df)[0]

    if probs[1] > 0.5:
        result = "✅ LEGITIMATE (SAFE)"
        confidence = probs[1]
        color_theme = "This site looks trustworthy."
    else:
        result = "⚠️ PHISHING (MALICIOUS)"
        confidence = probs[0]
        color_theme = "Warning: This site shows suspicious patterns!"

    return result, color_theme, round(float(confidence), 2)


# --- 4. Build the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Zero-Hour Phishing Detector")
    gr.Markdown("Enter a URL below to analyze it using our trained Random Forest model.")

    with gr.Row():
        url_input = gr.Textbox(placeholder="https://example.com", label="Website URL")

    with gr.Row():
        btn = gr.Button("Analyze URL", variant="primary")

    with gr.Column():
        output_label = gr.Textbox(label="Verdict")
        output_desc = gr.Textbox(label="Details")
        output_conf = gr.Number(label="Confidence Score (0.0 to 1.0)")

    btn.click(fn=analyze_url, inputs=url_input, outputs=[output_label, output_desc, output_conf])

    gr.Examples(
        examples=["https://www.google.com", "http://secure-login-bank-321.xyz"],
        inputs=url_input
    )

demo.launch(share=True)