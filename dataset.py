import pandas as pd

# Load your three datasets
df1 = pd.read_csv(r"C:\Users\amrit\Downloads\archive\phishing_features.csv")
df2 = pd.read_csv(r"C:\Users\amrit\Downloads\archive (1)\dataset_balanced.csv")
df3 = pd.read_csv(r"C:\Users\amrit\Downloads\archive (2)\phishing_legit_dataset_KD_10000.csv")

# Print columns to see the differences
print(f"DS1 Columns: {df1.columns.tolist()}")
print(f"DS2 Columns: {df2.columns.tolist()}")
print(f"DS3 Columns: {df3.columns.tolist()}")

# Align column names
df1_clean = df1[['url', 'label']]
df2_clean = df2[['url', 'label']]
df3_clean = df3.rename(columns={'text': 'url'})[['url', 'label']]

# Combine and Clean
df_master = pd.concat([df1_clean, df2_clean, df3_clean], axis=0, ignore_index=True)
df_master = df_master.drop_duplicates(subset='url').dropna()
df_master = df_master.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Base dataset created with {len(df_master)} unique rows.")
import pandas as pd

# 1. Clean DS1 (Already has url/label)
df1_final = df1[['url', 'label']].copy()

# 2. Clean DS2 (Ensure label is 0 or 1)
df2_final = df2[['url', 'label']].copy()

# 3. Clean DS3 (Map 'text' to 'url' and convert text labels to 0/1)
# First, see what labels DS3 actually has: print(df3['label'].unique())
df3_final = df3.rename(columns={'text': 'url'}).copy()

# This part converts words like 'phishing' to 1 and 'legitimate' to 0
label_map = {
    'phishing': 1,
    'legitimate': 0,
    'safe': 0,
    'malicious': 1,
    'benign': 0
}

# If DS3 already uses 0 and 1, this won't break anything
df3_final['label'] = df3_final['label'].map(label_map).fillna(df3_final['label'])

# 4. Final Merge
df_master = pd.concat([df1_final, df2_final, df3_final], ignore_index=True)

# 5. Final Force-Cast (Ensure everything is a number)
df_master['label'] = pd.to_numeric(df_master['label'], errors='coerce').fillna(0).astype(int)
import pandas as pd
import math
from urllib.parse import urlparse
import time


def get_entropy(text):
    """Calculates the Shannon Entropy of a string (higher = more random/AI-generated)."""
    if not text: return 0
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    return entropy


def extract_features(url):
    url = str(url)
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


# 1. Load your base dataset
# df_master = pd.read_csv('your_base_dataset.csv')

print(f"Starting feature extraction for {len(df_master)} rows. Please wait...")
start_time = time.time()

# 2. Apply the extraction
# We use apply(pd.Series) to turn the dictionary into columns
features_df = df_master['url'].apply(lambda x: pd.Series(extract_features(x)))

# 3. Join features back to the labels
final_df = pd.concat([df_master[['url', 'label']], features_df], axis=1)

# 4. Save the "Golden Dataset"
final_df.to_csv('phishing_final_features.csv', index=False)

end_time = time.time()
print(f"Success! Processed {len(df_master)} rows in {round(end_time - start_time, 2)} seconds.")
print(final_df.head())
import pandas as pd

# Load the file we created in the previous step
final_df = pd.read_csv('phishing_final_features.csv')

# Now you can check the labels!
print("--- Label Distribution ---")
print(final_df['label'].value_counts())
# Separate the classes
df_legit = final_df[final_df['label'] == 0]
df_phish = final_df[final_df['label'] == 1]

# Randomly sample the Phishing class to match the number of Legitimate sites
df_phish_balanced = df_phish.sample(n=len(df_legit), random_state=42)

# Combine them back together
df_balanced = pd.concat([df_legit, df_phish_balanced])

# Shuffle the new dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("--- New Balanced Distribution ---")
print(df_balanced['label'].value_counts())
