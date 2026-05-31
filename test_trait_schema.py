#!/usr/bin/env python3
"""Quick schema check for mirlab/TRAIT — run locally before data_trait.py."""
import os
import datasets
from huggingface_hub import login

# Auto-login from cached token if HF_TOKEN env var not set
token = os.environ.get("HF_TOKEN")
if not token:
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        with open(token_path) as f:
            token = f.read().strip()
if token:
    login(token=token, add_to_git_credential=False)
    print("✅ Logged in with token.")
else:
    print("⚠️  No HF token found. Set HF_TOKEN env var or run 'huggingface-cli login'.")

TRAIT_SPLITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness",
                "Neuroticism", "Machiavellianism", "Narcissism", "Psychopathy"]

# Print first row of first split to confirm field names
ds = datasets.load_dataset("mirlab/TRAIT", split=TRAIT_SPLITS[0], token=token)
print(f"\n✅ Split '{TRAIT_SPLITS[0]}' loaded: {len(ds)} samples")
row = ds[0]
print("Keys:", list(row.keys()))
for k, v in row.items():
    print(f"  {k}: {str(v)[:120]}")
