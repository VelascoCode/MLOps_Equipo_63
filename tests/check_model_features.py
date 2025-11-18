import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
from app import load_model, align_dataframe

MODEL_PATH = "models/final_model.pkl"


def main():
    model = load_model(MODEL_PATH)
    print("Model type:", type(model))

    fnames = getattr(model, "feature_names_in_", None)
    if fnames is None:
        print("Model has no feature_names_in_. Cannot assert feature order.")
        return

    print("Number of expected features:", len(fnames))
    print("First 10 features:", fnames[:10])

    # Check if n_tokens_title is first feature
    try:
        idx = list(fnames).index("n_tokens_title")
        print("n_tokens_title index in feature_names_in_:", idx)
    except ValueError:
        print("n_tokens_title not found in feature_names_in_")

    # Build a minimal input with only a couple of fields
    sample = {"n_tokens_title": 10, "n_tokens_content": 200}

    # Input A: without url/timedelta
    df_a = pd.DataFrame([sample])
    df_a_aligned = align_dataframe(df_a, model)
    pred_a = model.predict(df_a_aligned)
    print("Prediction A:", pred_a)

    # Input B: with url and timedelta included
    sample_b = {"url": "http://example.com/article", "timedelta": 5, **sample}
    df_b = pd.DataFrame([sample_b])
    df_b_aligned = align_dataframe(df_b, model)
    pred_b = model.predict(df_b_aligned)
    print("Prediction B:", pred_b)

    same = np.array_equal(pred_a, pred_b)
    print("Predictions equal when including url/timedelta?", same)

    # Show aligned dataframes columns and first row values (for debugging)
    print("Aligned A columns (first 10):", df_a_aligned.columns[:10].tolist())
    print("Aligned B columns (first 10):", df_b_aligned.columns[:10].tolist())
    print("Aligned A first row (first 10):", df_a_aligned.iloc[0].tolist()[:10])
    print("Aligned B first row (first 10):", df_b_aligned.iloc[0].tolist()[:10])


if __name__ == "__main__":
    main()
