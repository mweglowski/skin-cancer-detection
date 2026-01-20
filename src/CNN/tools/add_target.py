import pandas as pd

df = pd.read_csv(r"")

df["diagnosis_2"] = df["diagnosis_2"].astype(str).str.lower()

df["target"] = df["diagnosis_2"].str.contains("malignant", case=False, na=False).astype(int)

df.to_csv(r"", index=False)
