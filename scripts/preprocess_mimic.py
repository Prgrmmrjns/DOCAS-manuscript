import re
import pandas as pd

df = pd.read_csv("../mimic-iv.csv")

df.drop(
    columns=["stay_id", "subject_id", "hadm_id", "ARDS", "sirs_flag", "apache_II", "sofa", "saps_II", "admittime", "deathhour", "starttime", "Ventillation duration [min]", "MV_start_hour"
    ], inplace=True)

def _clean_name(name: str) -> str:
    # Drop bracket annotations like " [mmhG]" or "[nan]".
    s = re.sub(r"\[[^\]]*\]", "", name)
    s = s.replace("mmhG", "").replace("MMHG", "").replace("nan", "").replace("NaN", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s

# Apply _clean_name to all column names of df
df.columns = [ _clean_name(col) for col in df.columns ]

# Remove all columns with _min, _max, _median, or _std in the column name
agg_tokens = ("_min", "_max", "_median", "_std")
cols_to_remove = [col for col in df.columns if any(tok in col for tok in agg_tokens)]
df.drop(columns=cols_to_remove, inplace=True)

df.to_csv("../mimic-iv_processed.csv", index=False)