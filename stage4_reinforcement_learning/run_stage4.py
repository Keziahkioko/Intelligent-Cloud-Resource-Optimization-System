import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from cloud_env import run_stage4, run_stage5

print("Loading features CSV...")
df = pd.read_csv('../data/cloud_performance_features.csv')
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

results = run_stage4(df)
run_stage5(results, report_path='master_report.json')