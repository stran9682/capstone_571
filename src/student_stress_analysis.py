import pandas as pd
import sqlite3
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results", exist_ok = True)
os.makedirs("figures", exist_ok = True)

data = pd.read_csv('data/StressLevelDataset.csv')
con = sqlite3.connect(':memory:')

data.to_sql('data', con, index = False, if_exists = 'replace')

### Filter rows and columns first
def q(sql):
    return pd.read_sql(sql, con)

features = pd.read_sql("""
    SELECT      anxiety_level,
                self_esteem,
                mental_health_history,
                depression,
                headache,
                blood_pressure,
                sleep_quality,
                breathing_problem,
                noise_level,
                living_conditions,
                safety,
                basic_needs,
                academic_performance,
                study_load,
                teacher_student_relationship,
                future_career_concerns,
                social_support,
                peer_pressure,
                extracurricular_activities,
                bullying
    FROM        data
    ORDER BY    stress_level ASC
""", con)

print("Original rows:", len(features))
features = features.dropna()
print("Rows after filtering missing values:", len(features))
print(features)