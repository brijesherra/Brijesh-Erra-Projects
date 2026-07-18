# ============================================================
# NBA SALARY VS PERFORMANCE ANALYSIS PROJECT
# ============================================================
#
# Author: Your Name
#
# DESCRIPTION:
# This project analyzes whether NBA players are overpaid
# or underpaid relative to their statistical performance.
#
# The program:
# 1. Pulls FREE NBA stats from nba_api
# 2. Loads FREE salary data from a CSV
# 3. Cleans and filters the data
# 4. Trains a machine learning regression model
# 5. Predicts player salaries
# 6. Calculates residuals
# 7. Finds overpaid and underpaid players
# 8. Evaluates model performance
# 9. Creates professional visualizations
# 10. Saves results to CSV
#
# ============================================================
# REQUIRED PACKAGES
# ============================================================
#
# pip install pandas numpy matplotlib scikit-learn nba_api
#
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from nba_api.stats.endpoints import leaguedashplayerstats


# ============================================================
# SETTINGS
# ============================================================

SEASON = '2023-24'

MINIMUM_MPG = 10
MINIMUM_GP = 20

SALARY_FILE = "salaries.csv"

RANDOM_STATE = 42


# ============================================================
# STEP 1 — DOWNLOAD NBA PLAYER STATS
# ============================================================

print("\n================================================")
print("DOWNLOADING NBA PLAYER STATS")
print("================================================\n")

# Small delay helps avoid NBA API rate limits
time.sleep(1)

try:

    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        per_mode_detailed='PerGame'
    )

    stats_df = stats.get_data_frames()[0]

    print("NBA stats downloaded successfully.\n")

except Exception as e:

    print("ERROR: Failed to retrieve NBA stats.")
    print("\nPossible causes:")
    print("- Internet connection issue")
    print("- NBA API timeout")
    print("- NBA.com temporary block")
    print("\nTechnical details:")
    print(e)

    exit()


# ============================================================
# STEP 2 — SELECT RELEVANT COLUMNS
# ============================================================

stats_df = stats_df[
    [
        'PLAYER_NAME',
        'GP',
        'PTS',
        'AST',
        'REB',
        'STL',
        'BLK',
        'MIN'
    ]
]

# Rename columns for readability
stats_df.columns = [
    'Player',
    'GP',
    'PPG',
    'APG',
    'RPG',
    'SPG',
    'BPG',
    'MPG'
]


# ============================================================
# STEP 3 — LOAD SALARY DATA
# ============================================================

print("Loading salary data...\n")

try:

    salary_df = pd.read_csv(SALARY_FILE)

    print("Salary data loaded successfully.\n")

except FileNotFoundError:

    print(f"ERROR: Could not find file '{SALARY_FILE}'")
    print("\nMake sure salaries.csv is in the same folder.")
    exit()

except Exception as e:

    print("ERROR loading salary data.")
    print(e)
    exit()


# ============================================================
# STEP 4 — CLEAN DATA
# ============================================================

print("Cleaning data...\n")

# Remove extra spaces from names
stats_df['Player'] = stats_df['Player'].str.strip()
salary_df['Player'] = salary_df['Player'].str.strip()

# Filter low-minute players
stats_df = stats_df[stats_df['MPG'] > MINIMUM_MPG]

# Filter players with too few games
stats_df = stats_df[stats_df['GP'] > MINIMUM_GP]

print(f"Players remaining after filtering: {len(stats_df)}\n")


# ============================================================
# STEP 5 — MERGE DATASETS
# ============================================================

print("Merging stats and salary datasets...\n")

df = pd.merge(stats_df, salary_df, on='Player')

print(f"Final dataset contains {len(df)} players.\n")

print("Sample merged dataset:\n")
print(df.head())


# ============================================================
# STEP 6 — DEFINE FEATURES AND TARGET
# ============================================================

# Features used to predict salary
X = df[
    [
        'PPG',
        'APG',
        'RPG',
        'SPG',
        'BPG',
        'MPG'
    ]
]

# Actual salaries
y = df['Salary']


# ============================================================
# STEP 7 — TRAIN / TEST SPLIT
# ============================================================

print("\nCreating train/test split...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)


# ============================================================
# STEP 8 — TRAIN REGRESSION MODEL
# ============================================================

print("Training regression model...\n")

model = LinearRegression()

model.fit(X_train, y_train)

print("Model trained successfully.\n")


# ============================================================
# STEP 9 — MAKE PREDICTIONS
# ============================================================

print("Generating salary predictions...\n")

# Predict on full dataset for analysis
df['PredictedSalary'] = model.predict(X)

# Residuals
df['Residual'] = df['Salary'] - df['PredictedSalary']


# ============================================================
# STEP 10 — EVALUATE MODEL
# ============================================================

print("================================================")
print("MODEL PERFORMANCE")
print("================================================\n")

# Test predictions
test_predictions = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, test_predictions)

rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print(f"R² Score: {r2:.3f}")

print(f"RMSE: ${rmse:,.0f}")

print("\nInterpretation:")
print("- Higher R² is better")
print("- Lower RMSE is better\n")


# ============================================================
# STEP 11 — FEATURE IMPORTANCE
# ============================================================

print("================================================")
print("FEATURE IMPORTANCE")
print("================================================\n")

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

coefficients = coefficients.sort_values(
    by='Coefficient',
    ascending=False
)

print(coefficients)


# ============================================================
# STEP 12 — FIND OVERPAID PLAYERS
# ============================================================

print("\n================================================")
print("TOP 10 MOST OVERPAID PLAYERS")
print("================================================\n")

overpaid = df.sort_values(
    by='Residual',
    ascending=False
)

print(
    overpaid[
        [
            'Player',
            'Salary',
            'PredictedSalary',
            'Residual'
        ]
    ].head(10)
)


# ============================================================
# STEP 13 — FIND UNDERPAID PLAYERS
# ============================================================

print("\n================================================")
print("TOP 10 MOST UNDERPAID PLAYERS")
print("================================================\n")

underpaid = df.sort_values(
    by='Residual'
)

print(
    underpaid[
        [
            'Player',
            'Salary',
            'PredictedSalary',
            'Residual'
        ]
    ].head(10)
)


# ============================================================
# STEP 14 — CREATE SCATTERPLOT
# ============================================================

print("\nGenerating scatterplot...\n")

plt.figure(figsize=(10, 8))

plt.scatter(
    df['PredictedSalary'],
    df['Salary']
)

# Perfect prediction line
max_salary = max(
    df['Salary'].max(),
    df['PredictedSalary'].max()
)

plt.plot(
    [0, max_salary],
    [0, max_salary]
)

plt.xlabel("Predicted Salary")

plt.ylabel("Actual Salary")

plt.title("NBA Salary Prediction Model")

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# STEP 15 — CREATE RESIDUAL CHART
# ============================================================

print("Generating residual chart...\n")

top_residuals = pd.concat([
    overpaid.head(5),
    underpaid.head(5)
])

plt.figure(figsize=(12, 6))

plt.bar(
    top_residuals['Player'],
    top_residuals['Residual']
)

plt.xticks(rotation=45)

plt.ylabel("Residual")

plt.title("Most Overpaid and Underpaid NBA Players")

plt.tight_layout()

plt.show()


# ============================================================
# STEP 16 — SAVE RESULTS
# ============================================================

OUTPUT_FILE = "nba_salary_analysis.csv"

df.to_csv(OUTPUT_FILE, index=False)

print("================================================")
print("RESULTS SAVED")
print("================================================\n")

print(f"Analysis saved to: {OUTPUT_FILE}")


# ============================================================
# STEP 17 — FINAL SUMMARY
# ============================================================

print("\n================================================")
print("PROJECT COMPLETE")
print("================================================\n")

print("This project demonstrated:")

print("\n- API data collection")
print("- Data cleaning")
print("- Feature engineering")
print("- Regression modeling")
print("- Residual analysis")
print("- Data visualization")
print("- Model evaluation")
print("- Defensive programming")

print("\nNBA Salary vs Performance analysis complete!\n")