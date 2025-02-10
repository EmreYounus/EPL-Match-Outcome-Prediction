#!/usr/bin/env python3
"""
Project 1: Premier League Match Outcome Prediction

This script loads historical Premier League match data, computes pre-match features 
(such as each team’s cumulative average goals), and then predicts match outcomes (home win vs. not)
using three different classifiers. It compares models using 5-fold cross-validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# -------------------------------
# 1. Data Loading and Preprocessing
# -------------------------------

# Load dataset
df = pd.read_csv('epl_matches.csv')

# Convert 'Date' to datetime and sort by date to simulate sequential matches
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)

# Create target: 1 if home team wins, 0 otherwise (draw or away win)
df['target'] = (df['FTHG'] > df['FTAG']).astype(int)

# -------------------------------
# 2. Feature Engineering: Pre-Match Cumulative Averages
# -------------------------------

# We compute the home team's average goals scored (only in home matches) from previous games,
# and the away team's average goals conceded (only in away matches) from previous games.
# This simulates a “pre-match” feature that would be available before the game starts.
df['home_avg_goals_scored'] = np.nan
df['away_avg_goals_conceded'] = np.nan

# Dictionaries to store cumulative lists for each team
home_stats = {}
away_stats = {}

# Loop through matches in chronological order
for index, row in df.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    
    # For the home team: use previous home matches to compute average goals scored
    if home_team in home_stats and len(home_stats[home_team]) > 0:
        df.loc[index, 'home_avg_goals_scored'] = np.mean(home_stats[home_team])
    else:
        # If no history, assign the current match’s home goals as a placeholder
        df.loc[index, 'home_avg_goals_scored'] = row['FTHG']
    
    # For the away team: use previous away matches to compute average goals conceded
    if away_team in away_stats and len(away_stats[away_team]) > 0:
        df.loc[index, 'away_avg_goals_conceded'] = np.mean(away_stats[away_team])
    else:
        # If no history, assign the current match’s away goals conceded as a placeholder
        df.loc[index, 'away_avg_goals_conceded'] = row['FTAG']
    
    # Update the historical stats for future matches
    home_stats.setdefault(home_team, []).append(row['FTHG'])
    away_stats.setdefault(away_team, []).append(row['FTAG'])

# Additional Feature: Bookmaker odds difference (B365H and B365A are pre-match features)
df['odds_diff'] = df['B365H'] - df['B365A']

# Drop any rows with missing values (could happen in the early rounds)
df.dropna(subset=['home_avg_goals_scored', 'away_avg_goals_conceded', 'odds_diff'], inplace=True)

# Define feature set: expanded features
features = ['home_avg_goals_scored', 'away_avg_goals_conceded', 'odds_diff']
X = df[features]
y = df['target']

# -------------------------------
# 3. Train-Test Split and Model Comparison
# -------------------------------

# Split data into train and test sets (20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define three classifiers for comparison
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Use Stratified K-Fold cross-validation (5 folds)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_results = {}

print("Cross-validation results:")
for name, model in models.items():
    # Use cross_val_score to compute accuracy
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    model_results[name] = cv_scores.mean()
    print(f"{name}: Mean Accuracy = {cv_scores.mean():.3f} (Std: {cv_scores.std():.3f})")

# Choose the best model based on CV accuracy
best_model_name = max(model_results, key=model_results.get)
best_model = models[best_model_name]
print(f"\nSelected best model: {best_model_name}")

# Train the best model on the full training set and evaluate on the test set
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy ({best_model_name}): {test_accuracy:.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Optionally, compute ROC AUC (if probabilities available)
if hasattr(best_model, "predict_proba"):
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {auc:.3f}")
