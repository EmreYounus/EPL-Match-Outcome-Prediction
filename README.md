# EPL-Match-Outcome-Prediction
A machine learning project to predict Premier League match outcomes using historical data and feature engineering

Welcome to my EPL Match Outcome Prediction project. This project uses historical Premier League match data to predict whether the home team will win. I built this project to combine some fun football insights with machine learning techniques.

## What’s This Project About?

- **Goal:**  
  Predict if the home team wins (versus drawing or losing) using past match data.

- **Key Features:**  
  - **Cumulative Averages:** I calculate each team’s performance before the match—for example, the home team’s average goals scored at home and the away team’s average goals conceded.  
  - **Odds Difference:** I also look at the difference between bookmaker odds for home and away wins to add more context.

- **Models Explored:**  
  I compared three different models to see which one performs best:  
  - Logistic Regression  
  - Random Forest  
  - Gradient Boosting

- **Validation:**  
  To avoid overfitting, I used 5-fold cross-validation to compare the models on accuracy and ROC AUC.

## The Dataset

For this project, I’m using historical match data from [Football-Data.co.uk](http://www.football-data.co.uk/englandm.php). The CSV file (named `epl_matches.csv`) should include columns like:
- `Date` – the match date
- `HomeTeam` – the home team’s name
- `AwayTeam` – the away team’s name
- `FTHG` – full-time home goals
- `FTAG` – full-time away goals
- `B365H` – home win odds
- `B365A` – away win odds

**Tip:** Instead of uploading the full dataset here, please download it from the link above and save it as `epl_matches.csv` in this folder.

## How to Run It

1. **Download the dataset** as described and place it in this folder with the name `epl_matches.csv`.
2. Make sure you have Python 3 installed.
3. Install the necessary packages:
   ```bash
   pip install pandas numpy scikit-learn
4. run the script
   python match_outcome_prediction.py
