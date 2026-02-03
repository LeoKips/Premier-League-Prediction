import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np

# Load and preprocess data
matches = pd.read_csv("Premier League\matches.csv", index_col=0)
matches = matches.reset_index()
matches["date"] = pd.to_datetime(matches["date"])
matches = matches.drop(columns=["comp", "notes"], errors="ignore")
matches["target"] = matches["result"].map({"W": 1, "D": 0, "L": -1})
matches["venue_code"] = matches["venue"].map({"Home": 1, "Away": 0})
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek

# Feature engineering function
def add_features(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(5, closed='left').mean()
    group[new_cols] = rolling_stats
    group["form_wins"] = group["target"].rolling(5, closed='left').apply(lambda x: (x == 1).sum(), raw=True)
    group["form_draws"] = group["target"].rolling(10, closed='left').apply(lambda x: (x == 0).sum(), raw=True)
    group["home_win_rate"] = group["target"].where(group["venue"] == "Home").rolling(10, closed='left').mean().ffill()
    group["points_per_game"] = group["target"].replace({1: 3, 0: 1, -1: 0}).rolling(10, closed='left').mean()
    group["days_since_last"] = (group["date"] - group["date"].shift(1)).dt.days.fillna(7)
    return group

# Rolling average columns
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt", "xg", "xga"]
new_cols = [f"{c}_rolling" for c in cols]
opp_new_cols = [f"opp_{c}_rolling" for c in cols]

# Apply features
matches_rolling = matches.groupby("team").apply(lambda x: add_features(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])

# Merge opponent features
matches_rolling = matches_rolling.merge(
    matches_rolling[["date", "team"] + new_cols + ["points_per_game"]].rename(
        columns={"team": "opponent", **{c: f"opp_{c}" for c in new_cols}, "points_per_game": "opp_points_per_game"}
    ),
    left_on=["date", "opponent"], right_on=["date", "opponent"], how="left"
)

# Add head-to-head feature
def add_head_to_head(df):
    h2h = []
    for i, row in df.iterrows():
        past_matches = df[(df["team"] == row["team"]) & (df["opponent"] == row["opponent"]) & (df["date"] < row["date"])].tail(5)
        h2h.append(past_matches["target"].mean() if not past_matches.empty else 0)
    df["h2h_win_rate"] = h2h
    return df

matches_rolling = add_head_to_head(matches_rolling)

# Predictors
predictors = ["venue_code", "opp_code", "hour", "day_code", "form_wins", "form_draws", "home_win_rate", 
              "points_per_game", "days_since_last", "h2h_win_rate"] + new_cols + opp_new_cols + ["opp_points_per_game"]

# Fill NaN values
matches_rolling[predictors] = matches_rolling[predictors].fillna(matches_rolling[predictors].mean())

# Train XGBoost model
def train_model(data, predictors):
    train = data[data["date"] < '2025-02-01']
    val = data[(data["date"] >= '2025-02-01') & (data["date"] < '2025-03-31')]
    if train.empty or val.empty:
        print("Training or validation data empty. Using all data before March 31.")
        train = data[data["date"] < '2025-03-31']
    
    dtrain = xgb.DMatrix(train[predictors], label=train["target"] + 1)  # Shift to 0,1,2
    dval = xgb.DMatrix(val[predictors], label=val["target"] + 1)
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss'
    }
    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10, verbose_eval=False)
    return model

model = train_model(matches_rolling, predictors)

# Define weekend fixtures (April 01-03, 2025)
weekend_fixtures = pd.DataFrame({
    "date": ["2025-04-01", "2025-04-01", "2025-04-01", "2025-04-02", "2025-04-02", "2025-04-02", "2025-04-02", "2025-04-02", "2025-04-02", "2025-04-03"],
    "team": ["Arsenal", "Wolverhampton Wanderers", "Nottingham Forest", "Bournemouth", "Brighton and Hove Albion", 
             "Manchester City", "Newcastle United", "Southampton", "Liverpool", "Chelsea"],
    "opponent": ["Fulham", "West Ham United", "Manchester United", "Ipswich Town", 
                 "Aston Villa", "Leicester City", "Brentford", "Crystal Palace", "Everton", "Tottenham Hotspur"],
    "venue": ["Home", "Home", "Home", "Home", "Home", "Home", "Home", "Home", "Home", "Home"],
    "time": ["19:45", "19:45", "20:00", "19:45", "19:45", "19:45", "19:45", "19:45", "20:00", "20:00"]
})
weekend_fixtures["date"] = pd.to_datetime(weekend_fixtures["date"])
weekend_fixtures["venue_code"] = weekend_fixtures["venue"].map({"Home": 1, "Away": 0})
weekend_fixtures["opp_code"] = weekend_fixtures["opponent"].astype("category").cat.codes
weekend_fixtures["hour"] = weekend_fixtures["time"].str.replace(":.+", "", regex=True).astype("int")
weekend_fixtures["day_code"] = weekend_fixtures["date"].dt.dayofweek

# Add features for weekend fixtures
for team in weekend_fixtures["team"].unique():
    team_matches = matches_rolling[matches_rolling["team"] == team]
    last_matches = team_matches[team_matches["date"] < '2025-03-31'].tail(10)
    if len(last_matches) >= 5:
        for col, new_col in zip(cols, new_cols):
            weekend_fixtures.loc[weekend_fixtures["team"] == team, new_col] = last_matches[col].tail(5).mean()
        weekend_fixtures.loc[weekend_fixtures["team"] == team, "form_wins"] = (last_matches["target"] == 1).tail(5).sum()
        weekend_fixtures.loc[weekend_fixtures["team"] == team, "form_draws"] = (last_matches["target"] == 0).sum()
        weekend_fixtures.loc[weekend_fixtures["team"] == team, "home_win_rate"] = last_matches[last_matches["venue"] == "Home"]["target"].mean()
        weekend_fixtures.loc[weekend_fixtures["team"] == team, "points_per_game"] = last_matches["target"].replace({1: 3, 0: 1, -1: 0}).mean()
        weekend_fixtures.loc[weekend_fixtures["team"] == team, "days_since_last"] = (last_matches["date"].iloc[-1] - last_matches["date"].iloc[-2]).days if len(last_matches) > 1 else 7
        h2h_matches = matches_rolling[(matches_rolling["team"] == team) & 
                                     (matches_rolling["opponent"] == weekend_fixtures[weekend_fixtures["team"] == team]["opponent"].iloc[0]) & 
                                     (matches_rolling["date"] < '2025-03-31')].tail(5)
        weekend_fixtures.loc[weekend_fixtures["team"] == team, "h2h_win_rate"] = h2h_matches["target"].mean() if not h2h_matches.empty else 0
    else:
        print(f"Warning: Insufficient data for {team}. Using mean values.")
        for col, new_col in zip(cols, new_cols):
            weekend_fixtures.loc[weekend_fixtures["team"] == team, new_col] = matches_rolling[col].mean()
        weekend_fixtures.loc[weekend_fixtures["team"] == team, "form_wins"] = matches_rolling["target"].eq(1).mean() * 5
        weekend_fixtures.loc[weekend_fixtures["team"] == team, "form_draws"] = matches_rolling["target"].eq(0).mean() * 10
        weekend_fixtures.loc[weekend_fixtures["team"] == team, "home_win_rate"] = matches_rolling[matches_rolling["venue"] == "Home"]["target"].mean()
        weekend_fixtures.loc[weekend_fixtures["team"] == team, "points_per_game"] = matches_rolling["target"].replace({1: 3, 0: 1, -1: 0}).mean()
        weekend_fixtures.loc[weekend_fixtures["team"] == team, "days_since_last"] = 7

for opp in weekend_fixtures["opponent"].unique():
    opp_matches = matches_rolling[matches_rolling["team"] == opp]
    last_matches = opp_matches[opp_matches["date"] < '2025-03-31'].tail(5)
    if len(last_matches) >= 5:
        for col, new_col in zip(cols, opp_new_cols):
            weekend_fixtures.loc[weekend_fixtures["opponent"] == opp, new_col] = last_matches[col].mean()
        weekend_fixtures.loc[weekend_fixtures["opponent"] == opp, "opp_points_per_game"] = last_matches["target"].replace({1: 3, 0: 1, -1: 0}).mean()
    else:
        print(f"Warning: Insufficient data for opponent {opp}. Using mean values.")
        for col, new_col in zip(cols, opp_new_cols):
            weekend_fixtures.loc[weekend_fixtures["opponent"] == opp, new_col] = matches_rolling[col].mean()
        weekend_fixtures.loc[weekend_fixtures["opponent"] == opp, "opp_points_per_game"] = matches_rolling["target"].replace({1: 3, 0: 1, -1: 0}).mean()

# Fill NaN in weekend_fixtures
weekend_fixtures[predictors] = weekend_fixtures[predictors].fillna(weekend_fixtures[predictors].mean())

# Predict outcomes with probabilities
dweekend = xgb.DMatrix(weekend_fixtures[predictors])
probs = model.predict(dweekend)  # [Loss, Draw, Win]
preds = np.argmax(probs, axis=1) - 1  # Convert to -1, 0, 1

# Identify tight calls (top two probs within 10%)
tight_calls = []
for prob in probs:
    sorted_probs = np.sort(prob)[::-1]  # Sort descending
    tight_calls.append("Yes" if (sorted_probs[0] - sorted_probs[1]) < 0.10 else "No")

upcoming_combined = pd.DataFrame({
    "Date": weekend_fixtures["date"],
    "Home Team": weekend_fixtures["team"],
    "Away Team": weekend_fixtures["opponent"],
    "Venue": weekend_fixtures["venue"],
    "Win Prob": probs[:, 2],
    "Draw Prob": probs[:, 1],
    "Loss Prob": probs[:, 0],
    "Predicted Outcome": preds,
    "Tight Call": tight_calls
})
outcome_map = {1: "Home Win", 0: "Draw", -1: "Away Win"}
upcoming_combined["Predicted Outcome"] = upcoming_combined["Predicted Outcome"].map(outcome_map)

# Display and save predictions
print("\nPredictions for Premier League Matches This Gameweek (April 01-03, 2025):")
print(upcoming_combined.to_string(index=False))
upcoming_combined.to_csv("weekend_predictions_updated_april_01_03_2025.csv", index=False)
print("\nPredictions saved to 'updated_weekend_predictions_april_01_03_2025.csv'")