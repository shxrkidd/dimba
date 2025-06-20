import argparse
from datetime import datetime, timedelta
import pandas as pd
import jax
import jax.numpy as jnp
import flax.linen as nn
from colorama import Fore, Style
import requests
import json

# football data 
# use these free football data providers and utilities
# Get all competitions

#StatsBomb data
#fetch competitions and matches from StatsBomb's open data repository
comps_url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
competitions = requests.get(comps_url).json()

all_matches = []
for comp in competitions:
    comp_id = comp['competition_id']
    season_id = comp['season_id']
    matches_url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/{comp_id}/{season_id}.json"
    matches = requests.get(matches_url).json()
    all_matches.extend(matches)

print(f"Fetched {len(all_matches)} matches from {len(competitions)} competitions.")

#Club Elo data ---
elo_url = "http://api.clubelo.com/fixtures"
elo_matches = pd.read_csv(elo_url)
print(elo_matches.head())

# FiveThirtyEight data
fivethirtyeight_url = "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv"
fivethirtyeight_matches = pd.read_csv(fivethirtyeight_url)
print(fivethirtyeight_matches.head())

#get team names and their IDs

def get_teams_from_openfootball(raw_json_url):
    import requests
    response = requests.get(raw_json_url)
    data = response.json()
     # Some files use "clubs", some use "teams"
    key = "clubs" if "clubs" in data else "teams"
    # Not all teams have an 'id', so fallback to name if missing
    teams = {team['name']: team.get('id', team['name']) for team in data[key]}
    return teams

#use fuzzy matching to match team names from different sources

import requests
from fuzzywuzzy import process

def get_json_data(url):
    #fetch JSON data from a REST API endpoint.
    print(f"Fetching data from {url}...")
    response = requests.get(url)
    return response.json()

def build_standard_team_list(json_data):
    #extract standard team names from JSON for fuzz matching.
    key = "teams" if "teams" in json_data else "clubs"
    return [team['name'] for team in json_data.get(key, [])]
    

def standardize_team_name(name, standard_teams):
    #matches a raw name to the closest team name in the reference list.
    match, score = process.extractOne(name, standard_teams)
    return match if score > 80 else name  # You can tweak this threshold

# build index mapping 
# example: fetch a standard team list from openfootball (replace with actual URL as needed)
openfootball_teams_url = "https://raw.githubusercontent.com/openfootball/football.json/master/2020-21/en.1.clubs.json"
standard_teams_json = get_json_data(openfootball_teams_url)
standard_teams = build_standard_team_list(standard_teams_json)

football_team_index_current = {team: idx for idx, team in enumerate(standard_teams)}

#fuzzy-match and convert incoming team names
def team_to_index(name):
    standardized = standardize_team_name(name, standard_teams)
    return football_team_index_current.get(standardized, -1)  # -1 for unknowns

#create dataframe from matches

from collections import defaultdict, deque
from datetime import datetime
import pandas as pd

def statsbomb_matches_to_team_stats_with_form(matches):
    team_stats = defaultdict(lambda: {
        'TEAM_NAME': '', 'MP': 0, 'W': 0, 'D': 0, 'L': 0,
        'GF': 0, 'GA': 0, 'GD': 0, 'PTS': 0, 'PPG': 0.0,
        'FORM_ALL': deque(maxlen=5),
        'FORM_HOME': deque(maxlen=5),
        'FORM_AWAY': deque(maxlen=5),
        'LAST_MATCH_DATE': None  # Add last match date
    })

    h2h_tracker = defaultdict(lambda: deque(maxlen=5))  # (team1, team2) as key

    for match in matches:
        home = match['home_team']['home_team_name']
        away = match['away_team']['away_team_name']
        hs = match['home_score']
        as_ = match['away_score']
        date_str = match.get('match_date') or match.get('date')
        match_date = None
        if date_str:
            try:
                match_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
            except Exception:
                match_date = None

        for team in [home, away]:
            team_stats[team]['TEAM_NAME'] = team
            team_stats[team]['MP'] += 1
            # Update last match date if newer
            if match_date:
                prev_date = team_stats[team]['LAST_MATCH_DATE']
                if not prev_date or match_date > prev_date:
                    team_stats[team]['LAST_MATCH_DATE'] = match_date

        # Goals and base stats
        team_stats[home]['GF'] += hs
        team_stats[home]['GA'] += as_
        team_stats[away]['GF'] += as_
        team_stats[away]['GA'] += hs

        # Outcome and points
        if hs > as_:
            team_stats[home]['W'] += 1
            team_stats[home]['PTS'] += 3
            team_stats[away]['L'] += 1
            home_result, away_result = 'W', 'L'
        elif as_ > hs:
            team_stats[away]['W'] += 1
            team_stats[away]['PTS'] += 3
            team_stats[home]['L'] += 1
            home_result, away_result = 'L', 'W'
        else:
            team_stats[home]['D'] += 1
            team_stats[away]['D'] += 1
            team_stats[home]['PTS'] += 1
            team_stats[away]['PTS'] += 1
            home_result = away_result = 'D'

        # Track recent form (W/L/D)
        team_stats[home]['FORM_HOME'].append(home_result)
        team_stats[away]['FORM_AWAY'].append(away_result)
        team_stats[home]['FORM_ALL'].append(home_result)
        team_stats[away]['FORM_ALL'].append(away_result)

        # Track last 5 H2H
        key = tuple(sorted([home, away]))
        h2h_tracker[key].append({
            'date': date_str,
            'home': home,
            'away': away,
            'score': f"{hs}-{as_}"
        })

    # Final metrics
    for team in team_stats:
        stat = team_stats[team]
        stat['GD'] = stat['GF'] - stat['GA']
        stat['PPG'] = round(stat['PTS'] / stat['MP'], 2) if stat['MP'] else 0
        # Convert LAST_MATCH_DATE to string for DataFrame compatibility
        if stat['LAST_MATCH_DATE']:
            stat['LAST_MATCH_DATE'] = stat['LAST_MATCH_DATE'].strftime("%Y-%m-%d")

    df = pd.DataFrame(team_stats.values())
    df['TEAM_ID'] = range(len(df))
    
    return df, h2h_tracker 

#if two haven't played against each other:
pair = tuple(sorted(['Team A', 'Team B']))
recent_h2h = h2h_tracker.get(pair, [])
if not recent_h2h:
    print(f"No historical matchups between {pair[0]} and {pair[1]}")
else:
    print(f"Recent H2H between {pair[0]} and {pair[1]}:")
    for match in recent_h2h:
         print(match)

#handle missing stats and odds        
from datetime import datetime, timedelta
import pandas as pd

def create_todays_games_data(games, df, odds, football_team_index_current):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []
    home_team_days_rest = []
    away_team_days_rest = []

    for game in games:
        home_team, away_team = game[0], game[1]
        if home_team not in football_team_index_current or away_team not in football_team_index_current:
            continue

        # get odds or prompt user
        # odds is expected to be a dictionary with keys like "HomeTeam:AwayTeam"
        odds_key = f"{home_team}:{away_team}"
        odds_info = odds.get(odds_key)
        if odds_info:
            todays_games_uo.append(odds_info.get('under_over_odds', 2.5))
            home_team_odds.append(odds_info.get(home_team, {}).get('money_line_odds', 0))
            away_team_odds.append(odds_info.get(away_team, {}).get('money_line_odds', 0))
        else:
            print(f"Odds not found for {home_team} vs {away_team}.")
            try:
                todays_games_uo.append(float(input(f"{home_team} vs {away_team} (U/O): ")))
                home_team_odds.append(float(input(f"{home_team} odds: ")))
                away_team_odds.append(float(input(f"{away_team} odds: ")))
            except Exception:
                print("Invalid manual odds input. Skipping game.")
                continue
        # Merge Elo or FiveThirtyEight ratings into your match DataFrame
        # Example for FiveThirtyEight:
        matches = pd.read_csv("your_matches.csv")
        fivethirtyeight_matches = pd.read_csv("spi_matches.csv")
        
        # Merge SPI ratings for home and away teams
        matches = matches.merge(
            fivethirtyeight_matches[['date', 'team1', 'spi1', 'team2', 'spi2']],
            left_on=['date', 'home_team', 'away_team'],
            right_on=['date', 'team1', 'team2'],
            how='left'
        )
        matches['home_spi'] = matches['spi1']
        matches['away_spi'] = matches['spi2']
        # Calculate days rest from schedule CSV
        try:
            schedule_df = pd.read_csv('Data/football-2024-UTC.csv', parse_dates=['Date'], dayfirst=True)
            today = datetime.today()

            # Filter and find last match
            home_games = schedule_df[(schedule_df['HomeTeam'] == home_team) | (schedule_df['AwayTeam'] == home_team)]
            away_games = schedule_df[(schedule_df['HomeTeam'] == away_team) | (schedule_df['AwayTeam'] == away_team)]

            home_recent = home_games[home_games['Date'] <= today].sort_values('Date', ascending=False).head(1)
            away_recent = away_games[away_games['Date'] <= today].sort_values('Date', ascending=False).head(1)

            home_days = (today - home_recent.iloc[0]['Date']).days if not home_recent.empty else 7
            away_days = (today - away_recent.iloc[0]['Date']).days if not away_recent.empty else 7
        except FileNotFoundError:
            print("Schedule file not found. Using default rest of 7 days.")
            home_days = away_days = 7

        home_team_days_rest.append(home_days)
        away_team_days_rest.append(away_days)

        # Get team stat rows
        home_stats = df.loc[df['TEAM_NAME'] == home_team]
        away_stats = df.loc[df['TEAM_NAME'] == away_team]
        if home_stats.empty or away_stats.empty:
            print(f"Stats not found for {home_team} or {away_team}")
            continue

        # Combine and annotate
        combined = pd.concat([home_stats.squeeze(), away_stats.squeeze()])
        combined['Days-Rest-Home'] = home_days
        combined['Days-Rest-Away'] = away_days
        match_data.append(combined)

    if not match_data:
        return None, None, None, None, None

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1).T

    # Select only numeric data for ML
    frame_ml = games_data_frame.select_dtypes(include=['number'])
    data = frame_ml.values.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds   

def load_oddsportal_odds(filepath="full_scraper/odds.json"):
    with open(filepath, "r") as f:
        scraped_matches = json.load(f)

    odds = {}
    for match in scraped_matches:
        home = match["home_team"]
        away = match["away_team"]
        key = f"{home}:{away}"
        odds[key] = {
            home: {"money_line_odds": match.get("home_odds", 0)},
            away: {"money_line_odds": match.get("away_odds", 0)},
            "under_over_odds": match.get("over_under", 2.5)
        }
    return odds
    


odds = load_oddsportal_odds()
games = create_todays_games_from_odds(odds)

class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)
        return x
import optax

def cross_entropy_loss(logits, labels):
    return -jnp.sum(labels * nn.log_softmax(logits))

@jax.jit
def train_step(params, x, y, optimizer, opt_state):
    def loss_fn(params):
        logits = model.apply(params, x)
        return cross_entropy_loss(logits, y)
    grads = jax.grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Example usage:
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)
# x_train, y_train = ... # your training data and one-hot labels
params, opt_state = train_step(params, x_train, y_train, optimizer, opt_state)
if args.nn:
    print("\n------------Neural Network Model Predictions-----------")

    # Normalize the input data
    norm_data = normalize_jax(jnp.array(data))

    # Initialize model and parameters
    model = SimpleNN()
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, norm_data)

    # Predict raw logits and convert to probabilities
    logits = model.apply(params, norm_data)
    probs = nn.softmax(logits)

    for idx, match_key in enumerate(uo):
        home, away = match_key.split(":")

        print(f"\n📊 {away} @ {home}")
        match_probs = probs[idx]
        print(f"Model Win/Draw/Loss Probabilities: {match_probs}")

        # Get odds in match order: home, draw, away
        match_odds = [
            home_odds[idx],
            1 / (1 / home_odds[idx] + 1 / away_odds[idx]),  # Approx draw odds if missing
            away_odds[idx]
        ]

        tips = generate_tip(match_probs, match_odds, threshold=0.6)
        if tips:
            for tip in tips:
                print(f"✅ Value Bet Found: {tip['bet']} at odds {tip['odds']} (Confidence: {tip['confidence']}, Margin: {tip['value_margin']})")
        else:
            print("❌ No value bets found for this match.")


import flax.linen as nn
import jax
import jax.numpy as jnp

class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)  # win/draw/loss
        return x

def normalize_jax(x, axis=1):
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / norm

def implied_prob(odds):
    return round(1 / odds, 4) if odds else 0

def is_value_bet(model_prob, bookmaker_odds, margin=0.05):
    return model_prob > implied_prob(bookmaker_odds) + margin

def generate_tip(pred_probs, odds, threshold=0.6):
    outcomes = ['home_win', 'draw', 'away_win']
    tips = []
    for i, prob in enumerate(pred_probs):
        if prob > threshold and is_value_bet(prob, odds[i]):
            tips.append({
                'bet': outcomes[i],
                'confidence': round(prob, 3),
                'odds': odds[i],
                'value_margin': round(prob - implied_prob(odds[i]), 3)
            })
    return tips

