import argparse
from datetime import datetime, timedelta
import pandas as pd
import jax.numpy as jnp
from colorama import Fore, Style


# football data ---
# use these free football data providers and utilities


import requests

# Get all competitions
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

import pandas as pd

url = "http://api.clubelo.com/fixtures"
matches = pd.read_csv(url)
print(matches.head())

import pandas as pd

url = "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv"
matches = pd.read_csv(url)
print(matches.head())

#create a dictionary for football teams 
# team_names.py
standard_teams = [
    "Manchester United",
    "Chelsea",
    "Arsenal",
    "Liverpool",
    "Manchester City",
    "Tottenham Hotspur",
    # ...add all teams you expect in your data
]

# Create a mapping from team name to index for use in to_data_frame and elsewhere
football_team_index_current = {team: idx for idx, team in enumerate(standard_teams)}

#use fuzzy matching to match team names from different sources
from fuzzywuzzy import process

def standardize_team_name(name):
    match, score = process.extractOne(name, standard_teams)
    return match if score > 80 else name  # Adjust threshold as needed



def get_json_data(url):
    """
    Fetch JSON data from a REST API endpoint.
    """
    print(f"Fetching data from {url}...")
    # This should return a dictionary similar to football_team_index_current
    return {}

def to_data_frame(data):
    """
    TODO: Implement this function to convert the JSON response from your football API
    into a pandas DataFrame with team stats.
    This is a placeholder.
    """
    # Create a dummy DataFrame for demonstration
    teams = list(football_team_index_current.keys())
    dummy_stats = {
        'TEAM_ID': [i for i in range(len(teams))],
        'TEAM_NAME': teams,
        'W': [28, 26, 23, 20, 19, 18, 15, 16],
        'L': [5, 6, 9, 10, 11, 12, 15, 14],
        'PTS': [94, 84, 75, 70, 67, 64, 60, 62],
        # Add other relevant football stats (e.g., Goals For, Goals Against, etc.)
    }
    # Ensure the dummy data matches the number of teams
    num_teams = len(teams)
    for key in dummy_stats:
        if len(dummy_stats[key]) < num_teams:
            dummy_stats[key].extend([dummy_stats[key][-1]] * (num_teams - len(dummy_stats[key])))

    return pd.DataFrame(dummy_stats)


def create_todays_games(data):
    """
    TODO: Implement this function to parse the API response for today's games
    and return a list of [home_team, away_team] pairs.
    This is a placeholder.
    """
    return [['Manchester City', 'Liverpool'], ['Arsenal', 'Chelsea']]

def create_todays_games_from_odds(odds):
    """
    Creates a list of games from the odds dictionary.
    """
    return [game.split(':') for game in odds.keys()]


def create_todays_games_data(games, df, odds):
    """
    Prepares the data for today's games for prediction.
    """
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

        # Get odds
        if odds:
            game_odds = odds.get(f"{home_team}:{away_team}")
            if not game_odds:
                print(f"Odds not found for {home_team} vs {away_team}")
                continue
            todays_games_uo.append(game_odds.get('under_over_odds', 2.5)) # Default U/O
            home_team_odds.append(game_odds[home_team]['money_line_odds'])
            away_team_odds.append(game_odds[away_team]['money_line_odds'])
        else:
            # Manual input if odds provider fails
            todays_games_uo.append(float(input(f"{home_team} vs {away_team} (U/O): ")))
            home_team_odds.append(float(input(f"{home_team} odds: ")))
            away_team_odds.append(float(input(f"{away_team} odds: ")))

        # Calculate days rest
        try:
            # TODO: Make sure the CSV file path and column names are correct for your football data
            schedule_df = pd.read_csv('Data/football-2024-UTC.csv', parse_dates=['Date'], dayfirst=True)
            home_games = schedule_df[(schedule_df['HomeTeam'] == home_team) | (schedule_df['AwayTeam'] == home_team)]
            away_games = schedule_df[(schedule_df['HomeTeam'] == away_team) | (schedule_df['AwayTeam'] == away_team)]
            
            previous_home_games = home_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date', ascending=False).head(1)
            previous_away_games = away_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date', ascending=False).head(1)

            home_days_off = (datetime.today() - previous_home_games.iloc[0]['Date']) if not previous_home_games.empty else timedelta(days=7)
            away_days_off = (datetime.today() - previous_away_games.iloc[0]['Date']) if not previous_away_games.empty else timedelta(days=7)
        except FileNotFoundError:
            print("Schedule file 'Data/football-2024-UTC.csv' not found. Using default 7 days rest.")
            home_days_off, away_days_off = timedelta(days=7), timedelta(days=7)

        home_team_days_rest.append(home_days_off.days)
        away_team_days_rest.append(away_days_off.days)

        # Combine stats
        home_team_series = df.iloc[football_team_index_current.get(home_team)]
        away_team_series = df.iloc[football_team_index_current.get(away_team)]
        stats = pd.concat([home_team_series, away_team_series])
        stats['Days-Rest-Home'] = home_days_off.days
        stats['Days-Rest-Away'] = away_days_off.days
        match_data.append(stats)

    if not match_data:
        return None, None, None, None, None

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1).T
    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
    data = frame_ml.values.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def main():
    odds = None
    if args.odds:
        # Dummy SbrOddsProvider implementation for demonstration
        class SbrOddsProvider:
            def __init__(self, sportsbook):
                self.sportsbook = sportsbook
                self.sport = 'soccer'
            def get_odds(self):
                # Return a dummy odds dictionary for demonstration
                return {
                    "Manchester City:Liverpool": {
                        "Manchester City": {"money_line_odds": 1.8},
                        "Liverpool": {"money_line_odds": 2.1},
                        "under_over_odds": 2.5
                    },
                    "Arsenal:Chelsea": {
                        "Arsenal": {"money_line_odds": 1.9},
                        "Chelsea": {"money_line_odds": 2.0},
                        "under_over_odds": 2.5
                    }
                }
        odds_provider = SbrOddsProvider(sportsbook=args.odds)
        odds_provider.sport = 'soccer' # Example of how you might adapt it
        odds = odds_provider.get_odds()
        games = create_todays_games_from_odds(odds)
        if not games:
            print("No games found from the odds provider.")
            return
        # Verification
        if (games[0][0] + ':' + games[0][1]) not in odds:
            print(Fore.RED, "Odds data may be outdated or unavailable for today's games.", Style.RESET_ALL)
            odds = None
        else:
            print(f"------------------{args.odds} odds data------------------")
            for g, game_odds in odds.items():
                home, away = g.split(":")
                print(f"{away} ({game_odds[away]['money_line_odds']}) @ {home} ({game_odds[home]['money_line_odds']})")
    
    if not odds:
        # Fallback to a different source if odds provider fails or is not specified
        todays_games_url = "https://example.com/api/todays_games"  # TODO: Replace with actual API endpoint
        json_data = get_json_data(todays_games_url)
        games = create_todays_games(json_data)

    # Get team stats
    data_url = "https://example.com/api/team_stats"  # TODO: Replace with actual API endpoint for team stats
    stats_json = get_json_data(data_url)
    stats_df = to_data_frame(stats_json)

    # Prepare data for models
    data, uo, ml_frame, home_odds, away_odds = create_todays_games_data(games, stats_df, odds)

    if data is None:
        print("Could not prepare data for today's games. Exiting.")
        return

    # Run models
    if args.nn:
        print("\n------------Neural Network Model Predictions-----------")
        norm_data = tf.keras.utils.normalize(data, axis=1)
        NN_Runner.nn_runner(norm_data, uo, ml_frame, games, home_odds, away_odds, args.kc)
        print("-------------------------------------------------------")
    if args.xgb:
        print("\n---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, uo, ml_frame, games, home_odds, away_odds, args.kc)
        print("-------------------------------------------------------")
    if args.A:
        print("\n---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, uo, ml_frame, games, home_odds, away_odds, args.kc)
        print("-------------------------------------------------------")
        print("\n------------Neural Network Model Predictions-----------")
        norm_data = tf.keras.utils.normalize(data, axis=1)
        NN_Runner.nn_runner(norm_data, uo, ml_frame, games, home_odds, away_odds, args.kc)
        print("-------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run for Football Betting')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-nn', action='store_true', help='Run with Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run all Models')
    parser.add_argument('-odds', help='Sportsbook to fetch from (e.g., fanduel, draftkings)')
    parser.add_argument('-kc', action='store_true', help='Calculates percentage of bankroll to bet')
    args = parser.parse_args()
    main()

def normalize_jax(x, axis=1):
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / norm

norm_data = normalize_jax(jnp.array(data), axis=1)