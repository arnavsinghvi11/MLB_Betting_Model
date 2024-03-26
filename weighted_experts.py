import datetime
import os
import random
import re
import requests
import time
import warnings
import numpy as np
import pandas as pd
import pickle
import regex
import tensorflow as tf
import threading
from bs4 import BeautifulSoup
from pytz import timezone
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from unidecode import unidecode
from webdriver_manager.chrome import ChromeDriverManager
from xgboost import XGBRegressor
import xgboost as xgb

import model
import evaluator
import date
import box_score
import helper_functions

warnings.filterwarnings("ignore")

eta = 0.1
cost_threshold = 0.5
expert_names = ['optimizer_preds', 'mlp_preds', 'dnn_preds', 'xgb_preds', 'lstm_preds', 'rf_preds']
num_experts = len(expert_names)
window_size = 10
past_x_games = 5
directory = '2023_Season/MLB-Bets-Multiplicative-Weights/' #directory to be changed
hitting_file_suffixes = ['H', 'HR', 'RBI', 'R', 'H+R+RBI', 'BB', 'TB']
pitching_file_suffixes = ['SO', 'ER', 'HA', 'BBA', 'PO']

hitting_stats_dict = {
    'H': ['1B', '2B', '3B', 'AB', 'BB', 'HR', 'OBP', 'OPS', 'PA', 'SLG', 'SO', 'TB', 'WPA', 'cWPA', 'aLI', 'acLI'],
    'HR': ['AB', 'PA', 'SLG', 'TB', 'WPA', 'aLI', 'acLI', 'cWPA', 'OPS'],
    'RBI': ['1B', '2B', '3B', 'AB', 'H', 'HR', 'OBP', 'PA', 'RE24', 'SLG', 'TB', 'WPA', 'aLI', 'acLI', 'cWPA'],
    'R': ['1B', '2B', '3B', 'AB', 'BB', 'H', 'HR', 'OBP', 'PA', 'RBI', 'RE24', 'SLG', 'WPA', 'cWPA', 'aLI', 'acLI'],
    'H+R+RBI': ['1B', '2B', '3B', 'AB', 'H', 'HR', 'OBP', 'OPS', 'PA', 'SLG', 'TB', 'WPA', 'cWPA', 'aLI', 'acLI'],
    'BB': ['OBP', 'PA', 'R', 'SO', 'WPA', 'cWPA', 'aLI', 'acLI',],
    'TB': ['1B', '2B', '3B', 'AB', 'H', 'HR', 'OPS', 'PA', 'SLG', 'WPA', 'cWPA', 'aLI', 'acLI']
}


pitching_stats_dict = {
    'SO': ['IP', 'Pit', 'Str', 'StL', 'StS', 'FB', 'GB', 'GSc', 'ERA', 'RE24', 'WPA', 'aLI', 'acLI', 'PO'],
    'ER': ['BB', 'FB', 'GB', 'GSc', 'H', 'HR', 'IP', 'ERA', 'Pit', 'RE24', 'WPA', 'aLI', 'acLI', 'cWPA', 'PO'],
    'HA': ['Ctct', 'FB', 'GB', 'GSc', 'H', 'HR', 'IP', 'ERA', 'Pit', 'RE24', 'Str', 'WPA', 'aLI', 'acLI', 'cWPA', 'PO'],
    'BBA': ['ERA', 'GSc', 'IP', 'Pit', 'RE24', 'StL', 'Str', 'WPA', 'aLI', 'acLI', 'cWPA'],
    'PO': ['IP', 'GSc', 'ERA', 'RE24', 'WPA', 'aLI', 'acLI', 'BB', 'H', 'HR', 'SO', 'FB', 'GB', 'LD', 'Pit', 'StL', 'StS', 'Ctct']
}

combo_stats = ['H+R+RBI']
models = ["DNN", "MLP", "RF", "XGB", "LSTM"]

start_date = datetime.date(2023, 3, 29)
end_date = datetime.date(2023, 9, 30)
date_range = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]

month_ends = {}
for date in date_range:
    month = date.month
    last_day = date.strftime('%d')
    month_ends[month] = int(str(month) + last_day)
month_order = [3, 4, 5, 6, 7, 8, 9]
month_ends

def load_files(directory, suffixes, prefix):
    data = {}
    for suffix in suffixes:
        with open(f'{directory}{prefix}{suffix}.pkl', 'rb') as f:
            data[suffix] = pickle.load(f)
    return data

import date
date_tracker = date.Date()
# directory_month, directory_day = date_tracker.date_month_day(-1)
# print(directory_month, directory_day)
month, day = date_tracker.date_month_day(352)
current_month, current_day = month, day
# directory_base = f'{directory}{directory_month}/{directory_day}/'
#TODO - REMOVE THIS LOGIC AFTER 1st one 
hitting_expert_weights, hitting_expert_history = {}, {}
for stat in hitting_stats_dict:
    hitting_expert_weights[stat] = np.ones(num_experts)
    hitting_expert_history[stat] = np.zeros((num_experts, window_size))
pitching_expert_weights, pitching_expert_history = {}, {}
for stat in pitching_stats_dict:
    pitching_expert_weights[stat] = np.ones(num_experts)
    pitching_expert_history[stat] = np.zeros((num_experts, window_size))
# hitting_expert_weights = load_files(directory_base, hitting_file_suffixes, 'expert_weights_')
# hitting_expert_history = load_files(directory_base, hitting_file_suffixes, 'expert_history_')
# pitching_expert_weights = load_files(directory_base, pitching_file_suffixes, 'expert_weights_')
# pitching_expert_history = load_files(directory_base, pitching_file_suffixes, 'expert_history_')

print(hitting_expert_weights)
print(hitting_expert_history)
print(pitching_expert_weights)
print(pitching_expert_history)
date_counter = int(day) +1
date_counter2 = str(month) + str(date_counter - 1)
print('this is date_counter2')
print(date_counter2)

print('this is date_counter')
print(date_counter)
date_counter -= 1
month_folder = os.path.join(directory, str(month))
if not os.path.exists(month_folder):
    os.makedirs(month_folder)
day_folder = os.path.join(month_folder, str(date_counter))
if not os.path.exists(day_folder):
    os.makedirs(day_folder)
model_predictions_dict = {}

month, date_counter, date_counter2


#TODO check dates for this
import datetime
from datetime import datetime
from datetime import timedelta
date_format = '%m/%d/%Y %H:%M:%S %Z'
date = datetime.now(timezone('US/Pacific'))
date = date.astimezone(timezone('US/Pacific'))
today_date = date.strftime(date_format)
today = today_date.split('/')[0] + '/' + date.strftime(date_format).split('/')[1]
today_date = today_date.split(' ')[0]
month_name = datetime.strptime(today.split('/')[0].strip(), "%m").strftime("%B").lower()
month = today.split('/')[0]
day = today.split('/')[1]
action_date = month_name[:3].upper() + ' ' + day
time_delta = timedelta(days=1)
while not (month == '03' and day == '30'):
    date -= time_delta
    today_date = date.strftime(date_format)
    today = today_date.split('/')[0] + '/' + date.strftime(date_format).split('/')[1]
    month = today.split('/')[0]
    day = today.split('/')[1]

all_box_score_results = pd.DataFrame()
while not (str(month.lstrip("0")) == current_month and str(day.lstrip("0")) == current_day):
    hitting_box_type = 'Hitting'
    hitting_columns = ['Date','Name','Team', 'Opponent', 'Hmcrt_adv', 'AB', 'R', 'H', 'RBI', 'BB', 'SO', 'PA', 'TB', 'HRs']
    past_box_score_hitting = pd.read_csv(f"MLB-Box-Score-Results-UPDATED/{hitting_box_type}/" + f"MLB-Bets-{hitting_box_type}-Box-Score-Results-" + month.lstrip('0') + "-" + day.lstrip('0') +".csv")
    past_box_score_hitting['Date'] = month + "-" + day
    past_box_score_hitting['Name'] = past_box_score_hitting['Name'].apply(unidecode)
    past_box_score_hitting['Name'] = past_box_score_hitting['Name'].apply(helper_functions.abbrv)
    past_box_score_hitting['Date'] = past_box_score_hitting['Date'].apply(date.date_converter)
    all_box_score_results_hitting = all_box_score_results_hitting.append(past_box_score_hitting)
    all_box_score_results_hitting = all_box_score_results_hitting.dropna(subset=['Date'])

    all_box_score_results_hitting['Date'] = all_box_score_results_hitting['Date'].astype('int')
    all_box_score_results_hitting = all_box_score_results_hitting.sort_values(by=['Date'], ascending = False)
    all_box_score_results_hitting.set_index(['Name','Team']).index.factorize()[0]+1
    all_box_score_results_hitting = all_box_score_results_hitting.drop_duplicates(hitting_columns, keep='last')
    
    
    pitching_box_type = 'Pitching'
    pitching_columns = ['Date','Name','Team', 'Opponent', 'Hmcrt_adv', 'IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR']
    past_box_score_pitching = pd.read_csv(f"MLB-Box-Score-Results-UPDATED/{pitching_box_type}/" + f"MLB-Bets-{pitching_box_type}-Box-Score-Results-" + month.lstrip('0') + "-" + day.lstrip('0') +".csv")
    past_box_score_pitching['Date'] = month + "-" + day
    past_box_score_pitching['Name'] = past_box_score_pitching['Name'].apply(unidecode)
    past_box_score_pitching['Name'] = past_box_score_pitching['Name'].apply(helper_functions.abbrv)
    past_box_score_pitching['Date'] = past_box_score_pitching['Date'].apply(date.date_converter)
    all_box_score_results_pitching = all_box_score_results_pitching.append(past_box_score_pitching)
    all_box_score_results_pitching = all_box_score_results_pitching.dropna(subset=['Date'])

    all_box_score_results_pitching['Date'] = all_box_score_results_pitching['Date'].astype('int')
    all_box_score_results_pitching = all_box_score_results_pitching.sort_values(by=['Date'], ascending = False)
    all_box_score_results_pitching.set_index(['Name','Team']).index.factorize()[0]+1
    all_box_score_results_pitching = all_box_score_results_pitching.drop_duplicates(pitching_columns, keep='last')
    
    
    date += time_delta
    today_date = date.strftime(date_format)
    today = today_date.split('/')[0] + '/' + date.strftime(date_format).split('/')[1]
    month = today.split('/')[0]
    day = today.split('/')[1]


all_box_score_results_hitting_past_eval = all_box_score_results_hitting
all_box_score_results_hitting_past_eval.insert(loc=0, column='player_id', value = all_box_score_results_hitting_past_eval.set_index(['Name','Team']).index.factorize()[0]+1)
all_box_score_results_hitting_past_eval = all_box_score_results_hitting_past_eval.sort_values(by = ['Date', 'player_id'], ascending = [True, True])
all_box_score_results_hitting_past_eval

all_box_score_results_pitching_past_eval = all_box_score_results_pitching
all_box_score_results_pitching_past_eval.insert(loc=0, column='player_id', value = all_box_score_results_pitching_past_eval.set_index(['Name','Team']).index.factorize()[0]+1)
all_box_score_results_pitching_past_eval = all_box_score_results_pitching_past_eval.sort_values(by = ['Date', 'player_id'], ascending = [True, True])
all_box_score_results_pitching_past_eval


import date
date_tracker = date.Date()
#TODO need to update back after 1st one
month, day = date_tracker.date_month_day(352)
date_counter = int(day) +1
month_order = month_order[month_order.index(int(month)):]
month = int(month)
last_day = 31

date_counter -= 1
month_folder = os.path.join(directory, str(month))
if not os.path.exists(month_folder):
    os.makedirs(month_folder)
day_folder = os.path.join(month_folder, str(date_counter))
if not os.path.exists(day_folder):
    os.makedirs(day_folder)
month, date_counter, date_counter2


def process_csv(filename, prop_type):
    data = pd.read_csv(filename)
    data = data.loc[data['Play'].str.contains(prop_type, regex=False)]
    if '+' not in prop_type:
        data = data.loc[~data['Play'].str.contains(r'\+')]
    return data

def weighted_optimizer(folder_path, expert_names, expert_weights, prop_type, all_predictions_dfs):
    expert_to_df_map = {expert_name: all_predictions_dfs[i] for i, expert_name in enumerate(expert_names)}
    if len(set(expert_weights)) == 1:
        best_experts = random.sample(expert_names, 2)
    else:
        indices = np.array(expert_weights).argsort()[-2:][::-1]
        best_experts = [expert_names[index] for index in indices]
    major1 = expert_to_df_map[best_experts[0]]
    major2 = expert_to_df_map[best_experts[1]]
    if not os.path.exists(folder_path + '/optimizations'):
        os.makedirs(folder_path + '/optimizations')
    expert_to_df_map = {expert_name: df for expert_name, df in zip(expert_names, all_predictions_dfs)}
    if len(set(expert_weights)) == 1:
        best_experts = random.sample(expert_names, 2)
    else:
        indices = np.array(expert_weights).argsort()[-2:][::-1]
        best_experts = [expert_names[index] for index in indices]
    major1 = expert_to_df_map[best_experts[0]]
    major2 = expert_to_df_map[best_experts[1]]
    unnamed_columns_major1 = [col for col in major1.columns if col.startswith('Unnamed:')]
    major1 = major1.drop(columns=unnamed_columns_major1)
    unnamed_columns_major2 = [col for col in major2.columns if col.startswith('Unnamed:')]
    major2 = major2.drop(columns=unnamed_columns_major2)
    int_columns = ['Odds', 'Hmcrt_adv']
    float_columns = ['Units', 'Payout', 'Profit']
    major1[int_columns] = major1[int_columns].apply(pd.to_numeric, errors='coerce')
    major1[float_columns] = major1[float_columns].astype(float)
    major2[int_columns] = major2[int_columns].apply(pd.to_numeric, errors='coerce')
    major2[float_columns] = major2[float_columns].astype(float)
    intersection = pd.merge(major1, major2, how='inner').drop_duplicates()
    intersection.to_csv(folder_path + '/optimizations/' + 'intersection_' + str(prop_type) + '.csv')
    combined = pd.concat([major1, major2], axis=0).drop_duplicates()
    combined.to_csv(folder_path + '/optimizations/' + 'combined_' + str(prop_type) + '.csv')
    return intersection, combined

def get_model_names_from_predictions(predictions, prop_type):
    return [col.split('_')[1] for col in predictions.columns if f"PlayerPredictions_" in col and prop_type in col]

def get_prediction(predictions, model_name, prop_type, player_name, team):
    filtered_df = predictions[(predictions['PlayerName'] == player_name) &
                              (predictions['PlayerTeam'] == team)]
    if not filtered_df.empty:
        return float(filtered_df[f'PlayerPredictions_{model_name}_{prop_type}'].values[0][0])
    else:
        return 0.0

def models_predictions_eval(folder_path, predictions, optimized, unoptimized, prop_type):
    model_names = get_model_names_from_predictions(predictions, prop_type)
    preds_dict = {name: [] for name in model_names}
    optimizer_preds = []
    for play, team in zip(unoptimized['Play'].values, unoptimized['Teams'].values):
        player_name = play.split(' ')[0]
        bet_value = float(play.split(' ')[1][1:])
        over_under = play.split(' ')[1][0]
        if play not in optimized['Play'].values:
            optimizer_preds.append(0)
        else:
            optimizer_preds.append(1)
        for model_name in model_names:
            prediction = get_prediction(predictions, model_name, prop_type, player_name, team)
            is_over = int(prediction > bet_value)
            is_under = int(prediction < bet_value)
            preds_dict[model_name].append(is_over if over_under == 'o' else is_under)
    prediction_path = os.path.join(folder_path, 'predictions')
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)
    predictions_df_dict = {}
    for model_name in model_names:
        predictions_df_dict[model_name] = unoptimized[[bool(x) for x in preds_dict[model_name]]]
        predictions_df_dict[model_name].to_csv(os.path.join(prediction_path, f'{model_name}_{prop_type}.csv'))
    all_preds = [optimizer_preds] + list(preds_dict.values())
    all_predictions_dfs = [optimized] + list(predictions_df_dict.values())
    return all_preds, all_predictions_dfs

def preprocess_model(past_data, prop_type, features_set):
    info_data = past_data[features_set]
    info_data = info_data.dropna()
    features = list(info_data.columns)
    features.remove(prop_type.lower())
    target = [prop_type.lower()]
    X = info_data.loc[:, features].values
    y = info_data.loc[:, target].values
    return train_test_split(X, y, test_size=0.1, random_state=1)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def predict_model(model, player_last_avgs, player_preds):
    with open(os.devnull, 'w') as devnull:
        # sys.stdout = devnull
        player_pred = model.predict(player_last_avgs)
        # sys.stdout = sys.__stdout__
        player_preds.append(player_pred)

def define_models(X_train, y_train):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(X_train))
    model_defs = {
        'DNN': Sequential([
            normalizer,
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(1)
        ]),
        'MLP': MLPRegressor(hidden_layer_sizes=(100,), solver='adam', random_state=42, max_iter=5000),
        'RF': RandomForestRegressor(n_estimators=200, random_state=42),
        'XGB': XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05,
                            max_depth=8, subsample=0.9, colsample_bytree=0.9, random_state=42, verbosity=0),
        'LSTM': Sequential([
            LSTM(128, input_shape=(X_train.shape[1], 1)),
            Dense(1)
        ])
    }
    model_defs['DNN'].compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model_defs['LSTM'].compile(optimizer='adam', loss='mse')
    threads = {name: threading.Thread(target=train_model, args=(model, X_train, y_train)) for name, model in model_defs.items()}
    [thread.start() for thread in threads.values()]
    [thread.join() for thread in threads.values()]
    return model_defs

def last_x_avgs(data, prop_type, features_set, x):
    last_x_games = data[-x:]
    last_x_stats = last_x_games[features_set].mean(axis=0).to_frame().T
    last_x_stats = last_x_stats.drop(prop_type, axis=1)
    return last_x_stats

def models_eval(past_data, past_x_games, prop_type, features_set, trained_models):
    player_data_list = []
    player_preds = {name: [] for name in trained_models}
    for i in past_data['player_id'].unique():
        player_data = past_data[past_data['player_id'] == i].fillna(0)
        player_last_avgs = last_x_avgs(player_data, prop_type.lower(), features_set, past_x_games)
        player_data_list.append({
            'PlayerName': player_data['name'].values[0],
            'PlayerTeam': player_data['team'].values[0],
            'PlayerLastAvgs': player_last_avgs
        })
    for pdata in player_data_list:
        threads = {model_name: threading.Thread(target=predict_model, args=(model, pdata['PlayerLastAvgs'], player_preds[model_name]))
                   for model_name, model in trained_models.items()}
        [thread.start() for thread in threads.values()]
        [thread.join() for thread in threads.values()]
    predictions = pd.DataFrame({
        'PlayerName': [pdata['PlayerName'] for pdata in player_data_list],
        'PlayerTeam': [pdata['PlayerTeam'] for pdata in player_data_list],
        **{f'PlayerPredictions_{name}_{prop_type}': preds for name, preds in player_preds.items()}
    })
    return predictions

def prop_prediction_with_experts(folder_path, month, date_counter, date_counter2, prop_type, features_set, past_data, past_x_games, expert_names, expert_weights_diff, expert_history_diff, cost_threshold, num_experts, window_size, eta, passed_in_data1=None):
    if passed_in_data1 is not None:
        models_predictions = passed_in_data1
    elif '+' in prop_type:
        models_predictions = passed_in_data1
    else:
        X_train, X_val, y_train, y_val = preprocess_model(past_data, prop_type, features_set)
        trained_models = define_models(X_train.astype(float), y_train.astype(float))
        models_predictions = models_eval(past_data, past_x_games, prop_type, features_set, trained_models)
    return models_predictions

for stat, attributes in hitting_stats_dict.items():
    file_stat = stat
    model_predictions_dict[stat] = prop_prediction_with_experts(
        day_folder, month, date_counter, date_counter2, stat, attributes, all_box_score_results_hitting_past_eval, past_x_games, expert_names,
        hitting_expert_weights[file_stat], hitting_expert_history[file_stat], cost_threshold, num_experts, window_size, eta
    )

for stat, attributes in pitching_stats_dict.items():
    file_stat = stat
    model_predictions_dict[stat] = prop_prediction_with_experts(
        day_folder, month, date_counter, date_counter2, stat, attributes, all_box_score_results_hitting_past_eval, past_x_games, expert_names,
        pitching_expert_weights[file_stat], pitching_expert_history[file_stat], cost_threshold, num_experts, window_size, eta
    )


for combo in combo_stats:
    stats = [stat for stat in combo.split('+')]
    if all([not model_predictions_dict[stat].empty for stat in stats]):
        merged_predictions = model_predictions_dict[stats[0]]
        for stat in stats[1:]:
            merged_predictions = pd.merge(merged_predictions, model_predictions_dict[stat], on=['PlayerName', 'PlayerTeam'], how='inner')
        for model in models:
            merged_predictions[f'PlayerPredictions_{model}_{combo}'] = sum([merged_predictions[f'PlayerPredictions_{model}_{stat}'] for stat in stats])
        model_predictions_dict[combo] = prop_prediction_with_experts(
            day_folder, month, date_counter, date_counter2, combo, [], all_box_score_results_hitting_past_eval, past_x_games, expert_names, hitting_expert_weights[combo],
            hitting_expert_history[combo], cost_threshold, num_experts, window_size, eta, merged_predictions)