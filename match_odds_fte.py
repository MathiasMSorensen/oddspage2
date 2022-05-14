
from autokeras import StructuredDataClassifier
from tensorflow.keras.models import load_model
from dicts import fte_odds_dict
from getOdds import best_odds, best_oddsOU
from tensorflow.python.keras.layers import LayerNormalization
from missingpy import MissForest
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import get_close_matches
import datetime
from utils import *
from utils import kelly_and_provider, make_final_data, impute, Kelly_criteria, betting_function,  bookie_yield, odds_date, avg_points, stats_last_x_H2H, stats_last_x_matches, last_x_H2H, last_x_matches, betting_function_backtest
from pathlib import Path
import sys
from tensorflow import keras
import joblib
from sklearn.ensemble import RandomForestClassifier
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
import joblib
import os
from sklearn.model_selection import train_test_split
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
# pd.set_option('display.max_rows', 1000)

best_odds = pd.concat([best_odds, best_oddsOU], axis=0)


url = "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv"
fte = pd.read_csv(url)

fte[fte.league.isin(best_odds.league.unique())]
fte.date = pd.to_datetime(fte.date).dt.date

best_odds.time = pd.to_datetime(best_odds.time).dt.date

best_odds = best_odds.reset_index(drop=True)

# Put fte names to odds names
best_odds['fte_home'] = 0
best_odds['fte_away'] = 0

for i in range(len(best_odds)):
    if best_odds.league[i] != 'Mexican Primera Division Torneo Clausura':
        best_odds.loc[i, 'fte_home'] = fte_odds_dict[best_odds.home_team[i]]
        best_odds.loc[i, 'fte_away'] = fte_odds_dict[best_odds.away_team[i]]
        print(i)

odds_fte_df = pd.merge(best_odds, fte[((fte['date'] >= min(best_odds['time'])) & (fte['date'] < datetime.date(2022, 1, 1)))], left_on=['fte_home', 'fte_away'],
                       right_on=['team1', 'team2'],
                       how='left')
print(odds_fte_df.isna().sum())
odds_fte_dfOU = odds_fte_df[odds_fte_df['2_provider'].isna()]
odds_fte_df = odds_fte_df[odds_fte_df['O25_provider'].isna()]

rel_columns = ['season', 'date', 'league_id', 'league_y', 'home_team', 'away_team', 'team1', 'team2', 'spi1',
               'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2',
               'importance1', 'importance2', 'score1', 'score2', 'xg1', 'xg2',
               'nsxg1', 'nsxg2', 'adj_score1', 'adj_score2', '1', 'X', '2']

odds_fte_df = odds_fte_df[rel_columns]
odds_fte_df = odds_fte_df.rename(
    columns={'league_y': 'league', '1': 'mH', 'X': 'mD', '2': 'mA'})
odds_fte_df['Bookie_yield'] = 1 / \
    (1/odds_fte_df['mH']+1/odds_fte_df['mD']+1/odds_fte_df['mA'])

# Merge with old data and make variables to use for backward stuff
rel_fte_col = ['season', 'date', 'league_id', 'league', 'team1', 'team2', 'spi1',
               'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2',
               'importance1', 'importance2', 'score1', 'score2', 'xg1', 'xg2',
               'nsxg1', 'nsxg2', 'adj_score1', 'adj_score2']

fte = fte[fte['date'] < min(best_odds['time'])]
fte = fte[(fte['date'] > datetime.date(2019, 1, 1))]
fte[['FTAG', 'FTHG', 'FTR', 'mH', 'mD', 'mA', 'Bookie_yield']] = 0
fte = fte[rel_fte_col]

FTR = np.zeros(len(fte), str)
FTR[fte['score1'] > fte['score2']] = 'H'
FTR[fte['score1'] == fte['score2']] = 'D'
FTR[fte['score1'] < fte['score2']] = 'A'
fte['FTR'] = FTR

Final_df = fte.reset_index(drop=True)

print(Final_df.isna().sum())

# Impute missing values using historical data

DF = pd.concat([Final_df.reset_index(drop=True),
               odds_fte_df.reset_index(drop=True)]).reset_index(drop=True)

DF_temp = DF[['home_team', 'away_team']]

DF = impute(DF.drop(['home_team', 'away_team'], axis=1))

# Remove all columns which we cannot bet on
odds_fte_df = DF[DF['mH'] > 0.5]

print(odds_fte_df.isna().sum())

# Make final stuff
Final_df, odds_fte_df = make_final_data(odds_fte_df, Final_df)

print(Final_df.isna().sum())
Final_df = Final_df.dropna()

# clean up
del(DF)

# Merge OU table such that we can play on these odds as well.
odds_fte_df[['home_team', 'away_team']
            ] = DF_temp[DF_temp['home_team'].isna() == False].reset_index(drop=True)
final_df_temp = odds_fte_df[['home_team', 'away_team']]
final_df_temp = final_df_temp[odds_fte_df.index.isin(Final_df.index)]
final_df_temp_temp = pd.concat([final_df_temp, Final_df], axis=1)

final_dfOU = pd.merge(best_oddsOU.reset_index(drop=True), final_df_temp_temp.reset_index(drop=True), left_on=['home_team', 'away_team'],
                      right_on=['home_team', 'away_team'],
                      how='left')

print(final_dfOU.isna().sum())

final_dfOU = final_dfOU.dropna()
# %% Let's play
# First we play 1X2
col_order = list(pd.read_csv('col-order')['0'])
col_orderOU = list(pd.read_csv('col-orderOU')['0'])

RF = RandomForestClassifier()
RF = joblib.load("Random_forest.joblib")
YprobRF = pd.DataFrame(RF.predict_proba(Final_df[col_order]))

# Neural Network
# Number of epochs:
epochs = 250
# Number of neurons per layer:
neurons = 10
# Kernel initializer
kernel_ini = 'glorot_normal'
# Leaky_ReLU alpha
lrelu_a = 0.4

Xtrain = Final_df[col_order]
NN = Sequential()
# Adding layers:
NN.add(
    Dense(neurons, input_dim=Xtrain.shape[1], kernel_initializer=kernel_ini))
NN.add(LeakyReLU(alpha=lrelu_a))
# Add dropout:
NN.add(Dropout(0.1, input_shape=(Xtrain.shape[1],)))

NN.add(Dense(neurons, kernel_initializer=kernel_ini))
NN.add(LeakyReLU(alpha=lrelu_a))

NN.add(Dense(neurons, kernel_initializer=kernel_ini))
NN.add(LeakyReLU(alpha=lrelu_a))

NN.add(Dense(neurons, kernel_initializer=kernel_ini))
NN.add(LeakyReLU(alpha=lrelu_a))

NN.add(Dense(neurons, kernel_initializer=kernel_ini))
NN.add(LeakyReLU(alpha=lrelu_a))

NN.add(Dense(neurons, kernel_initializer=kernel_ini))
# Adding output layer:
NN.add(Dense(3, kernel_initializer=kernel_ini, activation='softmax'))

# Compiling model:
NN.compile(loss='categorical_crossentropy',
           optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-3),
           metrics=['categorical_crossentropy'])

# Load weights to the model
NN.load_weights("DNN.h5")

# Predict it
YprobNN = NN.predict(Final_df[col_order])

YprobNN = pd.DataFrame(YprobNN)

YprobNN = YprobNN[[2, 1, 0]]

YprobNN.columns = [0, 1, 2]

w_RF = 1/4
w_NN = 3/4
weight = 11
w_fte = 0
spread_limit = 0.075
# Make final prediction
Yprob = YprobRF*w_RF+YprobNN*w_NN

Probs = betting_function_backtest(Yprob, Final_df[col_order])

Probs = kelly_and_provider(best_odds, fte, Probs, weight)

# pd.concat([YprobRF,YprobNN],axis=1)
print("Number of bets"+" "+str(len(Probs[Probs['Bet'] != 'None'])))
print("Number of games"+" "+str(len(Probs)))
print(Probs[['time', 'league_x', 'fte_home', 'fte_away', 0, 'mH','mD','mA',
             'Hp_diff', 'Dp_diff', 'Ap_diff', 'Bet']][Probs['Bet'] != 'None'].sort_values(by='time'))

# AboveBelow25

RF = RandomForestClassifier()
RF = joblib.load("RF_OU.joblib")
YprobRF = pd.DataFrame(RF.predict_proba(final_dfOU[col_orderOU]))

NN = load_model("model_autokeras")
YprobNNOU = pd.DataFrame(NN.predict(final_dfOU[col_orderOU]))
YprobNNOU[1] = YprobNNOU
YprobNNOU[0] = 1 - YprobNNOU[1]
spread_limit = 0.075

YprobOU = YprobRF*w_RF+YprobNNOU*w_NN

ProbsOU = betting_function_backtest25(YprobOU, final_dfOU[col_orderOU])

ProbsOU = kelly_and_provider25(best_odds, fte, ProbsOU, weight)

ProbsOU['Bet'] = np.where(ProbsOU['Bet'] == 1, 'O25', ProbsOU['Bet'])
ProbsOU['Bet'] = np.where(ProbsOU['Bet'] == 0, 'U25', ProbsOU['Bet'])

print("Number of bets"+" "+str(len(ProbsOU[ProbsOU['Bet'] != 'None'])))
print("Number of games"+" "+str(len(ProbsOU)))
print(ProbsOU[['time', 'league_x', 'fte_home', 'fte_away', 0,
               'U25p_diff', 'O25p_diff', 'Bet']][ProbsOU['Bet'] != 'None'].sort_values(by='time'))
