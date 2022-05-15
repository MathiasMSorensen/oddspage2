from dicts import fte_odds_dict
from getOdds import best_odds, best_oddsOU
from missingpy import MissForest
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import get_close_matches
import datetime
from utils import *
from utils import betting_function_backtest, fte_data, kelly_and_provider, make_final_data, impute, Kelly_criteria, betting_function,  bookie_yield, odds_date, avg_points, stats_last_x_H2H, stats_last_x_matches, last_x_H2H, last_x_matches, betting_function_backtest
from pathlib import Path
import sys
import joblib
import urllib.request
import json
import os
import ssl
import joblib
import os
# pd.set_option('display.max_rows', 1000)

best_odds = pd.concat([best_odds, best_oddsOU], axis=0)

DF, Final_df = fte_data(best_odds)
DF = impute(DF.drop(['home_team', 'away_team'], axis=1))

# Remove all columns which we cannot bet on
odds_fte_df = DF[DF['mH'] > 0.5]

# Make final stuff
Final_df, odds_fte_df = make_final_data(odds_fte_df, Final_df)
Final_df = Final_df.dropna()

col_order = pd.read_csv('col_order')['0'].to_list()
for i in ['league', 'team1', 'team2']:
    Final_df[i] = "example_value"
    
for i in ['season', 'date', 'score1', 'score2', 'xg1', 'xg2', 'nsxg1', 'nsxg2', 'adj_score1', 'adj_score2', 'Bookie_yield']:
    Final_df[i] = 0
    

js_file = json.loads(Final_df[col_order].to_json(orient='records'))

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

res_list = []
for i in range(len(js_file)):
    # Request data goes here
    data = {
        "Inputs": {
            "data":
            [
                js_file[i],
            ]
        },
        "GlobalParameters": {
            "method": "predict_proba"
        }
    }

    body = str.encode(json.dumps(data))

    url = 'http://4b417d5e-4f6b-460e-af46-016002eef6ff.northeurope.azurecontainer.io/score'
    api_key = 'rjdSNWldony8yApNb0oc2LVK8scyicEj' # Replace this with the API key for the web service
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        res_list.append(json.loads(result)['Results'][0])
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

Yprob = pd.DataFrame(res_list)
Yprob.columns = ['A_p', 'D_p', 'H_p']
handicap = 0.05
Probs = betting_function_backtest(Yprob, Final_df[col_order], handicap)

weight = 0.1
Probs = kelly_and_provider(best_odds, fte, Probs, weight)

# pd.concat([YprobRF,YprobNN],axis=1)
print("Number of bets"+" "+str(len(Probs[Probs['Bet'] != 'None'])))
print("Number of games"+" "+str(len(Probs)))
print(Probs[['time', 'league_x', 'fte_home', 'fte_away','mH','mD','mA','H_p', 'D_p','A_p',
             'Hp_diff', 'Dp_diff', 'Ap_diff', 'Bet']][Probs['Bet'] != 'None'].sort_values(by='time'))

# AboveBelow25

# clean up
# del(DF)
# df_temp = pd.read_excel('1_Training-data.xlsx')
# # Merge OU table such that we can play on these odds as well.
# odds_fte_df[['home_team', 'away_team']
#             ] = DF_temp[DF_temp['home_team'].isna() == False].reset_index(drop=True)
# final_df_temp = odds_fte_df[['home_team', 'away_team']]
# final_df_temp = final_df_temp[odds_fte_df.index.isin(Final_df.index)]
# final_df_temp_temp = pd.concat([final_df_temp, Final_df], axis=1)

# final_dfOU = pd.merge(best_oddsOU.reset_index(drop=True), final_df_temp_temp.reset_index(drop=True), left_on=['home_team', 'away_team'],
#                       right_on=['home_team', 'away_team'],
#                       how='left')


print(Yprob)
# Rename correspondingly:
Yprob.columns = ['A_p', 'D_p', 'H_p']
# Set indices correctly:
Yprob = Yprob.set_index(Final_df.index.values)
# Generate Data Frame with implied probabilities:
odds_yield = 1/(1/Final_df[['mA','mD','mH']]).sum(axis=1)
Probs = pd.DataFrame(pd.concat([odds_yield/Final_df['mA'],odds_yield/Final_df['mD'],odds_yield/Final_df['mH']], axis=1))

Probs.columns = ['A_odds_p', 'D_odds_p', 'H_odds_p']
# Merge with estimated probabilites:
Probs = Probs.merge(right=Yprob, how='inner', left_index=True, right_index=True)

# Calculate difference between estimated and implied probability:
Probs['Hp_diff'] = Probs.H_p - Probs.H_odds_p
Probs['Dp_diff'] = Probs.D_p - Probs.D_odds_p
Probs['Ap_diff'] = Probs.A_p - Probs.A_odds_p
# Merge to get odds:
Probs = Probs.merge(right=Final_df[['mH', 'mD', 'mA']], how='inner',
                    left_index=True, right_index=True)

print(Probs)
Probs['Bet'] = Probs.apply(lambda y : bet(y, 0.025), axis=1)





max(Probs['Hp_diff'], Probs['Dp_diff'], Probs['Ap_diff'])   

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
