#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 16:51:51 2021

@author: MadsJorgensen
"""

# Libraries:
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import get_close_matches
import datetime
from missingpy import MissForest

# Working directory:
wd = '/Users/MadsJorgensen/Dropbox (Personal)/Betting/FiveThirtyEight/Model training/Raw data/0_Football data/'

def bookie_yield(H, D, A):
    try:
        x = (1/H + 1/D + 1/A - 1)
    except ZeroDivisionError:
        x = np.nan
    return x

#%% Main leagues
# Read csv files and concatenate:
files = Path(wd + "Main/").rglob("*.csv")
main_csvs = [pd.read_csv(file, engine='python') for file in files]
main_csvs = pd.concat(main_csvs)
main_csvs = main_csvs.reset_index(drop=True)

# Find odds for as many matches as possible:
home = ['VCH', 'B365H', 'BWH', 'IWH', 'WHH', 'PSH']
draw = ['VCD', 'B365D', 'BWD', 'IWD', 'WHD', 'PSD']
away = ['VCA', 'B365A', 'BWA', 'IWA', 'WHA', 'PSA']
main_csvs['mH'] = main_csvs.apply(lambda x: max(x[home].dropna(), default=np.nan), axis=1)
main_csvs['mD'] = main_csvs.apply(lambda x: max(x[draw].dropna(), default=np.nan), axis=1)
main_csvs['mA'] = main_csvs.apply(lambda x: max(x[away].dropna(), default=np.nan), axis=1)

# Save relevant columns only:
main_csvs = main_csvs[['HomeTeam', 'AwayTeam', 'Date', 'Div', 'FTAG', 'FTHG', 'FTR', 'mH', 'mD', 'mA']]
# Caluclate bookie yield:
main_csvs['Bookie_yield'] = main_csvs.apply(lambda x: bookie_yield(x.mH, x.mD, x.mA), axis=1)


#%% Other leagues

# Read csv files and concatenate:
files = Path(wd + "Other/").rglob("*.csv")
other_csvs = [pd.read_csv(file, engine='python') for file in files]
other_csvs = pd.concat(other_csvs)
other_csvs = other_csvs.reset_index(drop=True)

# Determine odds:
home = ['PH', 'AvgH']
draw = ['PD', 'AvgD']
away = ['PA', 'AvgA']
other_csvs['mH'] = other_csvs.apply(lambda x: max(x[home].dropna(), default=np.nan), axis=1)
other_csvs['mD'] = other_csvs.apply(lambda x: max(x[draw].dropna(), default=np.nan), axis=1)
other_csvs['mA'] = other_csvs.apply(lambda x: max(x[away].dropna(), default=np.nan), axis=1)

# Save relevant columns only:
other_csvs = other_csvs[['Home', 'Away', 'Date', 'League', 'AG', 'HG', 'Res', 'mH', 'mD', 'mA']]
# Caluclate bookie yield:
other_csvs['Bookie_yield'] = other_csvs.apply(lambda x: bookie_yield(x.mH, x.mD, x.mA), axis=1)
# Rename columns:
other_csvs = other_csvs.rename(columns={'Home':'HomeTeam', 'Away':'AwayTeam', 'League':'Div', 'AG':'FTAG', 'HG':'FTHG', 'Res':'FTR'})

# %% All leagues

# Concatenate leagues:
Odds_df = pd.concat([main_csvs, other_csvs])

# Remove NAs (not accounting for missing odds):
Odds_df = Odds_df.dropna(subset=Odds_df.columns.values[:-4], axis=0)

Odds_df.to_csv("/Users/MadsJorgensen/Dropbox (Personal)/Betting/FiveThirtyEight/Model training/Raw data/0_Odds data.csv")


#%% Five Thirty Eight:

# Working directory:
wd = '/Users/MadsJorgensen/Dropbox (Personal)/Betting/FiveThirtyEight/Model training/Raw data/'

# FiveThirtyEight data:
FTE = pd.read_csv(wd + '0_FiveThirtyEight/soccer-spi/spi_matches.csv')

# Only keep relevant leagues:
Leagues = ['Austrian T-Mobile Bundesliga','Barclays Premier League','Belgian Jupiler League','Brasileiro S√©rie A','Danish SAS-Ligaen',
           'Dutch Eredivisie','English League Championship','English League One','English League Two','French Ligue 1','French Ligue 2',
           'German 2. Bundesliga','German Bundesliga','Italy Serie A','Italy Serie B','Major League Soccer',
           'Mexican Primera Division Torneo Apertura','Mexican Primera Division Torneo Clausura','Norwegian Tippeligaen','Portuguese Liga',
           'Russian Premier Liga','Scottish Premiership','Spanish Primera Division','Spanish Segunda Division','Swedish Allsvenskan',
           'Swiss Raiffeisen Super League','Turkish Turkcell Super Lig']
FTE = FTE[(FTE['league'].isin(Leagues))]

# Change to datetime format:
FTE['date'] = pd.to_datetime(FTE['date'], format='%Y-%m-%d')

def odds_date(obs):
    try:
        x = pd.to_datetime(obs['Date'], format='%d/%m/%y')
    except:
        x = pd.to_datetime(obs['Date'], format='%d/%m/%Y')
    return x
Odds_df['Date'] = Odds_df.apply(lambda x: odds_date(x), axis=1)
Odds_df = Odds_df.rename(columns={'Date':'date'})

# Correct team names:
team_names = {"Chievo": "Verona", "Setubal": "Vitoria", "Atlanta Utd": "Atlanta United"}
Odds_df[['HomeTeam', 'AwayTeam']] = Odds_df[['HomeTeam', 'AwayTeam']].replace(team_names)

# Lookup:
Teams = pd.DataFrame((FTE.loc[:,'team1'].append(FTE.loc[:,'team2'])).unique(), columns=['FTE'])
Odds_teams = list((Odds_df.loc[:,'HomeTeam'].append(Odds_df.loc[:,'AwayTeam'])).unique())
guess = [str(get_close_matches(team, Odds_teams, n=1, cutoff=0.6)) for team in list(Teams.FTE)]
guess = [g.strip('[' + ']' + "'") for g in guess]
Teams.loc[:,'Odds'] = guess
Teams.to_excel(wd + '0_Lookup/Teams.xlsx')
Teams_final = pd.read_excel(wd + '0_Lookup/Teams_final.xlsx')

# Mapping of team names:
odds_map = dict(Teams_final.values)
FTE['HomeTeam'] = FTE['team1'].map(odds_map)
FTE['AwayTeam'] = FTE['team2'].map(odds_map)
    
# Merge data:
Final_df = FTE.merge(Odds_df, how='left', on=['date','HomeTeam','AwayTeam'])

#%% Final dataframe
# Fix USA and Mexico time zone discrepancy:
Temp_df = Final_df[Final_df['mH'].isna()]
Temp_df['date'] = Temp_df['date'] + datetime.timedelta(days=1)
Temp_df = Temp_df.merge(Odds_df, how='left', on=['date','HomeTeam','AwayTeam'])
Temp_df['date'] = Temp_df['date'] - datetime.timedelta(days=1)
for i in range(Temp_df.shape[0]):
    row = Final_df.loc[:,['FTAG', 'FTHG', 'FTR', 'mH', 'mD', 'mA', 'Bookie_yield']][(Final_df['date']==Temp_df.loc[i,'date']) & (Final_df['team1']==Temp_df.loc[i,'team1']) & (Final_df['team2']==Temp_df.loc[i,'team2'])].index.values
    obs = pd.DataFrame(Temp_df.loc[i,['FTAG_y', 'FTHG_y', 'FTR_y', 'mH_y', 'mD_y', 'mA_y', 'Bookie_yield_y']]).T.set_index(row)
    obs.columns = ['FTAG', 'FTHG', 'FTR', 'mH', 'mD', 'mA', 'Bookie_yield']
    Final_df.loc[row,['FTAG', 'FTHG', 'FTR', 'mH', 'mD', 'mA', 'Bookie_yield']] = obs
    print(round(i/Temp_df.shape[0]*100,2))

# Keep only relevant columns:
Final_df = Final_df[['season', 'date', 'league_id', 'league', 'team1', 'team2', 'spi1',
                     'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2',
                     'importance1', 'importance2', 'score1', 'score2', 'xg1', 'xg2',
                     'nsxg1', 'nsxg2', 'adj_score1', 'adj_score2', 'FTAG', 'FTHG', 'FTR', 
                     'mH', 'mD', 'mA', 'Bookie_yield']]

# Remove leagues with missing data:
leagues = ['Austrian T-Mobile Bundesliga', 'Belgian Jupiler League', 'Danish SAS-Ligaen', 'English League One', 'English League Two', 
           'French Ligue 2', 'Norwegian Tippeligaen', 'Russian Premier Liga', 'Scottish Premiership', 'Spanish Segunda Division', 
           'Swedish Allsvenskan', 'Swiss Raiffeisen Super League', 'Turkish Turkcell Super Lig']
Final_df = Final_df[~Final_df['league'].isin(leagues)]
Final_df[Final_df['league']=='Dutch Eredivisie'] = Final_df[(Final_df['league']=='Dutch Eredivisie') & (Final_df['season']==2020)]
Final_df[Final_df['league']=='Italy Serie B'] = Final_df[(Final_df['league']=='Italy Serie B') & ~(Final_df['season']==2017)]
Final_df = Final_df.dropna(axis = 0, how = 'all')


# Drop missing values:
x = [foo for foo in Final_df.columns.values if foo not in ['importance1', 'importance2']]
Final_df = Final_df.dropna(axis = 0, subset=x)
print(Final_df.isna().sum())

# Change columns to appropriate types when relevant:
Final_df = Final_df.astype({'Bookie_yield': 'float64', 'FTAG': 'float64', 'FTHG': 'float64', 'mH': 'float64', 'mD': 'float64', 'mA': 'float64'})
Final_df = Final_df.astype({'league_id': 'category', 'league': 'category', 'team1': 'category', 'team2': 'category', 'FTR': 'category'})

# Reset index of Final_df:
Final_df.reset_index(drop=True, inplace=True)

# Impute missing data on match importance:
imputer = MissForest()
X = Final_df[Final_df.columns.difference(['date', 'league', 'team1', 'team2', 'FTR'])]
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns.values)
Final_df[['importance1', 'importance2']] = X_imputed[['importance1', 'importance2']]
print(Final_df.isna().sum())

# Export final dataframe
Final_df.to_excel(wd + "1_Cleaned_data.xlsx")


