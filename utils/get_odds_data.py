# do not read the index column
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from utils.odds_calc_utils import bookie_yield
dpath = "data/"

def get_odds_data():
    df_odds = pd.read_csv(dpath + '0_Odds-data.csv', index_col=0)

    for year in [2022,2023]:
        files = Path(f"data/{year}/").rglob("*.csv")
        df_main_csvs = [pd.read_csv(file, engine='python') for file in files]
        df_main_csvs = pd.concat(df_main_csvs)
        df_main_csvs = df_main_csvs.reset_index(drop=True)

        # Find odds for as many matches as possible:
        home = ['VCH', 'B365H', 'BWH', 'IWH', 'WHH', 'PSH']
        draw = ['VCD', 'B365D', 'BWD', 'IWD', 'WHD', 'PSD']
        away = ['VCA', 'B365A', 'BWA', 'IWA', 'WHA', 'PSA']
        df_main_csvs['mH'] = df_main_csvs.apply(lambda x: max(x[home].dropna(), default=np.nan), axis=1)
        df_main_csvs['mD'] = df_main_csvs.apply(lambda x: max(x[draw].dropna(), default=np.nan), axis=1)
        df_main_csvs['mA'] = df_main_csvs.apply(lambda x: max(x[away].dropna(), default=np.nan), axis=1)

        # Save relevant columns only:
        df_main_csvs = df_main_csvs[['HomeTeam', 'AwayTeam', 'Date', 'Div', 'FTAG', 'FTHG', 'FTR', 'mH', 'mD', 'mA']]
        # Caluclate bookie yield:
        df_main_csvs['Bookie_yield'] = df_main_csvs.apply(lambda x: bookie_yield(x.mH, x.mD, x.mA), axis=1)
        df_odds = df_odds.append(df_main_csvs)
    
    return df_odds