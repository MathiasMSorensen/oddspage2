import pandas as pd
from difflib import get_close_matches
import sys
import os
sys.path.append('projects/oddspage')
from utils.get_odds_data import get_odds_data

os.chdir('projects/oddspage')
#%% Five Thirty Eight:

# Working directory:
url = "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv"
df_fte = pd.read_csv(url)
df_odds = get_odds_data()

# Only keep relevant leagues:
Leagues = ['Austrian T-Mobile Bundesliga','Barclays Premier League','Belgian Jupiler League','Brasileiro S√©rie A','Danish SAS-Ligaen',
           'Dutch Eredivisie','English League Championship','English League One','English League Two','French Ligue 1','French Ligue 2',
           'German 2. Bundesliga','German Bundesliga','Italy Serie A','Italy Serie B','Major League Soccer',
           'Mexican Primera Division Torneo Apertura','Mexican Primera Division Torneo Clausura','Norwegian Tippeligaen','Portuguese Liga',
           'Russian Premier Liga','Scottish Premiership','Spanish Primera Division','Spanish Segunda Division','Swedish Allsvenskan',
           'Swiss Raiffeisen Super League','Turkish Turkcell Super Lig']
df_fte = df_fte[(df_fte['league'].isin(Leagues))]

# Change to datetime format:
df_fte['date'] = pd.to_datetime(df_fte['date'], format='%Y-%m-%d')

def odds_date(obs):
    try:
        x = pd.to_datetime(obs['Date'], format='%d/%m/%y')
    except:
        x = pd.to_datetime(obs['Date'], format='%d/%m/%Y')
    return x
df_odds['Date'] = df_odds.apply(lambda x: odds_date(x), axis=1)
df_odds = df_odds.rename(columns={'Date':'date'})

# Correct team names:
team_names = {"Chievo": "Verona", "Setubal": "Vitoria", "Atlanta Utd": "Atlanta United"}
df_odds[['HomeTeam', 'AwayTeam']] = df_odds[['HomeTeam', 'AwayTeam']].replace(team_names)

# # Lookup:
# Teams = pd.DataFrame((df_fte.loc[:,'team1'].append(df_fte.loc[:,'team2'])).unique(), columns=['df_fte'])
# Odds_teams = list((df_odds.loc[:,'HomeTeam'].append(df_odds.loc[:,'AwayTeam'])).unique())
# guess = [str(get_close_matches(team, Odds_teams, n=1, cutoff=0.6)) for team in list(Teams.df_fte)]
# guess = [g.strip('[' + ']' + "'") for g in guess]
# Teams.loc[:,'Odds'] = guess
# Teams.to_csv("data/0_Lookup/Teams.csv")

Teams_final = pd.read_csv("data/LookupTeams_final.csv", index_col=0)
Teams_final.loc[Teams_final["Odds"].isna(), "Odds"] = "None"

# Mapping of team names:
odds_map = dict(Teams_final.values)
df_fte['HomeTeam'] = df_fte['team1'].map(odds_map)
df_fte['AwayTeam'] = df_fte['team2'].map(odds_map)
    
# Merge data:
Final_df = df_fte.merge(df_odds, how='left', on=['date','HomeTeam','AwayTeam'])

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

# Save:
Final_df.to_csv("data/df_abt.csv")