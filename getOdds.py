import pandas as pd
from pandas import json_normalize
import numpy as np
from soccerapi.api import Api888Sport
# from soccerapi.api import ApiUnibet
# from soccerapi.api import ApiBet365

### get 888 odds

odds_888_dict = dict({'French Ligue 1': 'https://www.888sport.com/#/filter/football/france/ligue_1/',
                        'Barclays Premier League': 'https://www.888sport.com/#/filter/football/england/premier_league/',
                        'Spanish Primera Division': 'https://www.888sport.com/#/filter/football/spain/la_liga/',
                        'Italy Serie A': 'https://www.888sport.com/#/filter/football/italy/serie_a/',
                        'German Bundesliga': 'https://www.888sport.com/#/filter/football/germany/bundesliga/',
                        'Mexican Primera Division Torneo Clausura': 'https://www.888sport.com/#/filter/football/mexico/liga_de_expansion_mx/',
                        'Major League Soccer':  'https://www.888sport.com/#/filter/football/usa/mls/',
                        'Mexican Primera Division Torneo Apertura':  'https://www.888sport.com/#/filter/football/mexico/liga_mx/',
                        'German 2. Bundesliga':  'https://www.888sport.com/#/filter/football/germany/2__bundesliga/',
                        'English League Championship': 'https://www.888sport.com/#/filter/football/england/the_championship/',
                        'Portuguese Liga':  'https://www.888sport.com/#/filter/football/portugal/primeira_liga/',
                        'Italy Serie B':  'https://www.888sport.com/#/filter/football/italy/serie_b/',
                        'Dutch Eredivisie':  'https://www.888sport.com/#/filter/football/netherlands/eredivisie/'})

leagues = json_normalize(odds_888_dict).transpose().index

api = Api888Sport()

count = 0
for league in leagues:
    url = odds_888_dict[league]
    odds = api.odds(url)
    odds_temp = pd.DataFrame.from_dict(odds)
    if odds_temp.empty==False: 
        odds_temp_OU = pd.concat([odds_temp[['time', 'home_team', 'away_team']],json_normalize(odds_temp['under_over'])/1000],axis=1)
        odds_temp_OU['league'] = league
        odds_temp = pd.concat([odds_temp[['time', 'home_team', 'away_team']],json_normalize(odds_temp['full_time_result'])/1000],axis=1)
        odds_temp['league'] = league
        
        if count == 0:
            odds_888 = odds_temp
            odds_888OU = odds_temp_OU
        else:
            odds_888 = pd.concat([odds_888,odds_temp], axis = 0)
            odds_888OU = pd.concat([odds_888OU,odds_temp_OU], axis = 0)
    count = count + 1

### get unibet odds
from soccerapi.api import ApiUnibet

api = ApiUnibet()
odds = api.competitions()


odds_unibet_dict = dict({'French Ligue 1': 'https://www.unibet.com/betting/sports/filter/football/france/ligue_1/',
                        'Barclays Premier League': 'https://www.unibet.com/betting/sports/filter/football/england/premier_league/',
                        'Spanish Primera Division': 'https://www.unibet.com/betting/sports/filter/football/spain/la_liga/',
                        'Italy Serie A': 'https://www.unibet.com/betting/sports/filter/football/italy/serie_a/',
                        'German Bundesliga': 'https://www.unibet.com/betting/sports/filter/football/germany/bundesliga/',
                        'Mexican Primera Division Torneo Clausura': 'https://www.unibet.com/betting/sports/filter/football/mexico/liga_de_expansion_mx/',
                        'Major League Soccer':  'https://www.unibet.com/betting/sports/filter/football/usa/mls/',
                        'Mexican Primera Division Torneo Apertura':  'https://www.unibet.com/betting/sports/filter/football/mexico/liga_mx/',
                        'German 2. Bundesliga': 'https://www.unibet.com/betting/sports/filter/football/germany/2__bundesliga/',
                        'English League Championship': 'https://www.unibet.com/betting/sports/filter/football/england/the_championship/',
                        'Portuguese Liga': 'https://www.unibet.com/betting/sports/filter/football/portugal/primeira_liga/',
                        'Italy Serie B':  'https://www.unibet.com/betting/sports/filter/football/italy/serie_b/',
                        'Dutch Eredivisie':  'https://www.unibet.com/betting/sports/filter/football/netherlands/eredivisie/'})
                    
count = 0
for league in leagues:
    url = odds_unibet_dict[league]
    odds = api.odds(url)
    odds_temp = pd.DataFrame.from_dict(odds)
    if odds_temp.empty==False: 
        odds_temp_OU = pd.concat([odds_temp[['time', 'home_team', 'away_team']],json_normalize(odds_temp['under_over'])/1000],axis=1)
        odds_temp_OU['league'] = league
        odds_temp = pd.concat([odds_temp[['time', 'home_team', 'away_team']],json_normalize(odds_temp['full_time_result'])/1000],axis=1)
        odds_temp['league'] = league

        if count == 0:
            odds_unibet = odds_temp
            odds_unibetOU = odds_temp_OU
        else:
            odds_unibet = pd.concat([odds_unibet,odds_temp], axis = 0)
            odds_unibetOU = pd.concat([odds_unibetOU,odds_temp_OU], axis = 0)

    count = count + 1

odds_unibet = odds_unibet.reset_index(drop=True)
odds_unibet['provider'] = 'Unibet'
odds_unibetOU = odds_unibetOU.reset_index(drop=True)
odds_unibetOU['provider'] = 'Unibet'
odds_888 = odds_888.reset_index(drop=True)
odds_888['provider'] = '888'
odds_888OU = odds_888OU.reset_index(drop=True)
odds_888OU['provider'] = '888'

best_odds = pd.merge(odds_unibet[['time','home_team', 'away_team', '1', 'X', '2', 'league']], odds_888[['time','home_team', 'away_team', '1', 'X', '2', 'league']], 
                     left_on=  ['home_team', 'away_team'],
                     right_on= ['home_team', 'away_team'], 
                     how = 'outer')

best_oddsOU = pd.merge(odds_unibetOU[['time','home_team', 'away_team', 'O2.5', 'U2.5', 'league']], odds_888OU[['time','home_team', 'away_team', 'O2.5', 'U2.5', 'league']], 
                     left_on=  ['home_team', 'away_team'],
                     right_on= ['home_team', 'away_team'], 
                     how = 'outer')

best_1 = pd.concat([best_odds['1_x'],best_odds['1_y']]).max(level=0)
best_1_provider = pd.DataFrame(np.where(best_1==best_odds['1_y'],'Unibet','888'))
best_X = pd.concat([best_odds['X_x'],best_odds['X_y']]).max(level=0)
best_X_provider = pd.DataFrame(np.where(best_X==best_odds['X_y'],'Unibet','888'))
best_2 = pd.concat([best_odds['2_x'],best_odds['2_y']]).max(level=0)
best_2_provider = pd.DataFrame(np.where(best_2==best_odds['2_y'],'Unibet','888'))

best_O25 = pd.concat([best_oddsOU['O2.5_x'],best_oddsOU['O2.5_y']]).max(level=0)
best_O25_provider = pd.DataFrame(np.where(best_1==best_oddsOU['O2.5_y'],'Unibet','888'))
best_U25 = pd.concat([best_oddsOU['U2.5_x'],best_oddsOU['U2.5_y']]).max(level=0)
best_U25_provider = pd.DataFrame(np.where(best_X==best_oddsOU['U2.5_y'],'Unibet','888'))


best_odds = pd.concat([best_odds[['time_x','home_team', 'away_team','league_x']],
                       best_1,
                       best_1_provider,
                       best_X,
                       best_X_provider,
                       best_2,
                       best_2_provider], axis=1)

best_odds.columns = ['time','home_team', 'away_team', 'league', '1', '1_provider', 'X', 'X_provider', '2', '2_provider']

best_oddsOU = pd.concat([best_oddsOU[['time_x','home_team', 'away_team','league_x']],
                       best_O25,
                       best_O25_provider,
                       best_U25,
                       best_U25_provider], axis=1)

best_oddsOU.columns = ['time','home_team', 'away_team', 'league', 'O25', 'O25_provider', 'U25', 'U25_provider']

best_oddsOU = best_oddsOU.dropna()