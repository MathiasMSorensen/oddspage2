import pandas as pd 
import numpy as np
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
import sklearn.neighbors._base
from missingpy import MissForest
import datetime

def bookie_yield(H, D, A):
    try:
        x = (1/H + 1/D + 1/A - 1)
    except ZeroDivisionError:
        x = np.nan
    return x

def odds_date(obs):
    try:
        x = pd.to_datetime(obs['Date'], format='%d/%m/%y')
    except:
        x = pd.to_datetime(obs['Date'], format='%d/%m/%Y')
    return x

def last_x_matches(match_data, team, date, x):
    # Extract all matches for a given team:
    team_matches = match_data[(match_data['team1']==team) | 
                              (match_data['team2']==team)]
    # Sort matches previous to the current match and select the last x of them:
    latest_matches = team_matches[team_matches['date']<date].sort_values(by='date', ascending=False)[:x]
    # Output these x matches:
    return latest_matches

# Last x matches against each other (Head-to-Head):
def last_x_H2H(match_data, team1, team2, date, x):
    # Extract all Head-to-Head matches of the two team:
    H2H_matches = match_data[((match_data['team1']==team1) & 
                             (match_data['team2']==team2)) |
                             ((match_data['team1']==team2) &
                             (match_data['team2']==team1))]
    # Sort matches previous to the current match and select the last x of them:
    latest_H2H = H2H_matches[H2H_matches['date']<date].sort_values(by='date', ascending=False)[:x]
    # Output last x Head-to-Head games:
    return latest_H2H

### Compute statistics for last x matches:
def stats_last_x_matches(match_data, team, date, x):
    # Extract the last x matches:
    latest_matches = last_x_matches(match_data, team, date, x)
    if(len(latest_matches)>0):
        # Goals for:
        goals = pd.concat([latest_matches.adj_score1[latest_matches['team1']==team],
                           latest_matches.adj_score2[latest_matches['team2']==team]],
                           axis=0).sum()
        xgoals = pd.concat([latest_matches.xg1[latest_matches['team1']==team],
                           latest_matches.xg2[latest_matches['team2']==team]],
                           axis=0).sum()
        nsxgoals = pd.concat([latest_matches.nsxg1[latest_matches['team1']==team],
                           latest_matches.nsxg2[latest_matches['team2']==team]],
                           axis=0).sum()
        proj_goals = pd.concat([latest_matches.proj_score1[latest_matches['team1']==team],
                           latest_matches.proj_score2[latest_matches['team2']==team]],
                           axis=0).sum()
        goals_diff = goals-proj_goals
        xgoals_diff = xgoals-proj_goals
        nsxgoals_diff = nsxgoals-proj_goals
        
        # Goals against:
        goals_against = pd.concat([latest_matches.adj_score2[latest_matches['team1']==team],
                           latest_matches.adj_score1[latest_matches['team2']==team]],
                           axis=0).sum()
        xgoals_against = pd.concat([latest_matches.xg2[latest_matches['team1']==team],
                           latest_matches.xg1[latest_matches['team2']==team]],
                           axis=0).sum()
        nsxgoals_against = pd.concat([latest_matches.nsxg2[latest_matches['team1']==team],
                           latest_matches.nsxg1[latest_matches['team2']==team]],
                           axis=0).sum()
        proj_goals_against = pd.concat([latest_matches.proj_score2[latest_matches['team1']==team],
                           latest_matches.proj_score1[latest_matches['team2']==team]],
                           axis=0).sum()
        goals_against_diff = goals_against-proj_goals_against
        xgoals_against_diff = xgoals_against-proj_goals_against
        nsxgoals_against_diff = nsxgoals_against-proj_goals_against
        
        # Number of points:
        wins = (len(latest_matches[(latest_matches['team1']==team) & 
                   (latest_matches['FTR']=='H')])
                +
                len(latest_matches[(latest_matches['team2']==team) & 
                   (latest_matches['FTR']=='A')])
               )
        draws = len(latest_matches[latest_matches['FTR']=='D'])
        points = 3*wins + draws
        # Expected points:
        xwins = pd.concat([latest_matches.prob1[latest_matches['team1']==team],
               latest_matches.prob2[latest_matches['team2']==team]],
               axis=0).sum()
        xdraws = latest_matches['probtie'].sum()
        xpoints = 3*xwins + xdraws
        # Points score:
        points_diff = points-xpoints
        
        # SPI:
        spi_against = pd.concat([latest_matches.spi2[latest_matches['team1']==team],
                      latest_matches.spi1[latest_matches['team2']==team]],
                      axis=0).sum()
        
        # Importance: 
        importance = pd.concat([latest_matches.importance1[latest_matches['team1']==team],
                     latest_matches.importance2[latest_matches['team2']==team]],
                     axis=0).sum()
        importance_against = pd.concat([latest_matches.importance2[latest_matches['team1']==team],
                             latest_matches.importance1[latest_matches['team2']==team]],
                             axis=0).sum()
       
        # Output statistics:
        stats = pd.DataFrame(columns=['GS', 'xGS', 'nsxGS', 'GS_diff', 'xGS_diff', 'nsxGS_diff',
                                      'GSA', 'xGSA', 'nsxGSA', 'GSA_diff', 'xGSA_diff', 'nsxGSA_diff',
                                      'P', 'xP_diff',
                                      'SPIA', 'IMP', 'IMPA'])
        stats.loc[0] = [goals, xgoals, nsxgoals, goals_diff, xgoals_diff, nsxgoals_diff, 
                        goals_against, xgoals_against, nsxgoals_against, goals_against_diff, xgoals_against_diff, nsxgoals_against_diff,
                        points, points_diff,
                        spi_against, importance, importance_against]
        if(len(latest_matches)<x):
            stats = stats*(x/len(latest_matches))
    
    else:
        stats = pd.DataFrame(columns=['GS', 'xGS', 'nsxGS', 'GS_diff', 'xGS_diff', 'nsxGS_diff',
                                      'GSA', 'xGSA', 'nsxGSA', 'GSA_diff', 'xGSA_diff', 'nsxGSA_diff',
                                      'P', 'xP_diff',
                                      'SPIA', 'IMP', 'IMPA'])
        stats.loc[0] = None
    return stats.iloc[0]

def avg_points(match_data, team, date, season):
    # Check number of matches overall:
    N = len(match_data[(match_data.date<date) & ((match_data.team1==team) | (match_data.team2==team))])
    if(N>=10):
        # Matches played in current season:
        matches = match_data[(match_data.season==season) & (match_data.date<date)]
        matches = matches[(matches.team1==team) | (matches.team2==team)]
        if(len(matches)>10):
            # Number of points:
            wins = (len(matches[(matches['team1']==team) & 
                       (matches['FTR']=='H')])
                    +
                    len(matches[(matches['team2']==team) & 
                       (matches['FTR']=='A')])
                   )
            draws = len(matches[matches['FTR']=='D'])
            points = 3*wins + draws
            # Number of games:
            N = len(matches)
            # Compute average points per match:
            avgP = points/N
        elif(len(matches)>5):
            # Number of points in current season:
            wins = (len(matches[(matches['team1']==team) & 
                       (matches['FTR']=='H')])
                    +
                    len(matches[(matches['team2']==team) & 
                       (matches['FTR']=='A')])
                   )
            draws = len(matches[matches['FTR']=='D'])
            points = 3*wins + draws
            # Number of games in current season:
            N = len(matches)
            # Compute average points per match in current season:
            avgP = points/N
            
            # Last 38 matches:
            matches = match_data[match_data.date<date]
            matches = matches[(matches.team1==team) | (matches.team2==team)]
            matches = matches.sort_values(by='date', ascending=False)[0:38]
            # Number of points in Last 38 matches:
            wins = (len(matches[(matches['team1']==team) & 
                       (matches['FTR']=='H')])
                    +
                    len(matches[(matches['team2']==team) & 
                       (matches['FTR']=='A')])
                   )
            draws = len(matches[matches['FTR']=='D'])
            points = 3*wins + draws
            # Compute average points per match in Last 38 matches:
            avgP38 = points/38
            
            # Compute average of avgP and avgP38
            avgP = (avgP+avgP38)/2
        else:
            # Last 38 matches:
            matches = match_data[match_data.date<date]
            matches = matches[(matches.team1==team) | (matches.team2==team)]
            matches = matches.sort_values(by='date', ascending=False)[0:38]
            # Number of points in Last 38 matches:
            wins = (len(matches[(matches['team1']==team) & 
                       (matches['FTR']=='H')])
                    +
                    len(matches[(matches['team2']==team) & 
                       (matches['FTR']=='A')])
                   )
            draws = len(matches[matches['FTR']=='D'])
            points = 3*wins + draws
            # Compute average points per match in Last 38 matches:
            avgP = points/38
    else:
        avgP = None
    # Output average points:
    return avgP


### Compute statistics for last x H2H matches:
def stats_last_x_H2H(match_data, team1, team2, date, x):
    # Extract the last x H2H matches:
    latest_matches = last_x_H2H(match_data, team1, team2, date, x)
    
    if(len(latest_matches)>0):
        # Number of points:
        team1_wins = (len(latest_matches[(latest_matches['team1']==team1) & 
                                        (latest_matches['FTR']=='H')])
                    +
                    len(latest_matches[(latest_matches['team2']==team1) & 
                                       (latest_matches['FTR']=='A')])
                    )
        team2_wins = (len(latest_matches[(latest_matches['team1']==team2) & 
                                        (latest_matches['FTR']=='H')])
                    +
                    len(latest_matches[(latest_matches['team2']==team2) & 
                                       (latest_matches['FTR']=='A')])
                    )
        draws = len(latest_matches[latest_matches['FTR']=='D'])
        team1_points = 3*team1_wins + draws
        team2_points = 3*team2_wins + draws
        # Expected points:
        team1_xwins = pd.concat([latest_matches.prob1[latest_matches['team1']==team1],
                      latest_matches.prob2[latest_matches['team2']==team1]],
                      axis=0).sum()
        team2_xwins = pd.concat([latest_matches.prob1[latest_matches['team1']==team2],
                      latest_matches.prob2[latest_matches['team2']==team2]],
                      axis=0).sum()
        xdraws = latest_matches['probtie'].sum()
        team1_xpoints = 3*team1_xwins + xdraws
        team2_xpoints = 3*team2_xwins + xdraws
        # Points score:
        team1_xpoints_diff = team1_points-team1_xpoints
        team2_xpoints_diff = team2_points-team2_xpoints
        
        # Goal difference:
        team1_goals = pd.concat([latest_matches.adj_score1[latest_matches['team1']==team1],
                           latest_matches.adj_score2[latest_matches['team2']==team1]],
                           axis=0).sum()
        team2_goals = pd.concat([latest_matches.adj_score1[latest_matches['team1']==team2],
                           latest_matches.adj_score2[latest_matches['team2']==team2]],
                           axis=0).sum()
        team1_xgoals = pd.concat([latest_matches.xg1[latest_matches['team1']==team1],
                           latest_matches.xg2[latest_matches['team2']==team1]],
                           axis=0).sum()
        team2_xgoals = pd.concat([latest_matches.xg1[latest_matches['team1']==team2],
                           latest_matches.xg2[latest_matches['team2']==team2]],
                           axis=0).sum()
        team1_nsxgoals = pd.concat([latest_matches.nsxg1[latest_matches['team1']==team1],
                           latest_matches.nsxg2[latest_matches['team2']==team1]],
                           axis=0).sum()
        team2_nsxgoals = pd.concat([latest_matches.nsxg1[latest_matches['team1']==team2],
                           latest_matches.nsxg2[latest_matches['team2']==team2]],
                           axis=0).sum()
        team1_proj_goals = pd.concat([latest_matches.proj_score1[latest_matches['team1']==team1],
                           latest_matches.proj_score2[latest_matches['team2']==team1]],
                           axis=0).sum()
        team2_proj_goals = pd.concat([latest_matches.proj_score1[latest_matches['team1']==team2],
                           latest_matches.proj_score2[latest_matches['team2']==team2]],
                           axis=0).sum()
        goals_diff = team1_goals - team2_goals
        xgoals_diff = team1_xgoals - team2_xgoals
        nsxgoals_diff = team1_nsxgoals - team2_nsxgoals
        proj_goals_diff = team1_proj_goals - team2_proj_goals
        
        # Importance: 
        team1_importance = pd.concat([latest_matches.importance1[latest_matches['team1']==team1],
                     latest_matches.importance2[latest_matches['team2']==team1]],
                     axis=0).sum()
        team2_importance = pd.concat([latest_matches.importance1[latest_matches['team1']==team2],
                             latest_matches.importance2[latest_matches['team2']==team2]],
                             axis=0).sum()
        importance = team1_importance - team2_importance
        
        # Collect as dataframe:
        H2H_stats = pd.DataFrame(columns=['team1_H2H_P', 'team2_H2H_P', 'team1_H2H_xP_diff', 'team2_H2H_xP_diff', 
                                          'H2H_G_diff', 'H2H_xG_diff', 'H2H_nsxG_diff', 'H2H_projG_diff',
                                          'H2H_IMP_diff'])
        H2H_stats.loc[0] = [team1_points, team2_points, team1_xpoints_diff, team2_xpoints_diff,
                            goals_diff, xgoals_diff, nsxgoals_diff, proj_goals_diff,
                            importance]
        if(len(latest_matches)<x):
            H2H_stats = H2H_stats*(x/len(latest_matches))
    else:
        # Define dataframe:
        H2H_stats = pd.DataFrame(columns=['team1_H2H_P', 'team2_H2H_P', 'team1_H2H_xP_diff', 'team2_H2H_xP_diff', 
                                          'H2H_G_diff', 'H2H_xG_diff', 'H2H_nsxG_diff', 'H2H_projG_diff',
                                          'H2H_IMP_diff'])
        # Set nan:
        H2H_stats.loc[0] = None
        
    # Output row with values:
    return H2H_stats.iloc[0]


### Average points in current season:
def avg_points(match_data, team, date, season):
    # Check number of matches overall:
    N = len(match_data[(match_data.date<date) & ((match_data.team1==team) | (match_data.team2==team))])
    if(N>=10):
        # Matches played in current season:
        matches = match_data[(match_data.season==season) & (match_data.date<date)]
        matches = matches[(matches.team1==team) | (matches.team2==team)]
        if(len(matches)>10):
            # Number of points:
            wins = (len(matches[(matches['team1']==team) & 
                       (matches['FTR']=='H')])
                    +
                    len(matches[(matches['team2']==team) & 
                       (matches['FTR']=='A')])
                   )
            draws = len(matches[matches['FTR']=='D'])
            points = 3*wins + draws
            # Number of games:
            N = len(matches)
            # Compute average points per match:
            avgP = points/N
        elif(len(matches)>5):
            # Number of points in current season:
            wins = (len(matches[(matches['team1']==team) & 
                       (matches['FTR']=='H')])
                    +
                    len(matches[(matches['team2']==team) & 
                       (matches['FTR']=='A')])
                   )
            draws = len(matches[matches['FTR']=='D'])
            points = 3*wins + draws
            # Number of games in current season:
            N = len(matches)
            # Compute average points per match in current season:
            avgP = points/N
            
            # Last 38 matches:
            matches = match_data[match_data.date<date]
            matches = matches[(matches.team1==team) | (matches.team2==team)]
            matches = matches.sort_values(by='date', ascending=False)[0:38]
            # Number of points in Last 38 matches:
            wins = (len(matches[(matches['team1']==team) & 
                       (matches['FTR']=='H')])
                    +
                    len(matches[(matches['team2']==team) & 
                       (matches['FTR']=='A')])
                   )
            draws = len(matches[matches['FTR']=='D'])
            points = 3*wins + draws
            # Compute average points per match in Last 38 matches:
            avgP38 = points/38
            
            # Compute average of avgP and avgP38
            avgP = (avgP+avgP38)/2
        else:
            # Last 38 matches:
            matches = match_data[match_data.date<date]
            matches = matches[(matches.team1==team) | (matches.team2==team)]
            matches = matches.sort_values(by='date', ascending=False)[0:38]
            # Number of points in Last 38 matches:
            wins = (len(matches[(matches['team1']==team) & 
                       (matches['FTR']=='H')])
                    +
                    len(matches[(matches['team2']==team) & 
                       (matches['FTR']=='A')])
                   )
            draws = len(matches[matches['FTR']=='D'])
            points = 3*wins + draws
            # Compute average points per match in Last 38 matches:
            avgP = points/38
    else:
        avgP = None
    # Output average points:
    return avgP

def impute(DF):
    imputer = MissForest()
    DF = DF.reset_index(drop=True)
    X = DF[DF.columns.difference(['date', 'league', 'team1', 'team2', 'FTR'])]
    X_imputed = imputer.fit_transform(X)

    X_imputed = pd.DataFrame(X_imputed, columns=X.columns.values)
    DF[['importance1', 'importance2']] = X_imputed[['importance1', 'importance2']]
    return DF

# Avg points:

def make_final_data(odds_fte_df,Final_df):
    temp = pd.DataFrame(odds_fte_df.apply(lambda y: avg_points(Final_df, y.team1, y.date, y.season), axis=1))
    temp.columns = ['avg_P_home']
    temp = pd.DataFrame(odds_fte_df.apply(lambda y: avg_points(Final_df, y.team2, y.date, y.season), axis=1))
    temp.columns = ['avg_P_away']

    # x number of previous matches to consider:
    x = [1, 3, 5]

    # Home team features:
    for i in x:
        print('home_' + str(i) + ' ...')
        temp = odds_fte_df.apply(lambda y: stats_last_x_matches(Final_df, y.team1, y.date, i), axis=1)
        temp.columns = [str(col) + '_home_' + str(i) for col in temp.columns]
        odds_fte_df = pd.concat([odds_fte_df, temp], axis=1)

    # Away team features:
    for i in x:
        print('away_' + str(i) + ' ...')
        temp = odds_fte_df.apply(lambda y: stats_last_x_matches(Final_df, y.team2, y.date, i), axis=1)
        temp.columns = [str(col) + '_away_' + str(i) for col in temp.columns]
        odds_fte_df = pd.concat([odds_fte_df, temp], axis=1)

    # H2H features
    x = 2
    print('H2H_' + str(x) + ' ...')
    temp = odds_fte_df.apply(lambda y: stats_last_x_H2H(Final_df, y.team1, y.team2, y.date, i), axis=1)
    temp.columns = [str(col) + '_' + str(x) for col in temp.columns]
    odds_fte_df = pd.concat([odds_fte_df, temp], axis=1)
    del(temp)

    # Change columns to appropriate types when relevant:
    odds_fte_df = odds_fte_df.astype({ 'mH': 'float64', 'mD': 'float64', 'mA': 'float64'})
    odds_fte_df = odds_fte_df.astype({'league_id': 'category', 'league': 'category', 'team1': 'category', 'team2': 'category'})

    # Reset index of Final_df:
    odds_fte_df.reset_index(drop=True, inplace=True)

    Final_df = odds_fte_df.drop(['season', 'date', 'league', 'team1', 'team2', 'score1', 'score2', 'xg1', 'xg2',
                        'nsxg1', 'nsxg2', 'adj_score1', 'adj_score2', 'Bookie_yield','FTR'], axis=1)

    return Final_df, odds_fte_df


def bet(prob_data, margin):
    Bet = 'None'
    maxi = max(prob_data['Hp_diff'], prob_data['Dp_diff'], prob_data['Ap_diff'])
    if(maxi>margin):
        y = prob_data[['Hp_diff','Dp_diff','Ap_diff']]==maxi
        col = pd.DataFrame(y.index[y==True])
        if col.iloc[0,0]=='Hp_diff':
            Bet = 'H'
        elif col.iloc[0,0]=='Dp_diff':
            Bet = 'None'
        else:
            Bet = 'A'
    else:
        Bet = 'None'
    return Bet


def bet25(prob_data, margin):
    Bet = 'None'
    maxi = max(prob_data['U25p_diff'], prob_data['O25p_diff'])
    if(maxi>margin):
        y = prob_data[['U25p_diff','O25p_diff']]==maxi
        col = pd.DataFrame(y.index[y==True])
        if col.iloc[0,0]=='U25p_diff':
            Bet = 0
        else:
            Bet = 1
    else:
        Bet = 'None'
    return Bet

def betting_function(Yprob, Final_df, best_odds):
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
    # Merge with Ytest to get outcome:
    Probs = Probs.merge(right=pd.DataFrame(odds_fte_df), how='inner', left_index=True, right_index=True)
    # Calculate difference between estimated and implied probability:
    Probs['Hp_diff'] = Probs.H_p - Probs.H_odds_p
    Probs['Dp_diff'] = Probs.D_p - Probs.D_odds_p
    Probs['Ap_diff'] = Probs.A_p - Probs.A_odds_p
    # Merge to get odds:
    Probs = Probs.merge(right=Final_df[['mH', 'mD', 'mA']], how='inner',
                        left_index=True, right_index=True)
    
    Probs['Bet'] = Probs.apply(lambda y : bet(y, spread_limit), axis=1)

    odds_temp = pd.merge(best_odds, fte[((fte['date']>=min(best_odds['time'])) & (fte['date']<datetime.date(2022, 1, 1)))], left_on=  ['fte_home', 'fte_away'],
                                        right_on= ['team1', 'team2'], 
                                        how = 'left')[['fte_home', 'fte_away','1_provider','X_provider','2_provider']]

    Probs = pd.merge(Probs, odds_temp, left_on= ['team1', 'team2'],
                                        right_on= ['fte_home', 'fte_away'] , 
                                        how = 'left')

    return Probs


def Kelly_criteria(Probs, Ytest, weight):
    b = np.where(Probs['Bet']=='A', Probs['mA'], 0) + np.where(Probs['Bet']=='H', Probs['mH'], 0)
    p = np.where(Probs['Bet']=='A', Probs['A_p'], 0) + np.where(Probs['Bet']=='H', Probs['H_p'], 0)

    kelly  = pd.DataFrame(np.where(b!=0,(b*p-(1-p))/b,0))/weight
    return_algo = 1
    for i in range(len(kelly)):
        if Probs['Bet'].iloc[i]=='A':
            return_algo = kelly.iloc[i,0]*return_algo*Probs['mA'].iloc[i]*(Ytest.iloc[i]==Probs['Bet'].iloc[i])+(1-kelly.iloc[i,0])*return_algo
        elif Probs['Bet'].iloc[i]=='H':
            return_algo = kelly.iloc[i,0]*return_algo*Probs['mH'].iloc[i]*(Ytest.iloc[i]==Probs['Bet'].iloc[i])+(1-kelly.iloc[i,0])*return_algo
        else :
            return_algo = return_algo

    max_bet = kelly.max()[0]

    return return_algo, max_bet

def Kelly_criteria25(Probs, Ytest, weight):
    b = np.where(Probs['Bet']==0, Probs['U25'], 0) + np.where(Probs['Bet']==1, Probs['O25'], 0)
    p = np.where(Probs['Bet']==0, Probs['U25_p'], 0) + np.where(Probs['Bet']==1, Probs['O25_p'], 0)

    kelly  = pd.DataFrame(np.where(b!=0,(b*p-(1-p))/b,0))/weight
    return_algo = 1
    for i in range(len(kelly)):
        if Probs['Bet'].iloc[i]==0:
            return_algo = kelly.iloc[i,0]*return_algo*Probs['U25'].iloc[i]*(Ytest.iloc[i]==Probs['Bet'].iloc[i])+(1-kelly.iloc[i,0])*return_algo
        elif Probs['Bet'].iloc[i]==1:
            return_algo = kelly.iloc[i,0]*return_algo*Probs['O25'].iloc[i]*(Ytest.iloc[i]==Probs['Bet'].iloc[i])+(1-kelly.iloc[i,0])*return_algo
        else :
            return_algo = return_algo

    max_bet = kelly.max()[0]

    return return_algo, max_bet


def betting_function_backtest(Yprob, Final_df):
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
    
    Probs['Bet'] = Probs.apply(lambda y : bet(y, 0.075), axis=1)

    return Probs



def betting_function_backtest25(Yprob, Final_df):
# Rename correspondingly:
    Yprob.columns = ['U25_p', 'O25_p']
    # Set indices correctly:
    Yprob = Yprob.set_index(Final_df.index.values)
    # Generate Data Frame with implied probabilities:
    odds_yield = 1/(1/Final_df[['U25', 'O25']]).sum(axis=1)
    Probs = pd.DataFrame(pd.concat([odds_yield/Final_df['U25'],odds_yield/Final_df['O25']], axis = 1))

    Probs.columns = ['U25_odds_p', 'O25_odds_p']
    # Merge with estimated probabilites:
    Probs = Probs.merge(right=Yprob, how='inner', left_index=True, right_index=True)
 
    # Calculate difference between estimated and implied probability:
    Probs['U25p_diff'] = Probs.U25_p - Probs.U25_odds_p
    Probs['O25p_diff'] = Probs.O25_p - Probs.O25_odds_p

    # Merge to get odds:
    Probs = Probs.merge(right=Final_df[['U25', 'O25']], how='inner',
                        left_index=True, right_index=True)
    
    Probs['Bet'] = Probs.apply(lambda y : bet25(y, 0.075), axis=1)

    return Probs

def kelly_and_provider(best_odds, fte, Probs, weight):
    odds_temp = pd.merge(best_odds, fte[((fte['date']>=min(best_odds['time'])) & (fte['date'] < datetime.date(2022, 1, 1)))], left_on=  ['fte_home', 'fte_away'],
                                        right_on= ['team1', 'team2'], 
                                        how = 'left')[['time','league_x','fte_home', 'fte_away','1_provider','X_provider','2_provider']].dropna()

    Probs = pd.concat([Probs,odds_temp[odds_temp.index.isin(Probs.index)]], axis=1)

    b = np.where(Probs['Bet']=='A', Probs['mA'], 0) + np.where(Probs['Bet']=='H', Probs['mH'], 0)
    p = np.where(Probs['Bet']=='A', Probs['A_p'], 0) + np.where(Probs['Bet']=='H', Probs['H_p'], 0)

    kelly  = pd.DataFrame(np.where(b!=0,(b*p-(1-p))/b,0))/weight
    Probs = pd.concat([Probs.reset_index(drop=True),kelly], axis = 1)

    Probs.to_csv('todays_bets')
    
    return Probs


def kelly_and_provider25(best_odds, fte,ProbsOU,weight):
    odds_temp = pd.merge(best_odds, fte[((fte['date']>=min(best_odds['time'])) & (fte['date']<datetime.date(2022, 1, 1)))], left_on=  ['fte_home', 'fte_away'],
                                        right_on= ['team1', 'team2'], 
                                        how = 'left')[['time','league_x','fte_home', 'fte_away', 'O25_provider', 'U25_provider']].dropna().reset_index(drop=True)

    ProbsOU = pd.concat([ProbsOU,odds_temp[odds_temp.index.isin(ProbsOU.index)]], axis=1)

    b = np.where(ProbsOU['Bet']==0, ProbsOU['U25'], 0) + np.where(ProbsOU['Bet']==1, ProbsOU['O25'], 0)
    p = np.where(ProbsOU['Bet']==0, ProbsOU['U25_p'], 0) + np.where(ProbsOU['Bet']==1, ProbsOU['O25_p'], 0)

    kelly  = pd.DataFrame(np.where(b!=0,(b*p-(1-p))/b,0))/weight
    ProbsOU = pd.concat([ProbsOU.reset_index(drop=True),kelly], axis = 1)
    ProbsOU[['time','league_x','fte_home','fte_away','Bet',pd.DataFrame(np.where(ProbsOU['Bet']=='1','O25_odds_p',(np.where(ProbsOU['Bet']=='0','U25_odds_p','')))),
                                                           pd.DataFrame(np.where(ProbsOU['Bet']=='1','O25_p',(np.where(ProbsOU['Bet']=='0','U25_p',0)))),
                                                           pd.DataFrame(np.where(ProbsOU['Bet']=='1','O25_provider',(np.where(ProbsOU['Bet']=='0','U25_provider',0))))]]
    ProbsOU.loc[:,pd.MultiIndex.from_frame(pd.DataFrame(np.where(ProbsOU['Bet']=='1','O25_odds_p',(np.where(ProbsOU['Bet']=='0','U25_odds_p',0)))))]
    ProbsOU.to_csv('todays_betsOU')
    todays_betsOU.columns
    return ProbsOU
