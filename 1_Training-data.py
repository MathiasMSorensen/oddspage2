#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:27:40 2021

@author: MadsJorgensen
"""

# Libraries:
import pandas as pd
import numpy as np
from datetime import datetime
from missingpy import MissForest

# Working directory:
wd = '/Users/MadsJorgensen/Dropbox (Personal)/Betting/FiveThirtyEight/Model training/'

DF = pd.read_excel(wd + 'Raw data/1_Cleaned_data.xlsx')

#%% Define functions:

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


#%% Create features

# Copy to final dataframe:
Final_df = DF.copy(deep=True)

# Avg points:
temp = pd.DataFrame(Final_df.apply(lambda y: avg_points(Final_df, y.team1, y.date, y.season), axis=1))
temp.columns = ['avg_P_home']
temp = pd.DataFrame(Final_df.apply(lambda y: avg_points(Final_df, y.team2, y.date, y.season), axis=1))
temp.columns = ['avg_P_away']

# x number of previous matches to consider:
x = [1, 3, 5]

# Home team features:
for i in x:
    print('home_' + str(i) + ' ...')
    temp = Final_df.apply(lambda y: stats_last_x_matches(Final_df, y.team1, y.date, i), axis=1)
    temp.columns = [str(col) + '_home_' + str(i) for col in temp.columns]
    Final_df = pd.concat([Final_df, temp], axis=1)


# Away team features:
for i in x:
    print('away_' + str(i) + ' ...')
    temp = Final_df.apply(lambda y: stats_last_x_matches(Final_df, y.team2, y.date, i), axis=1)
    temp.columns = [str(col) + '_away_' + str(i) for col in temp.columns]
    Final_df = pd.concat([Final_df, temp], axis=1)


# H2H features
x = 2
print('H2H_' + str(x) + ' ...')
temp = Final_df.apply(lambda y: stats_last_x_H2H(Final_df, y.team1, y.team2, y.date, i), axis=1)
temp.columns = [str(col) + '_' + str(x) for col in temp.columns]
Final_df = pd.concat([Final_df, temp], axis=1)
del(temp)


# Impute values for nan:
imputer = MissForest()
X = Final_df[Final_df.columns.difference(['date', 'league', 'team1', 'team2', 'FTR'])]
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns.values)
Final_df[list(X_imputed.columns.values)] = X_imputed
print(Final_df.isna().sum().sum())

Final_df.to_excel(wd + 'Training data/1_Training data.xlsx')
