
fte_names = []
odds_names = []
no_match = []
for i in range(len(best_odds)):
    if get_close_matches(best_odds.loc[i, 'home_team'], fte.team1, n=1, cutoff=0.5) != []:
        fte_names.append(get_close_matches(
            best_odds.loc[i, 'home_team'], fte.team1, n=1, cutoff=0.5))
        odds_names.append(best_odds.loc[i, 'home_team'])

    elif get_close_matches(best_odds.loc[i, 'home_team'], fte.team2, n=1, cutoff=0.5) != []:
        fte_names.append(get_close_matches(
            best_odds.loc[i, 'home_team'], fte.team2, n=1, cutoff=0.5))
        odds_names.append(best_odds.loc[i, 'home_team'])

    else:
        no_match.append(best_odds.loc[i, 'home_team'])

    if get_close_matches(best_odds.loc[i, 'away_team'], fte.team2, n=1, cutoff=0.5) != []:
        fte_names.append(get_close_matches(
            best_odds.loc[i, 'away_team'], fte.team2, n=1, cutoff=0.5))
        odds_names.append(best_odds.loc[i, 'away_team'])
    elif get_close_matches(best_odds.loc[i, 'away_team'], fte.team1, n=1, cutoff=0.5) != []:
        fte_names.append(get_close_matches(
            best_odds.loc[i, 'away_team'], fte.team1, n=1, cutoff=0.5))
        odds_names.append(best_odds.loc[i, 'away_team'])
    else:
        no_match.append(best_odds.loc[i, 'away_team'])


fte_odds_dict1 = pd.Series(pd.DataFrame(
    fte_names)[0].values, index=pd.DataFrame(odds_names)[0].values).to_dict()
fte_odds_dict1.save('fte_odds_dict1')

fte_odds_dict1[]
fte[fte.league.isin(best_odds.league.unique())][[
    'team1', 'league']].to_csv('fte.csv')
best_odds[['league', 'home_team', 'away_team']].to_csv('best_odds.csv')

best_odds[best_odds.league == 'Mexican Primera Division Torneo Clausura']


def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z


fte_odds_dict = merge_two_dicts(fte_odds_dict1, fte_odds_dict)
