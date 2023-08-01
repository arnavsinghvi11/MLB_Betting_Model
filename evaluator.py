from unidecode import unidecode
from itertools import combinations
import numpy as np
import pandas as pd
import os

class Evaluator:
    def stat_greater_than(self, stat, player_data, threshold):
        #returns if player's given statistic is higher than passed in threshold
        return float(player_data[stat].values[0]) > float(threshold)

    def stat_less_than(self, stat, player_data, threshold):
        #returns if player's given statistic is lower than passed in threshold
        return float(player_data[stat].values[0]) < float(threshold)

    def evaluator(self, symbol, first_stat, second_stat, third_stat, player_data, threshold):
        #returns if player achieved above prop threshold if an over bet, under prop threshold if an under bet

        #preprocessing for triple double props
        if third_stat:
            triple = float(player_data[first_stat].values[0]) + float(player_data[second_stat].values[0]) + float(player_data[third_stat].values[0])
            if symbol == '>':
                if triple > threshold:
                    return 'Y'
                return 'N' 
            else:
                if triple < threshold:
                    return 'Y'
                return 'N' 
        
        #preprocessing for double double props
        if second_stat:
            double = float(player_data[first_stat].values[0]) + float(player_data[second_stat].values[0])
            if symbol == '>':
                if double > threshold:
                    return 'Y'
                return 'N'
            else:
                if double < threshold:
                    return 'Y'
                return 'N'

        if symbol == '>':
            if self.stat_greater_than(first_stat, player_data, threshold):
                return 'Y'
            return 'N'
        else:
            if self.stat_less_than(first_stat, player_data, threshold):
                return 'Y'
            return 'N'
        
    def predictions_evaluator(self, predictions, hitting_box_score, pitching_box_score):
        #returns evaluations of past predictions based on past box score corresponding
        correct, names = [], []
        for p in range(len(predictions)):
            bet = predictions.loc[p]
            bet['Play'] = bet['Play'].replace('  ', ' ')
            bet['Play'] = bet['Play'].lower()
            i = bet['Play']
            name = unidecode(i.split(' ')[0])
            teams = [bet['Team']]
            prop_bet_types = bet['Play'].split(' ')[2:]
            if len(prop_bet_types) > 1:
                prop_bet_type = ''
                for prop_bet in prop_bet_types:
                    prop_bet_type += prop_bet
                    prop_bet_type += ' '
            else:
                prop_bet_type = prop_bet_types[0]
            prop_bet_type = prop_bet_type.strip()
            if prop_bet_type == 'ks' or prop_bet_type == 'strikeouts' or prop_bet_type == 'earned runs' or prop_bet_type == 'earned runs allowed' or prop_bet_type == 'hits allowed'or prop_bet_type == 'walks allowed': 
                pitching_box_score['Name'] = pitching_box_score['Name'].apply(unidecode)
                matching_name = pitching_box_score[pitching_box_score['Name'].str.lower() == name]
                player_data = matching_name[matching_name['Team'].isin(teams)]
            elif prop_bet_type == 'total bases' or prop_bet_type == 'hits' or prop_bet_type == 'home runs' or prop_bet_type == 'hr' or prop_bet_type == 'rbi' or prop_bet_type == 'hits+runs+rbi' or prop_bet_type == 'hits + runs + rbis' or prop_bet_type == 'runs scored' or prop_bet_type == 'bb' or prop_bet_type == 'walks':
                hitting_box_score['Name'] = hitting_box_score['Name'].apply(unidecode)
                matching_name = hitting_box_score[hitting_box_score['Name'].str.lower() == name]
                player_data = matching_name[matching_name['Team'].isin(teams)]
            else:
                player_data = pd.DataFrame()
            #return 'X' if player did not play in past box score
            if len(player_data) == 0:
                correct.append('X')
                continue
            prediction = [j for j in i.split(' ')[1:] if len(j) > 0]
            print(prediction)
            if prediction[0][0] == 'o':
                if prop_bet_type == 'strikeouts' or prop_bet_type == 'ks':
                    correct.append(self.evaluator('>', 'SO', None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'earned runs' or prop_bet_type == 'earned runs allowed':
                    correct.append(self.evaluator('>', 'ER', None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'hits allowed':
                    correct.append(self.evaluator('>', 'H', None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'walks allowed':
                    correct.append(self.evaluator('>', 'BB', None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'hits':
                    correct.append(self.evaluator('>', 'H', None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'home runs' or prop_bet_type == 'hr':
                    correct.append(self.evaluator('>', 'HRs', None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'rbi':
                    correct.append(self.evaluator('>', 'RBI', None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'hits+runs+rbi' or prop_bet_type == 'hits + runs + rbis':
                    correct.append(self.evaluator('>', 'H', 'R', 'RBI', player_data, float(prediction[0][1:])))
                if prop_bet_type == 'runs scored':
                    correct.append(self.evaluator('>', 'R',  None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'bb' or prop_bet_type == 'BB' or prop_bet_type == 'walks':
                    correct.append(self.evaluator('>', 'BB',  None, None,player_data, float(prediction[0][1:])))
                if prop_bet_type == 'total bases':
                    correct.append(self.evaluator('>', 'TB',  None, None, player_data, float(prediction[0][1:])))
            elif prediction[0][0] == 'u':
                if prop_bet_type == 'strikeouts' or prop_bet_type == 'ks':
                    correct.append(self.evaluator('<', 'SO', None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'earned runs' or prop_bet_type == 'earned runs allowed':
                    correct.append(self.evaluator('<', 'ER', None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'hits allowed':
                    correct.append(self.evaluator('<', 'H', None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'walks allowed':
                    correct.append(self.evaluator('<', 'BB', None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'hits':
                    correct.append(self.evaluator('<', 'H', None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'home runs' or prop_bet_type == 'hr':
                    correct.append(self.evaluator('<', 'HRs', None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'rbi':
                    correct.append(self.evaluator('<', 'RBI', None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'hits+runs+rbi' or prop_bet_type == 'hits + runs + rbis':
                    correct.append(self.evaluator('<', 'H', 'R', 'RBI', player_data, float(prediction[0][1:])))
                if prop_bet_type == 'runs scored':
                    correct.append(self.evaluator('<', 'R',  None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'bb' or prop_bet_type == 'BB' or prop_bet_type == 'walks':
                    correct.append(self.evaluator('<', 'BB',  None, None, player_data, float(prediction[0][1:])))
                if prop_bet_type == 'total bases':
                    correct.append(self.evaluator('<', 'TB',  None, None, player_data, float(prediction[0][1:])))
        predictions['Correct'] = correct
        return predictions.loc[predictions['Correct'] != 'X']

    def past_games_stats_evaluator(self, vals, prop_bet_number, is_over):
        #returns array of how many games player performs above or below given threshold
        ovr_avg_correct = []
        if is_over:
            for i in vals:
                if prop_bet_number < i:
                    ovr_avg_correct.append('Y')
                else:
                    ovr_avg_correct.append('N')
            return ovr_avg_correct
        else:
            for i in vals:
                if prop_bet_number > i:
                    ovr_avg_correct.append('Y')
                else:
                    ovr_avg_correct.append('N')
            return ovr_avg_correct

    def past_evaluator(self, games, prop_bet_type, prop_bet_number, is_over):
        #returns past game evaluations for prop types
        if len(games) == 0:
            return 0.0, 0
        if prop_bet_type == 'strikeouts' or prop_bet_type == 'ks':
            vals = games['SO'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'earned runs' or prop_bet_type == 'earned runs allowed':
            vals = games['ER'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'hits allowed':
            vals = games['H'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'walks allowed':
            vals = games['BB'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'hits':
            vals = games['H'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'home runs' or prop_bet_type == 'hr':
            vals = games['HRs'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'rbi':
            vals = games['RBI'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'runs scored':
            vals = games['R'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'BB' or prop_bet_type == 'bb' or prop_bet_type == 'walks':
            vals = games['BB'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'hits+runs+rbi' or prop_bet_type == 'hits + runs + rbis':
            vals = [v1 + v2 + v3 for v1, v2, v3 in zip(games['H'].values, games['R'].values, games['RBI'].values)]
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'total bases':
            vals = games['TB'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)

        ovr_avg_correct_prct = sum([1 for i in ovr_avg_correct if i == 'Y']) / len(ovr_avg_correct)
        if ovr_avg_correct_prct > 0.5:
            over_half = 1
        else:
            over_half = 0
        return ovr_avg_correct_prct, over_half

    def past_games_trends(self, predictions, hitting_box_score_results, pitching_box_score_results, is_evaluation):
        #returns evaluations if a player performs above or below given threshold for all games played, past 10 games played, past 5 games played
        all_games_prcts, all_games_trues, last5_prcts, last5_trues, last10_prcts, last10_trues = [], [], [], [], [], []
        for i in predictions.index:
            bet = predictions.loc[i]
            bet['Play'] = bet['Play'].replace('  ', ' ')
            bet['Play'] = bet['Play'].lower()
            name = bet['Play'].split(' ')[0]
            teams = [bet['Team']]
            prop = bet['Play'].split(' ')[1]
            #determine type of prop bets
            prop_bet_types = bet['Play'].split(' ')[2:]
            if len(prop_bet_types) > 1:
                prop_bet_type = ''
                for prop_bet in prop_bet_types:
                    prop_bet_type += prop_bet
                    prop_bet_type += ' '
            else:
                prop_bet_type = prop_bet_types[0]
            prop_bet_type = prop_bet_type.strip()
            over = prop[0] == 'o'
            prop_bet_number = float(prop[1:].split(' ')[0])
            
            #evaluate past trends based on identified player's past statistics 
            if prop_bet_type == 'ks' or prop_bet_type == 'strikeouts' or prop_bet_type == 'earned runs' or prop_bet_type == 'earned runs allowed' or prop_bet_type == 'hits allowed' or prop_bet_type == 'walks allowed':
                matching_name = pitching_box_score_results[pitching_box_score_results['Name'].str.lower() == name]
                all_games = matching_name[matching_name['Team'].isin(teams)]
            elif prop_bet_type == 'total bases' or prop_bet_type == 'hits' or prop_bet_type == 'home runs' or prop_bet_type == 'hr' or prop_bet_type == 'rbi' or prop_bet_type == 'hits+runs+rbi' or prop_bet_type == 'hits + runs + rbis' or prop_bet_type == 'runs scored' or prop_bet_type == 'bb' or prop_bet_type == 'walks':
                matching_name = hitting_box_score_results[hitting_box_score_results['Name'].str.lower() == name]
                all_games = matching_name[matching_name['Team'].isin(teams)]
            print(prop_bet_type)
            print(name, teams)
            last5_games = all_games[:5]
            last10_games =  all_games[:10]
            all_games_prct, all_games_true = self.past_evaluator(all_games, prop_bet_type, prop_bet_number, over)
            last5_prct, last5_true = self.past_evaluator(last5_games, prop_bet_type, prop_bet_number, over)
            last10_prct, last10_true = self.past_evaluator(last10_games, prop_bet_type, prop_bet_number, over)
            all_games_prcts.append(all_games_prct)
            all_games_trues.append(all_games_true)
            last5_prcts.append(last5_prct)
            last5_trues.append(last5_true)
            last10_prcts.append(last10_prct)
            last10_trues.append(last10_true)  

        predictions['All Game Percentages'], predictions['All Game Correct'], predictions['Last 5 Percentages'], predictions['Last 5 Correct'], predictions['Last 10 Percentages'], predictions['Last 10 Correct']  = all_games_prcts, all_games_trues, last5_prcts, last5_trues, last10_prcts, last10_trues
        #determine if inputs for evaluted predictions which have 'Correct' feature identified or current predictions which are to be determined
        if is_evaluation:
            features_list = ['Play', 'Expert', 'Odds',
            'Units', 'Payout', 'Team', 'Name', 'Opponent',
            'Hmcrt_adv', 'Profit', 'Correct', 'All Game Percentages',
            'All Game Correct', 'Last 5 Percentages', 'Last 5 Correct',
            'Last 10 Percentages', 'Last 10 Correct']
        else:
            features_list = ['Play', 'Expert', 'Team', 'Name', 'Opponent',
            'Hmcrt_adv', 'All Game Percentages',
        'All Game Correct', 'Last 5 Percentages', 'Last 5 Correct',
        'Last 10 Percentages', 'Last 10 Correct']
        return predictions[features_list]



    def optimized_predictions_evaluator(self, optimized_predictions, current_evaluation):
        #return evaluations of past optimized predictions
        optimized_correct = []
        for i in range(len(optimized_predictions)):
            optimized_prediction = optimized_predictions.loc[i]
            if len(current_evaluation[(current_evaluation['Play'] == optimized_prediction['Play']) & (current_evaluation['Expert'] == optimized_prediction['Expert'])  & (current_evaluation['Odds'] == optimized_prediction['Odds'])]) == 0:
                optimized_correct.append('X')
            else:
                optimized_correct.append(current_evaluation[(current_evaluation['Play'] == optimized_prediction['Play']) & (current_evaluation['Expert'] == optimized_prediction['Expert'])  & (current_evaluation['Odds'] == optimized_prediction['Odds'])]['Correct'].values[0])
        return optimized_correct

    def all_evaluations(self):
        #return cumulative evaluations
        all_evals = pd.DataFrame()
        for i in os.listdir('MLB-Bets-Evaluations/'):
            eval_ = pd.read_csv('MLB-Bets-Evaluations/' + i)
            all_evals = all_evals.append(eval_)
        return all_evals