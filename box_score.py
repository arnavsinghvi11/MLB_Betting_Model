from bs4 import BeautifulSoup, Comment
import date
import helper_functions
import pandas as pd
import re
import requests
import time
from unidecode import unidecode

class BoxScore:
    date = date.Date()

    def full_box_scores(self, month, day):
        #extract yesterday's box score statistics
        time.sleep(60)
        total_site_data = helper_functions.site_scrape('https://www.baseball-reference.com' + '?month=' + month + '&day=' + day + '&year=2023')
        print('got site data')
        time.sleep(60)
        links = []
        for a_href in total_site_data.find_all("a", href=True):
            if "boxes" in a_href["href"] and "shtml" in a_href["href"]:
                links.append((a_href["href"]))
        links = ['https://www.baseball-reference.com' + i for i in links]
        hitting_box_scores, pitching_box_scores = pd.DataFrame(), pd.DataFrame()
        for link in links:
            time.sleep(10)
            print('each link')
            columns_hitting = {'Name': [], 'Team': [], 'Opponent': [], 'Hmcrt_adv': [], 'AB': [], 'R': [], 'H': [], 'RBI': [], 'BB': [], 'SO': [], 'PA': []}
            a = []
            page = requests.get(link)
            soup = BeautifulSoup(page.content, 'lxml')
            for comment in soup.findAll(text=lambda text: isinstance(text, Comment)):
                a.append(comment)
            b = []
            for val in a:
                if 'player' in val:
                    b.append(val)
            hitting = b[:2]
            for val in range(len(hitting)):
                matches = re.findall(r'<caption>(.*?)<\/caption>', hitting[val], flags=re.DOTALL)
                filtered_matches = [match.replace('Table', '').strip() for match in matches]
                if len(filtered_matches) == 1:
                    if val == 0:
                        away = filtered_matches[0]
                    else:
                        home = filtered_matches[0]
            home_team, away_team = None, None
            for val in range(len(hitting)):
                matches = re.findall(r'<caption>(.*?)<\/caption>', hitting[val], flags=re.DOTALL)
                filtered_matches = [match.replace('Table', '').strip() for match in matches]
                if len(filtered_matches) == 1:
                    if val == 0:
                        away_team = filtered_matches[0]
                        home_team = None
                    else:
                        home_team = filtered_matches[0]
                        away_team = None
                x = hitting[val].split('.shtml">')
                x = x[1:]
                for i in range(len(x)):
                    if '</a> P</th>' not in x[i]:
                        name = x[i].split('</a>')[0]
                        stats = re.findall('["][A-Za-z]+["]', x[i].split('</a>')[1])
                        stats_values = re.findall(">\d+<", x[i].split('</a>')[1])
                        stats = [j.replace('"', '') for j in stats][:7]
                        stats_values = [j.replace('>', '') for j in stats_values]
                        stats_values = [j.replace('<', '') for j in stats_values][:7]
                        if away_team:
                            for value in reversed([name, away_team, home, 0]):
                                stats_values.insert(0, value)
                        if home_team:
                            for value in reversed([name, home_team, away, 1]):
                                stats_values.insert(0, value)
                        if len(stats_values) == len(columns_hitting):
                            count = 0
                            for key in columns_hitting.keys():
                                columns_hitting[key].append(stats_values[count])
                                count += 1
            hitting_box_score = pd.DataFrame.from_dict(columns_hitting)
            #add home and away teams logic 
            # Adding Total Bases to dataframe
            if len(hitting[0].split('TB:')) != 1:
                total_base_values = re.findall("( [A-Z]\.[A-Z]\.)| ([a-z'A-z]+)|([A-Z][a-zA-z]+)|( \d+)",
                                            hitting[0].split('TB:')[1].split('</div')[0])
                indexer = -1
                lst = []
                for tb in total_base_values:
                    for data in tb:
                        if data:
                            lst.append(data.strip())
                tb_entry = []
                if len(lst) < 3:
                    name = lst[0] + ' ' + lst[1]
                    tb_entry.append([name, 1])
                else:
                    for data in range(2, len(lst), 3):
                        if str(lst[data]).strip().isnumeric():
                            name = str(lst[data - 2]) + ' ' + str(lst[data - 1])
                            tb_entry.append([name, int(str(lst[data]).strip())])
                        else:
                            break
                    if len(lst) - data <= 3:
                        name = str(lst[len(lst) - 2]) + ' ' + str(lst[len(lst) - 1])
                        tb_entry.append([name, 1])
                    else:
                        for data1 in range(data, len(lst) + 1, 2):
                            name = str(lst[data1 - 2]) + ' ' + str(lst[data1 - 1])
                            tb_entry.append([name, 1])
                hitting_box_score['TB'] = 0
                for plyr in tb_entry:
                    hitting_box_score.loc[hitting_box_score['Name'] == plyr[0], 'TB'] = plyr[1]
            else:
                hitting_box_score['TB'] = 0

            if len(hitting[0].split('HR:')) != 1:
                homerun_values = re.findall("( [A-z ,.'-]+ \()|( [A-z ,.'-]+ \d+ \()",
                                            hitting[0].split('HR:')[1].split('</div>')[0])
                hr_values = []
                for j in homerun_values:
                    for k in j:
                        if k:
                            hr_values.append(k)
                hr_values = [hr.strip().replace('(', '').strip() for hr in hr_values]
                hr_data = []
                for j in hr_values:
                    if j[-1].isnumeric():
                        hr_data.append([j[:-2], j[-1]])
                    else:
                        hr_data.append([j, 1])
                hitting_box_score['HRs'] = 0
                for plyr in hr_data:
                    hitting_box_score.loc[hitting_box_score['Name'] == plyr[0], 'HRs'] = int(plyr[1])
            else:
                hitting_box_score['HRs'] = 0

            if len(hitting[1].split('TB:')) != 1:
                total_base_values = re.findall("( [A-Z]\.[A-Z]\.)| ([a-z'A-z]+)|([A-Z][a-zA-z]+)|( \d+)",
                                            hitting[1].split('TB:')[1].split('</div')[0])
                indexer = -1
                lst = []
                for tb in total_base_values:
                    for data in tb:
                        if data:
                            lst.append(data.strip())
                tb_entry = []
                if len(lst) < 3:
                    name = lst[0] + ' ' + lst[1]
                    tb_entry.append([name, 1])
                else:
                    for data in range(2, len(lst), 3):
                        if str(lst[data]).strip().isnumeric():
                            name = str(lst[data - 2]) + ' ' + str(lst[data - 1])
                            tb_entry.append([name, int(str(lst[data]).strip())])
                        else:
                            break
                    if len(lst) - data <= 3:
                        name = str(lst[len(lst) - 2]) + ' ' + str(lst[len(lst) - 1])
                        tb_entry.append([name, 1])
                    else:
                        for data1 in range(data, len(lst) + 1, 2):
                            name = str(lst[data1 - 2]) + ' ' + str(lst[data1 - 1])
                            tb_entry.append([name, 1])
                for plyr in tb_entry:
                    hitting_box_score.loc[hitting_box_score['Name'] == plyr[0], 'TB'] = plyr[1]
            else:
                hitting_box_score['TB'] = 0

            if len(hitting[1].split('HR:')) != 1:
                homerun_values = re.findall("( [A-z ,.'-]+ \()|( [A-z ,.'-]+ \d+ \()",
                                            hitting[1].split('HR:')[1].split('</div>')[0])
                hr_values = []
                for j in homerun_values:
                    for k in j:
                        if k:
                            hr_values.append(k)
                hr_values = [hr.strip().replace('(', '').strip() for hr in hr_values]
                hr_data = []
                for j in hr_values:
                    if j[-1].isnumeric():
                        hr_data.append([j[:-2], j[-1]])
                    else:
                        hr_data.append([j, 1])
                for plyr in hr_data:
                    hitting_box_score.loc[hitting_box_score['Name'] == plyr[0], 'HRs'] = int(plyr[1])
            else:
                hitting_box_score['HRs'] = 0

            hitting_box_scores = hitting_box_scores.append(hitting_box_score)
            pitching = [p for p in b[2:] if 'pitching' in p][0]
            pitching = pitching.split('data-cols-to-freeze')[1:]
            columns_pitching = {'Name': [], 'Team': [], 'Opponent': [], 'Hmcrt_adv': [], 'IP': [], 'H': [], 'R': [], 'ER': [], 'BB': [], 'SO': [], 'HR': []}

            for val in range(len(pitching)):
                matches = re.findall(r'<caption>(.*?)<\/caption>', pitching[val], flags=re.DOTALL)
                filtered_matches = [match.replace('Table', '').strip() for match in matches if 'Play' not in match.replace('Table', '')]
                if len(filtered_matches) == 1:
                    if val == 0:
                        away = filtered_matches[0]
                    else:
                        home = filtered_matches[0]
            home_team, away_team = None, None
            for val in range(len(pitching)):
                matches = re.findall(r'<caption>(.*?)<\/caption>', pitching[val], flags=re.DOTALL)
                filtered_matches = [match.replace('Table', '').strip() for match in matches if 'Play' not in match.replace('Table', '')]
                if len(filtered_matches) == 1:
                    if val == 0:
                        away_team = filtered_matches[0]
                        home_team = None
                    else:
                        home_team = filtered_matches[0]
                        away_team = None
                x = pitching[val].split('.shtml">')
                x = x[1:]
                for i in range(len(x)):
                    if '</a> P</th>' not in x[i]:
                        name = x[i].split('</a>')[0]
                        stats = re.findall('["][A-Za-z]+["]', x[i].split('</a>')[1])[:7]
                        stats_values = re.findall("(>\d+<)|(> \d*\.?\d* <)|(> \d*\.?\d*<)",
                                                    x[i].split('</a>')[1])[:7]
                        new_stats_values = []
                        for j in stats_values:
                            for k in j:
                                if k:
                                    new_stats_values.append(k)
                        new_stats_values = [j.replace('>', '') for j in new_stats_values]
                        new_stats_values = [j.replace('<', '') for j in new_stats_values]
                        new_stats_values = [j.strip() for j in new_stats_values]
                        if away_team:
                            for value in reversed([name, away_team, home, 0]):
                                new_stats_values.insert(0, value)
                        if home_team:
                            for value in reversed([name, home_team, away, 1]):
                                new_stats_values.insert(0, value)
                        if len(new_stats_values) == len(columns_pitching):
                            count = 0
                            for key in columns_pitching.keys():
                                columns_pitching[key].append(new_stats_values[count])
                                count += 1
            pitching_box_score = pd.DataFrame.from_dict(columns_pitching)
            pitching_box_scores = pitching_box_scores.append(pitching_box_score)
        hitting_box_scores = hitting_box_scores.astype(
            {'AB': int, 'R': int, 'H': int, 'RBI': int, 'BB': int, 'SO': int, 'PA': int, 'TB': int, 'HRs': int}).drop_duplicates()
        pitching_box_scores = pitching_box_scores.astype(
            {'IP': float, 'H': int, 'R': int, 'ER': int, 'BB': int, 'SO': int, 'HR': int}).drop_duplicates()

        return hitting_box_scores, pitching_box_scores

    def update_all_box_score_results(self,  date, month, day, is_hitting):
        #returns cumulative box score statistics for players
        if is_hitting == 'Hitting':
            box_type = is_hitting
            columns = ['Date','Name','Team', 'Opponent', 'Hmcrt_adv', 'AB', 'R', 'H', 'RBI', 'BB', 'SO', 'PA', 'TB', 'HRs']
        else:
            box_type = 'Pitching'
            columns = ['Date','Name','Team', 'Opponent', 'Hmcrt_adv', 'IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR']
        print(f'All_{box_type}_Box_Score_Results.csv')
        all_box_score_results = pd.read_csv(f'All_{box_type}_Box_Score_Results.csv')
        box_score = pd.read_csv(f"MLB-Box-Score-Results/{box_type}/" + f"MLB-Bets-{box_type}-Box-Score-Results-" + month.lstrip('0') + "-" + day.lstrip('0') +".csv")
        box_score['Date'] = month + "-" + day
        box_score['Name'] = box_score['Name'].apply(unidecode)
        box_score['Name'] = box_score['Name'].apply(helper_functions.abbrv)
        box_score['Date'] = box_score['Date'].apply(date.date_converter)
        all_box_score_results = all_box_score_results.append(box_score)
        all_box_score_results.to_csv(f'All_{box_type}_Box_Score_Results.csv')
        
        #formatting for names and dates for predictions/evaluations 
        all_box_score_results['Date'] = all_box_score_results['Date'].astype('int')
        all_box_score_results = all_box_score_results.sort_values(by=['Date'], ascending = False)
        all_box_score_results.set_index(['Name','Team']).index.factorize()[0]+1
        all_box_score_results = all_box_score_results.drop_duplicates(columns, keep='last')
        return all_box_score_results