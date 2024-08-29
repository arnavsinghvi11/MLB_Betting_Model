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
        def process_html_string(html_string):
            results = {'2B': [], '3B': [], 'HR': [], 'TB': []}
            for line in html_string.split('<div'):
                if "<strong>2B:</strong>" in line:
                    category = '2B'
                elif "<strong>3B:</strong>" in line:
                    category = '3B'
                elif "<strong>HR:</strong>" in line:
                    category = 'HR'
                elif "<strong>TB:</strong>" in line:
                    category = 'TB'
                else:
                    continue
                data1 = line.split('>', 2)[-1].rsplit('</div>', 1)[0]
                if category != 'TB':
                    entries = data1.split(';')
                else:
                    entries = data1.replace('&nbsp;', ' ').split(';')
                data = {}
                for entry in entries:
                    entry = entry.replace(f"{category}:</strong>", '').strip()
                    if category != 'TB':
                        if '(' not in entry:
                            continue
                        name = entry.split('(')[0].strip()
                        count = int(entry.split('(')[1].split(',')[0])
                        name_parts = name.split(' ')
                        clean_name = ' '.join(filter(lambda x: not x.isdigit(), name_parts))
                    else:
                        parts = entry.split()
                        if parts:
                            try:
                                count = int(parts[-1])
                                name_parts = parts[:-1]
                            except ValueError:
                                count = 1
                                name_parts = parts
                            clean_name = ' '.join(name_parts).split('.')[0]
                    data[clean_name] = count
                results[category].append(data)
            return results
        def process_ip_value(ip_str):
            if '.' in ip_str:
                first, second = ip_str.split('.')
                return int(first) * 3 + int(second)
            else:
                return int(ip_str) * 3
        time.sleep(60)
        total_site_data = helper_functions.site_scrape('https://www.baseball-reference.com/boxes/' + '?year=2024' + '&month=' + month + '&day=' + day)
        time.sleep(60)
        links = []
        for a_href in total_site_data.find_all("a", href=True):
            if "boxes" in a_href["href"] and "shtml" in a_href["href"]:
                links.append((a_href["href"]))
        links = ['https://www.baseball-reference.com' + i for i in links]
        hitting_box_scores, pitching_box_scores = pd.DataFrame(), pd.DataFrame()
        for link in links:
            time.sleep(10)
            page = requests.get(link)
            soup = BeautifulSoup(page.content)
            title = soup.title.string
            match = re.match(r"(.+) vs (.+) Box Score", title)
            if match:
                away = match.group(1)
                home = match.group(2)
            else:
                away, home = None, None
            divs = soup.find_all("div", id=lambda x: x and x.endswith("batting"))
            for div in divs:
                inner_html = ''.join([str(x) for x in div.contents])
                columns = ['Name', 'Team', 'Opponent', 'Hmcrt_adv',  '1B', '2B', '3B', 'AB', 'BB', 'H', 'HR', 'OBP', 'OPS', 'PA', 'R', 'RBI', 'RE24', 'SLG', 'SO', 'TB', 'WPA', 'aLI', 'acLI', 'cWPA', 'H+R+RBI']
                data_rows = []
                page_content = inner_html
                team_sections = page_content.split('<div class="table_container"')[1:]
                footer_section = re.search(r'<div class="footer[^>]*>(.+)</div>', page_content, re.DOTALL).group(1)
                for section in team_sections:
                    hmcrt_adv, opp = None, None
                    team_name_raw = re.search(r'id="div_(.+?)batting"', section).group(1)
                    team_name = re.sub(r'([A-Z])', r' \1', team_name_raw).strip()
                    if team_name == home:
                        hmcrt_adv = 1
                        opp = away
                    else:
                        hmcrt_adv = 0
                        opp = home
                    player_rows = re.findall(r'<tr[^>]*>(.+?)</tr>', section, re.DOTALL)
                    for row in player_rows:
                        if 'scope="col"' in row or 'Team Totals' in row:
                            continue
                        soup_row = BeautifulSoup(row, 'html.parser')
                        player_name_tag = soup_row.find('th', {'data-stat': 'player'}).find('a')
                        player_name = player_name_tag.text if player_name_tag else "Unknown Player"
                        tds = soup_row.find_all('td')
                        player_data = {}
                        for td in tds:
                            stat = td.get('data-stat')
                            value = td.text
                            player_data[stat] = value
                        data = {
                            'Team': team_name,
                            'Name': player_name,
                            'Hmcrt_adv': hmcrt_adv,
                            'Opponent': opp,
                            'AB': player_data.get('AB', ''),
                            'BB': player_data.get('BB', ''),
                            'H': player_data.get('H', ''),
                            'HR': '',
                            'OBP': player_data.get('onbase_perc', ''),
                            'OPS': player_data.get('onbase_plus_slugging', ''),
                            'PA': player_data.get('PA', ''),
                            'R': player_data.get('R', ''),
                            'RBI': player_data.get('RBI', ''),
                            'RE24': player_data.get('re24_bat', '').strip(),
                            'SLG': player_data.get('slugging_perc', ''),
                            'SO': player_data.get('SO', ''),
                            'TB': '',
                            'WPA': player_data.get('wpa_bat', ''),
                            'aLI': player_data.get('leverage_index_avg', '').strip(),
                            'acLI': player_data.get('cli_avg', '').strip(),
                            'cWPA': player_data.get('cwpa_bat', '').strip().replace('%', ''),
                        }
                        h_r_rbi = None
                        if all(player_data.get(stat, '') not in [None, ''] for stat in ['H', 'R', 'RBI']):
                            h_r_rbi = str(int(player_data.get('H', '0')) + int(player_data.get('R', '0')) + int(player_data.get('RBI', '0')))
                        data['H+R+RBI'] = h_r_rbi
                        data_rows.append(data)
                        name_match = soup_row.find('td', {'data-stat': 'player'})
                        if name_match and player_data:
                            name = name_match.text
                            data = {'Name': name, 'Team': team_name}
                            for col in columns[2:]:
                                data[col] = player_data.get(col.lower(), '')
                            data_rows.append(data)
                    columns_order = [
                        'Name', 'Team', 'Opponent', 'Hmcrt_adv', 'AB', 'R', 'H', 'RBI', 'BB', 'SO', 'PA', 'OBP', 'SLG', 'OPS', 'WPA', 'aLI', 'cWPA', 'acLI',  'RE24', 'H+R+RBI', '1B', '2B', '3B', 'HR', 'TB'
                    ]
                    df = pd.DataFrame(data_rows, columns=columns_order)
                    df = df[df['PA'] != '']
                    pd.set_option('display.max_columns', None)
                    results = process_html_string(footer_section)
                    df['2B'] = 0
                    df['3B'] = 0
                    df['HR'] = 0
                    df['TB'] = 0
                    for category in ['2B', '3B', 'HR', 'TB']:
                        for player_dict in results[category]:
                            for name, count in player_dict.items():
                                df.loc[df['Name'] == name, category] = count
                    df['H'] = pd.to_numeric(df['H'], errors='coerce')
                    df['2B'] = pd.to_numeric(df['2B'], errors='coerce')
                    df['3B'] = pd.to_numeric(df['3B'], errors='coerce')
                    df['HR'] = pd.to_numeric(df['HR'], errors='coerce')
                    df['1B'] = df['H'] - (df['2B'] + df['3B'] + df['HR'])
                    df = df[df['PA'] != '0']
                    hitting_box_scores = pd.concat([hitting_box_scores, df], ignore_index=True)

            def class_ends_with(soup, ending):
                result = []
                for tag in soup.find_all("div", class_=True):  
                    if any(cls.endswith(ending) for cls in tag.get("class")):
                        if 'pitching' in str(tag):
                            result.append(tag)
                return result

            divs = class_ends_with(soup, "setup_commented")
            for div in divs:
                inner_html = str(div)
                if 'View Pitches' in inner_html:
                    continue
                columns = ['Name', 'Team', 'Opponent', 'Hmcrt_adv',  'IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'ERA', 'Pit', 'Str', 'Ctct', 'StS', 'StL', 'GB', 'FB', 'GSc', 'WPA', 'aLI', 'cWPA', 'acLI',  'RE24', 'PO']
                data_rows = []
                page_content = inner_html
                team_sections = page_content.split('<div class="table_container"')[1:]
                for section in team_sections:
                    hmcrt_adv, opp = None, None
                    team_name_raw = re.search(r'id="div_(.+?)pitching"', section).group(1)
                    team_name = re.sub(r'([A-Z])', r' \1', team_name_raw).strip()
                    if team_name == home:
                        hmcrt_adv = 1
                        opp = away
                    else:
                        hmcrt_adv = 0
                        opp = home
                    player_rows = re.findall(r'<tr[^>]*>(.+?)</tr>', section, re.DOTALL)
                    for row in player_rows:
                        if 'scope="col"' in row or 'Team Totals' in row:
                            continue
                        soup_row = BeautifulSoup(row, 'html.parser')
                        player_name_tag = soup_row.find('th', {'data-stat': 'player'}).find('a')
                        player_name = player_name_tag.text if player_name_tag else "Unknown Player"
                        tds = soup_row.find_all('td')
                        player_data = {}
                        for td in tds:
                            stat = td.get('data-stat')
                            value = td.text
                            player_data[stat] = value
                        data = {
                            'Team': team_name,
                            'Name': player_name,
                            'Hmcrt_adv': hmcrt_adv,
                            'Opponent': opp,
                            'IP': player_data.get('IP', ''),
                            'H': player_data.get('H', ''),
                            'R': player_data.get('R', ''),
                            'ER': player_data.get('ER', ''),
                            'BB': player_data.get('BB', ''),
                            'SO': player_data.get('SO', ''),
                            'HR': player_data.get('HR', ''),
                            'ERA': player_data.get('earned_run_avg', ''),
                            'Pit': player_data.get('pitches', ''),
                            'Str': player_data.get('strikes_total', ''),
                            'Ctct': player_data.get('strikes_contact', ''),
                            'StS': player_data.get('strikes_swinging', ''),
                            'StL': player_data.get('strikes_looking', ''),
                            'GB': player_data.get('inplay_gb_total', ''),
                            'FB': player_data.get('inplay_fb_total', ''),
                            'GSc': player_data.get('game_score', ''),
                            'WPA': player_data.get('wpa_def', ''),
                            'aLI': player_data.get('leverage_index_avg', '').strip(),
                            'cWPA': player_data.get('cwpa_def', '').strip().replace('%', ''),
                            'acLI': player_data.get('cli_avg', '').strip(),
                            'RE24': player_data.get('re24_def', '').strip(),
                            'PO': '',
                        }
                        po = None
                        data['PO'] = str(process_ip_value(player_data.get('IP', '0')))
                        data_rows.append(data)
                        name_match = soup_row.find('td', {'data-stat': 'player'})
                        if name_match and player_data:
                            name = name_match.text
                            data = {'Name': name, 'Team': team_name}
                            for col in columns[2:]:
                                data[col] = player_data.get(col.lower(), '')
                            data_rows.append(data)
                        df = pd.DataFrame(data_rows, columns=columns)
                        df = df[df['GSc'] != '']
                        pitching_box_scores = pd.concat([pitching_box_scores, df], ignore_index=True)
                        pitching_box_scores = pitching_box_scores.drop_duplicates()

        hitting_box_scores = hitting_box_scores.loc[:, ~hitting_box_scores.columns.str.contains('^Unnamed')]
        pitching_box_scores = pitching_box_scores.loc[:, ~pitching_box_scores.columns.str.contains('^Unnamed')]
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