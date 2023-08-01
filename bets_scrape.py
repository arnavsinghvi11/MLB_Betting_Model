from bs4 import BeautifulSoup
import chromedriver_autoinstaller
import date
import datetime
from datetime import datetime
from datetime import timedelta
import pytz
from pytz import timezone
import os
import numpy as np
import pandas as pd
import regex as re
import requests
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
import chromedriver_autoinstaller
import time


class Bets:
    date = date.Date()

    def site_scrape_chrome(self, url):
        # initiating webdriver settings for Google Chrome
        path = chromedriver_autoinstaller.install()
        os.environ["LANG"] = "en_US.UTF-8"
        options = Options()
        options.headless = True
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--no-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--dns-prefetch-disable')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        driver = webdriver.Chrome(options=options)
        tz_params = {'timezoneId': 'America/Los_Angeles'}
        driver.execute_cdp_cmd('Emulation.setTimezoneOverride', tz_params)
        driver.get(url)
        time.sleep(5)
        html = driver.page_source

        # apply BeautifulSoup to scrape web page contents
        soup = BeautifulSoup(html, "html.parser")
        driver.close()  # closing the webdriver
        return soup

    def today_games(self, month, day):
        if int(day) < 10:
            soup = self.site_scrape_chrome('https://theathletic.com/mlb/schedule/2023-' + '0' + month + '-' + '0' + day + '/')
        else:
            soup = self.site_scrape_chrome('https://theathletic.com/mlb/schedule/2023-' + '0' + month + '-' + day + '/')
        games = soup.select('tr.MuiTableRow-root .jss6')
        matchups = []
        for i in range(0, len(games), 2):
            home_team = str(games[i]).split('>')[2].split('<')[0]
            away_team = str(games[i+1]).split('>')[2].split('<')[0]
            matchup = f"{home_team} vs {away_team}"
            matchups.append(matchup)
        return matchups

    def schedule(self, month, day):
        #returning time 10 mins before earliest scheduled MLB game for today
        #figure out how date is entered here
        if int(day) < 10:
            soup = self.site_scrape_chrome('https://theathletic.com/mlb/schedule/2023-' + '0' + month + '-' + '0' + day + '/')
        else:
            soup = self.site_scrape_chrome('https://theathletic.com/mlb/schedule/2023-' + '0' + month + '-' + day + '/')
        # running_time = str(pacific_time.hour) + ':' + \
        #     str(pacific_time.minute) + ':00'
        earliest = soup.select_one('tr.MuiTableRow-root a[href^="https://theathletic.com/mlb/game/"]').text
        if int(earliest.split(':')[0]) < 13 and int(earliest.split(':')[0]) > 8:
            pacific_time = datetime.strptime((str(int(earliest.split(':')[0])) + ':' + earliest.split(':')[1]).strip().split(' ')[0], "%H:%M") - timedelta(minutes=15)
        else:
            pacific_time = datetime.strptime((str(int(earliest.split(':')[0]) + 12) + ':' + earliest.split(':')[1]).strip().split(' ')[0], "%H:%M") - timedelta(minutes=15)
        running_time = str(pacific_time.hour) + ':' + \
            str(pacific_time.minute) + ':00'
        print(running_time)
        return str(running_time)
    
    def calculate_payout(self, row):
        odds = row['Odds']
        units = row['Units']
        if int(odds) > 0:
            multiplier = (int(odds) / 100) + 1
        else:
            multiplier = (-100 / int(odds)) + 1
        return float(multiplier * units)

    def extract_prop_type(self, props):
        match = re.search(r'\b(\w+(?: \w+)*) (?:o|u)\d+(?:\.\d+)? (\w+(?: \w+)*)\b', props)
        if match:
            return match.group(2)
        return None

    def draftkings_scrape_player_prop_bets(self, url):
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        prop_bets = []
        prop_bet_elements = soup.find_all("h3")

        for prop_bet_element in prop_bet_elements:
            prop_bet = prop_bet_element.text.strip()
            if prop_bet.startswith("MLB player prop bets:") or prop_bet.startswith("Share") or prop_bet.startswith("All sharing options") or prop_bet.startswith("Best MLB player prop bets") or prop_bet.endswith("Featured Videos"):
                continue
            prop_bets.append(prop_bet)

        return prop_bets

    # Scrape author from a given URL
    def draftkings_scrape_author(self, url):
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        author_meta = soup.find("meta", property="author")
        if author_meta:
            author = author_meta["content"]
            return author
        else:
            return None


    # Function to process the entries in the 'Props' column
    def draftkings_entry(self, entry):
        entry = entry.lower().strip()
        if re.match(r'[ou]\d+\.\d+', entry):
            return entry
        entry = re.sub(r'\bO\b', 'o', entry)
        entry = re.sub(r'\bU\b', 'u', entry)
        entry = re.sub(r'(over|under)\s+(\d+\.\d+)', r'\1\2', entry)
        return entry

    def draftkings(self, dk_date):
        draftkings_props, draftkings_authors = [], []
        links = []

        page = requests.get('https://dknation.draftkings.com/mlb-odds-lines/archives/')
        soup = BeautifulSoup(page.content, "html.parser")

        for a_href in soup.find_all("a", href=True):
            if "prop-bets" in a_href["href"] and dk_date in a_href["href"]:
                links.append(a_href["href"])
        if not links:
            return pd.DataFrame()
        prop_bets = []
        authors = []
        
        for link in set(links):
            draftkings_props += self.draftkings_scrape_player_prop_bets(link)
            author = self.draftkings_scrape_author(link)
            if draftkings_props and author:
                for prop in draftkings_props:
                    draftkings_authors.append(author)
                df = pd.DataFrame({'Props': draftkings_props,'Expert': draftkings_authors})
                if len(df) < 1:
                    return pd.DataFrame()
                df = df[~df['Props'].str.contains(r'\b(and|win|outs)\b', case=False)]
                df['Props'] = df['Props'].str.replace(' Jr.', '')
                df['Props'] = df['Props'].str.replace('hits + runs + rbis', 'hits+runs+rbi')
                df['Props'] = df['Props'].apply(self.draftkings_entry)
                df['Props'] = df['Props'].str.replace(r'over(\d+\.\d+)', r'o\1')
                df['Props'] = df['Props'].str.replace(r'under(\d+\.\d+)', r'u\1')
                df['Props'] = df['Props'].str.replace(r'o\s+(\d+\.\d+)', r'o\1')
                df['Props'] = df['Props'].str.replace(r'u\s+(\d+\.\d+)', r'u\1')
                # Save the df DataFrame as a single CSV file
                df['First Initial'] = df['Props'].str.split().str[:1].str.join(' ').str.replace('[,.]', '').str[:1].str.capitalize() + '.'
                df['Last Name'] = df['Props'].str.split().str[1:2].str.join(' ').str.replace('[,.]', '').str.capitalize()
                df['Name'] = df['First Initial'] + df['Last Name'] 
                df['Prop Num'] = df['Props'].str.split().str[2:3].str.join(' ')
                df['Prop Type'] = df['Props'].str.extract(r'5\s(.+?)\s\(')
                df['Odds'] = df['Props'].str.extract(r'\(([-+]\d+)\)')
                df['Units'] = 1
                df = df.dropna()
                df['Payout'] = df.apply(self.calculate_payout, axis=1)
                df['Profit'] = df['Payout'] - df['Units']
                df['Play'] = df['Name'] + ' ' + df['Prop Num'] + ' ' + df['Prop Type']
                df = df.dropna()
                df = df.drop_duplicates(subset=['Play', 'Expert'], keep='first').reset_index(drop=True)
                # return df[['Play', 'Prop Type', 'Authors', 'Odds', 'Units', 'Payout', 'Name']] 
                return df[['Play', 'Expert', 'Odds', 'Units', 'Payout', 'Profit', 'Name']]
            else:
                return pd.DataFrame()

    def underdog_scrape_props(self, url):
        soup = self.site_scrape_chrome(url)
        props = []
        prop_tags = soup.find_all('h2')
        for prop_tag in prop_tags:
            if prop_tag.find('a'):
                player_name = prop_tag.find('a').text
                prop_text = prop_tag.text.strip()
                if '-' in prop_text:
                    separator = '-'
                elif '–' in prop_text:
                    separator = '–'
                else:
                    separator = None
                if separator:
                    prop_type = prop_text.split(separator)[1].strip()
                else:
                    prop_type = prop_text.replace(player_name, '').strip() 
                props.append(player_name + ' ' + prop_type)
        author = str(soup).split('"@type": "Person", "name": "')[1].split('"')[0]
        return props, author
    
    def convert_value(self, value):
        match = re.search(r'[ou]\s?\d+\.?\d* ', value)
        if match:
            match_value = match.group()
            letter = match_value[0]
            number = float(match_value[1:])
            print(str(number)[-1:])
            if letter == 'o' and str(number)[-1:] != '5':
                number -= 0.5
            elif letter == 'u' and str(number)[-1:] != '5':
                number += 0.5
            return re.sub(r'[ou]\s?\d+\.?\d* ', f'{letter}{number} ', value)
        return value

    def underdog(self, underdog_date):
        underdog_props, underdog_authors = [], []
        underdog_url = "https://www.fantasyalarm.com/articles/mlb/mlb-player-props/underdog-fantasy"
        underdog_link = ""
        page = requests.get(underdog_url)
        soup = BeautifulSoup(page.content, "html.parser")
        for a_href in soup.find_all("a", href=True):
            if underdog_date in a_href["href"]:
                underdog_link = a_href["href"]
                break
        if not underdog_link:
            return pd.DataFrame()
        else:
            props, author = self.underdog_scrape_props(underdog_link)
            for prop in props:
                underdog_props.append(prop)
                underdog_authors.append(author)
            df = pd.DataFrame({'Props': underdog_props, 'Expert': underdog_authors})
            df = df[~df['Props'].str.contains(r'\b(singles|count|fantasy|and|win|outs)\b', case=False) & ~df['Props'].str.contains(r'&')]
            if len(df) < 1:
                return pd.DataFrame()
            df['Props'] = df['Props'].str.replace(' Jr.', '')
            df['Props'] = df['Props'].str.replace('runs', 'Runs')
            df['Props'] = df['Props'].str.replace('strikeouts', 'Strikeouts')
            df['Props'] = df['Props'].str.replace('RBIs', 'RBI')
            df['Props'] = df['Props'].str.replace(r'(OVER|UNDER) (\d+\.?\d*)', r'\1\2')
            df['Props'] = df['Props'].str.replace('OVER', 'o').str.replace('UNDER', 'u')
            df['Props'] = df['Props'].str.replace('HIGHER than', 'o').str.replace('LOWER than', 'u')
            df['Props'] = df['Props'].str.replace('MORE than', 'o').str.replace('LESS than', 'u')
            df['Props'] = df['Props'].apply(self.convert_value)
            df['First Initial'] = df['Props'].str.split().str[:1].str.join(' ').str.replace('[,.]', '').str[:1].str.capitalize() + '.'
            df['Last Name'] = df['Props'].str.split().str[1:2].str.join(' ').str.replace('[,.]', '').str.capitalize()
            df['Name'] = df['First Initial'] + df['Last Name'] 
            df['Prop Num'] = df['Props'].str.split().str[2:3].str.join(' ')
            df['Prop Type'] = df['Props'].apply(self.extract_prop_type)
            df['Odds'] = -110
            df['Units'] = 1
            print(df)
            df = df.dropna()
            df['Payout'] = df.apply(self.calculate_payout, axis=1).astype(float)
            df['Units'] = df['Units'].astype(float)
            df['Profit'] = df['Payout'] - df['Units']
            df['Play'] = df['Name'] + ' ' + df['Prop Num'] + ' ' + df['Prop Type']
            df = df.dropna()
            return df[['Play', 'Expert', 'Odds', 'Units', 'Payout', 'Profit', 'Name']]

    def get_modified_html(self, initial_html, updated_html):
        initial_soup = BeautifulSoup(initial_html, "html.parser")
        updated_soup = BeautifulSoup(updated_html, "html.parser")
        modified_parts = []
        for initial_tag, updated_tag in zip(initial_soup.find_all(), updated_soup.find_all()):
            if str(initial_tag) != str(updated_tag):
                modified_parts.append(updated_tag)
        modified_html = '\n'.join(str(tag) for tag in modified_parts)
        return modified_html

    def calculate_payout(self, row):
        odds = row['Odds']
        units = row['Units']
        if int(odds) > 0:
            multiplier = (int(odds) / 100) + 1
        else:
            multiplier = (-100 / int(odds)) + 1
        return float(multiplier) * float(units)

    def site_scrape_chrome_action_button(self, url):
        path = chromedriver_autoinstaller.install()
        options = Options()
        options.headless = True
        options.add_argument("disable-infobars")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        driver = webdriver.Chrome(service=Service(executable_path=path), options=options)
        tz_params = {'timezoneId': 'America/Los_Angeles'}
        driver.execute_cdp_cmd('Emulation.setTimezoneOverride', tz_params)
        driver.get(url)
        time.sleep(5)
        
        date_format = '%m/%d/%Y %H:%M:%S %Z'
        date = datetime.now(timezone('US/Pacific'))
        date = date.astimezone(timezone('US/Pacific'))
        entries = driver.find_elements(By.CSS_SELECTOR, '.css-1pfshtc.e11h3jb20')
        print(len(entries))
        initial_html = driver.page_source
        #maybe add validation for action_date later
        modified_htmls = []
        for entry in entries:
            print('each button')
            driver.execute_script("arguments[0].scrollIntoView(true);", entry)
            see_all_button = entry.find_element(By.CSS_SELECTOR, 'button[data-testid="expert-picks__see-all-picks"]')
            actions = ActionChains(driver)
            actions.move_to_element(see_all_button).click().perform()
            WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.game-pick-modal__info')))        
            time.sleep(10)
            updated_html = driver.page_source
            soup = BeautifulSoup(updated_html, "html.parser")
            modified_html = self.get_modified_html(initial_html, updated_html)
            modified_htmls.append(modified_html)
            close_button = driver.find_element(By.CSS_SELECTOR, '.game-pick-modal__close-button')
            actions.move_to_element(close_button).click().perform()
            WebDriverWait(driver, 10).until(EC.invisibility_of_element_located((By.CSS_SELECTOR, '.game-pick-modal__info')))
            time.sleep(10)
        return modified_htmls

    def action(self, url):
        site_data = self.site_scrape_chrome_action_button(url)
        play_list, expert_list, odds_list, units_list, name_list = [], [], [], [], []
        for html in site_data:
            for picks in str(html).split('Player Props')[1].split('<div class="pick-card__header">')[1:]:
                soup = BeautifulSoup(picks, 'html.parser')
                play_elements = soup.select('.base-pick__name')
                odds_elements = soup.select('.base-pick__secondary-text')
                units_elements = soup.select('.base-pick__units')
                expert_element = soup.select_one('.pick-card__expert-info > a')
                if expert_element:
                    expert = expert_element['href'].split('/')[-1]
                else:
                    expert = ''
                for play_element, odds_element, units_element in zip(play_elements, odds_elements, units_elements):
                    if re.match(r'[A-Z]\.[A-Za-z]+ [ou][\d]+\.[\d] [A-Za-z]+', play_element.text):
                        play_list.append(play_element.text)
                        expert_list.append(expert)
                        odds_list.append(odds_element.text if odds_element else '')
                        units_list.append(units_element.text.split('u')[0] if units_element else '')
                        name_list.append(str(play_element.text).split(' ')[0])
        data = {
            'Play': play_list,
            'Expert': expert_list,
            'Odds': odds_list,
            'Units': units_list,
            'Name': name_list
        }
        df = pd.DataFrame(data)
        df['Payout'] = df.apply(self.calculate_payout, axis=1).astype(float)
        df['Units'] = df['Units'].astype(float)
        df['Profit'] = df['Payout'] - df['Units']
        df = df.dropna()
        return df[['Play', 'Expert', 'Odds', 'Units', 'Payout', 'Profit', 'Name']] 


        
    def output(self, bets, matchups, all_hitting_box_score_results, all_pitching_box_score_results):
        #determine variables for each bet - player's current team, matchup oppponent, matchup's homecourt advantage
        names, set_teams, opponents, hmcrt_advantages = [], [], [], []
        all_teams = set(all_hitting_box_score_results['Team'].values)
        for i in range(len(bets)):
            bet = bets.loc[i]
            bet['Play'] = bet['Play'].lower()
            name = bet['Name'].split(' ')[0]
            print(name)
            print(bet['Play'])
            if 'ks' in bet['Play'] or 'strikeouts' in bet['Play'] or 'earned runs' in bet['Play'] or 'hits allowed' in bet['Play'] or 'earned runs allowed' in bet['Play']:
                matching_name = all_pitching_box_score_results[all_pitching_box_score_results['Name'] == name]
            elif 'total bases' in bet['Play'] or 'hits' in bet['Play'] or 'home runs' in bet['Play'] or 'hr' in bet['Play'] or 'rbi' in bet['Play'] or 'hits+runs+rbi' in bet['Play'] or 'hits + runs + rbis' in bet['Play'] or 'runs scored' in bet['Play'] or 'bb' in bet['Play'] or 'walks' in bet['Play']:
                matching_name = all_hitting_box_score_results[all_hitting_box_score_results['Name'] == name]
            else:
                bets = bets.drop(i)
                continue 
            if len(set(matching_name['Name'].values)) > 1:
                print('more than 1')
                set_teams.append('')
                names.append(np.NAN)
                opponents.append(np.NAN)
                hmcrt_advantages.append(np.NAN)
            elif len(set(matching_name['Name'].values)) < 1:
                print('less than 1')
                set_teams.append('')
                names.append(np.NAN)
                opponents.append(np.NAN)
                hmcrt_advantages.append(np.NAN)
            else:
                #gets most recent team regardless
                team = matching_name.iloc[0]['Team']
                print(team)
                found_today = False
                for matchup in matchups:
                    matchup_home = matchup.split('vs')[0].strip()
                    matchup_away = matchup.split('vs')[1].strip()            
                    if matchup_home in team:
                        set_teams.append(team)
                        names.append(name)
                        for t in all_teams:
                            if matchup_away in t:
                                opponents.append(t)
                                break
                        hmcrt_advantages.append(1)
                        found_today = True
                        break
                    elif matchup_away in team:
                        set_teams.append(team)
                        names.append(name)
                        for t in all_teams:
                            if matchup_home in t:
                                opponents.append(t)
                                break
                        hmcrt_advantages.append(0)
                        found_today = True
                        break
                if found_today:
                    continue
                else:
                    set_teams.append('')
                    names.append(np.NAN)
                    opponents.append(np.NAN)
                    hmcrt_advantages.append(np.NAN)
        bets['Name'] = names
        bets['Team'] = set_teams
        bets['Opponent'] = opponents
        bets['Hmcrt_adv'] = hmcrt_advantages
        bets = bets.dropna()
        return bets

    def draftkings_output(self, matchups, all_hitting_box_score_results, all_pitching_box_score_results):
        #extract bets predictions for today's games
        date_format = '%m/%d/%Y %H:%M:%S %Z'
        date = datetime.now(timezone('US/Pacific'))
        date = date.astimezone(timezone('US/Pacific'))
        today_date = date.strftime(date_format)
        today = today_date.split('/')[0] + '/' + date.strftime(date_format).split('/')[1]
        today_date = today_date.split(' ')[0]
        month = today.split('/')[0].lstrip('0')
        day = today.split('/')[1].lstrip('0')
        dk_date = month + '/' + day + '/'
        draftkings_bets = self.draftkings(dk_date)
        print('DRAFTKINGS')
        print(draftkings_bets)
        return draftkings_bets

    def underdog_output(self, matchups, all_hitting_box_score_results, all_pitching_box_score_results):
        #extract bets predictions for today's games
        date_format = '%m/%d/%Y %H:%M:%S %Z'
        date = datetime.now(timezone('US/Pacific'))
        date = date.astimezone(timezone('US/Pacific'))
        today_date = date.strftime(date_format)
        today = today_date.split('/')[0] + '/' + date.strftime(date_format).split('/')[1]
        today_date = today_date.split(' ')[0]
        month_name = datetime.strptime(today.split('/')[0].strip(), "%m").strftime("%B").lower()
        month = today.split('/')[0].lstrip('0')
        day = today.split('/')[1].lstrip('0')
        underdog_date = month_name + '-' + day
        underdog_bets = self.underdog(underdog_date)
        print('UNDERDOG')
        print(underdog_bets)
        return underdog_bets

    def action_output(self, matchups, all_hitting_box_score_results, all_pitching_box_score_results):
        #extract bets predictions for today's games
        action_bets = self.action('https://www.actionnetwork.com/mlb/picks')
        print('ACTION')
        print(action_bets)
        return action_bets