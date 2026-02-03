from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time

# Function to load page with retry
def get_page_with_retry(driver, url, max_attempts=3, wait_time=10):
    for attempt in range(max_attempts):
        try:
            driver.get(url)
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}. Retrying...")
            time.sleep(5)
    print(f"Failed to load {url} after {max_attempts} attempts.")
    return False

# Set up Selenium Chrome driver
options = Options()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=options)
driver.set_page_load_timeout(300)  # 5-minute timeout

# Base URL and years
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
years = list(range(2026, 2025, -1))
all_matches = []

for year in years:
    if not get_page_with_retry(driver, standings_url):
        print(f"Skipping season {year} due to page load failure.")
        continue
    
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CLASS_NAME, "stats_table"))
    )
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    standings_table = soup.select('table.stats_table')[0]
    
    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]
    
    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{previous_season}"
    
    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        print(f"Scraping data for {team_name} ({year})...")
        
        if not get_page_with_retry(driver, team_url):
            print(f"Skipping {team_name} due to page load failure.")
            continue
        
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
        matches = pd.read_html(driver.page_source, match="Scores & Fixtures")[0]
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        if not links:
            print(f"No shooting stats link found for {team_name}. Skipping.")
            continue
        
        if not get_page_with_retry(driver, f"https://fbref.com{links[0]}"):
            print(f"Skipping {team_name} shooting stats due to page load failure.")
            continue
        
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
        shooting = pd.read_html(driver.page_source, match="Shooting")[0]
        
        shooting.columns = shooting.columns.droplevel()
        try:
            team_data = matches.merge(
                shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], 
                on="Date"
            )
        except ValueError:
            print(f"Could not merge data for {team_name}. Skipping.")
            continue
        
        team_data = team_data[team_data["Comp"] == "Premier League"]
        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)
        
        time.sleep(2)

driver.quit()
match_df = pd.concat(all_matches)
match_df.columns = [c.lower() for c in match_df.columns]
match_df.to_csv("updated_matches.csv", index=False)
print("Data scraping complete. Saved to 'updated_matches.csv'.")