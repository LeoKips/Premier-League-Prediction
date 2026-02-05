# Premier-League-Prediction

**Description**: Premier League match-data scraper and match-outcome predictor. The scraper collects match and shooting stats from FBref; the predictor trains an XGBoost model on historical matches and produces probability-based predictions for upcoming fixtures.

Files

* _premierScrapping.py_: Selenium + BeautifulSoup scraper. Produces updated_matches.csv.

* _PremierLeaguePredictor.py_: Data preprocessing, feature engineering, XGBoost training and weekend fixture prediction. Produces a CSV of predicted outcomes.

* _updated_matches.csv_: Matches scrapped by the _premierScrapping.py_ 

**Requirements**

**Python**: 3.8+ recommended.

**Python packages**: selenium, beautifulsoup4, pandas, xgboost, scikit-learn, numpy.

**Browser driver**: Chrome + matching chromedriver for selenium.

**Packages/Requirements**

  * pip install selenium beautifulsoup4 pandas xgboost scikit-learn numpy

**Usage**

* Scrape latest match data (requires Chrome + chromedriver in PATH):
  
 * python premierScrapping.py

 * Output: updated_matches.csv

* Train model & predict fixtures:

  * Ensure updated_matches.csv has been scraped and saved in the project folder

  * python PremierLeaguePredictor.py

  * Output: predicted fixtures

**Configuration & Notes**

* _premierScrapping.py_ uses Selenium.

* The scraper must respect FBref's terms of use and politeness (rate limits). Use responsibly.
