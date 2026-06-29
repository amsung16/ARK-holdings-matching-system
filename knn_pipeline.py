import yfinance as yf
import pandas as pd
from tqdm import tqdm
import FinanceDataReader as fdr

def get_ark_fundamentals(tickers):
   '''loops through ARK tickers, calls yfinance, 
   returns DataFrame with rev_growth, gross_margin, 
   ps_ratio, market_cap, sector'''
   records = []
   for ticker in tqdm(tickers, desc='Pulling ARK fundamentals'):
      try:
         stock = yf.Ticker(ticker)
         info = stock.info
         record = {
            'Sym': ticker,
            'rev_growth': info.get('revenueGrowth'),
            'gross_margin': info.get('grossMargins'),
            'ps_ratio': info.get('priceToSalesTrailing12Months'),
            'market_cap': info.get('marketCap'),
            'sector': info.get('sector')
         }
         records.append(record)
      except Exception as e:
         continue
   return pd.DataFrame(records)

   

def get_korean_universe():
   '''pulls KOSPI 200 list from FinanceDataReader, returns top 200 by market cap'''
   kospi = fdr.StockListing('KOSPI')
   kospi = kospi.nlargest(200, 'Marcap').reset_index(drop=True)
   return kospi[['Code', 'Name', 'Dept', 'Marcap']]

def get_korean_fundamentals(tickers):
   '''pulls same metrics for Korean stocks via FinanceDataReader / Naver API'''

def map_themes(df, sector_col, mapping_dict):
   '''maps GICS sector (US) or KRX sector (Korean) to shared theme labels
   →used on both DataFrames before KNN'''

def build_feature_matrix(df, features):
   '''selects feature columns, drops rows with nulls, returns clean DataFrame'''

def scale_features(ark_df, kr_df, features):
    '''fits StandardScaler on ARK, transforms both
    returns ark_scaled, kr_scaled as numpy arrays'''

def run_knn(ark_scaled, kr_scaled, k=3):
   '''fits NearestNeighbors on kr_scaled
   queries with ark_scaled
   returns distances, indices'''

def build_output(ark_df, kr_df, distances, indices):
   '''combines ARK stock + 3 Korean matches + distance scores into one DataFrame
   adds ✓/⚠/✗ flag based on distance threshold'''

def export_results(df, path):
   '''writes output DataFrame to Excel'''