import gspread
import feedparser

import pandas as pd
from . import core as tb

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from time import sleep


#########################################
#                                       #
#     GALERTS RSS FEED UTILS            #
#                                       #
#########################################


getGALink = lambda x: x.replace('https://www.google.com/url?rct=j&sa=t&url=','').split('&ct=ga')[0].split('%')[0]
stripHTML = lambda x: re.sub(r'<.*?>', '', x)

def feedToDf(feedURL):
    """Takes a google alerts feed url and returns an article df"""
    keepCols = ['link','title','published','updated','summary']
    feed = feedparser.parse(feedURL)
    entries = feed['entries']
    if entries == []:
        return pd.DataFrame(keepCols)
    feedDf = pd.DataFrame(entries)[keepCols]
    feedDf.loc[:,'url'] = feedDf.link.apply(getGALink)
    feedDf.summary = feedDf.summary.apply(stripHTML)
    feedDf.title = feedDf.title.apply(stripHTML)
    feedDf = feedDf.drop(['link'],axis=1)
    return feedDf

    
def checkViability(pageText):
    """Checks if a scraped page is viable for analysis"""
    if len(pageText) < 200:
        return False
    failStrs = {'Error:',' 404 ',' 429 ', 'Javascript must be'}
    for failStr in failStrs:
        if failStr in pageText:
            return False
    return True


def _translateAlertColumns(df, columns=('title', 'summary'), workers=8):
    """Translate alert fields concurrently using TabulAIrity's configured translator.

    Preserve autoTranslate's output columns, row order, and failure behavior.
    Shared texts are translated once, avoiding duplicate concurrent cache misses.
    """
    df = df.copy(deep=True)
    pending = {}
    for column in columns:
        language_col = f'{column}_language'
        translated_col = f'{column}_translated'
        df[language_col] = df[column].apply(tb.getLanguage)
        df[translated_col] = df[column]
        for text in df.loc[df[language_col] != tb.targetLanguage, column]:
            pending.setdefault(text, None)

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(tb.translateOne, text): text for text in pending}
            for future in as_completed(futures):
                pending[futures[future]] = future.result()  # propagate inference failures

        for column in columns:
            mask = df[f'{column}_language'] != tb.targetLanguage
            df.loc[mask, f'{column}_translated'] = df.loc[mask, column].map(pending)
    return df


def getGAlerts(alertsDf):
    """Takes a google alerts feed google sheet and returns a full feeds df with translated titles"""
    feedDfs = []
    for index, row in alertsDf.iterrows():
        feedURL = row['rss feed']
        feedDf = feedToDf(feedURL)
        feedDf['rss feed'] = feedURL
        feedDfs.append(feedDf)

    mergedFeeds = pd.concat(feedDfs)
    mergedFeeds = mergedFeeds.groupby('url').first().reset_index()
    mergedFeeds = _translateAlertColumns(mergedFeeds, workers=8)
    mergedFeeds = mergedFeeds.merge(alertsDf)
    mergedFeeds['scraped_text'] = mergedFeeds.url.apply(tb.cachePage)
    mergedFeeds['viable_page'] = mergedFeeds.scraped_text.apply(checkViability)
    mergedFeeds['domain'] = mergedFeeds.url.str.split('/').str[2]
    mergedFeeds = mergedFeeds.groupby(['title','domain']).first().reset_index()
    mergedFeeds = mergedFeeds.groupby('url').first().reset_index()

    return mergedFeeds


#########################################
#                                       #
#     FEED DOWNLOADERS                  #
#                                       #
#########################################


def gSheetToDf(spreadsheetName: str,
              worksheetName: str,
              config: dict) -> pd.DataFrame:
    """
    Pulls a table from a specified Google Sheet worksheet into a Pandas DataFrame.

    Args:
        spreadsheetName: The name of the Google Sheet.
        worksheetName: The name of the specific worksheet (tab) within the sheet.

    Returns:
        A Pandas DataFrame containing the data from the worksheet.
    """
    print(f"Attempting to read worksheet '{worksheetName}' from spreadsheet '{spreadsheetName}'...")
    try:
        gc = gspread.service_account(filename=config['g_service_json_path'])
        spreadsheet = gc.open(spreadsheetName)
        worksheet = spreadsheet.worksheet(worksheetName)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        print("Successfully loaded data into DataFrame.")
        return df 
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"Error: Spreadsheet '{spreadsheetName}' not found.")
        print("Please check the spreadsheetName variable and ensure you have access.")
        return pd.DataFrame()
    except gspread.exceptions.WorksheetNotFound:
        print(f"Error: Worksheet '{worksheetName}' not found in '{spreadsheetName}'.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()



def dfToGSheet(df: pd.DataFrame,
               spreadsheetName: str,
               worksheetName: str,
               config: dict):
    """
    Pushes a Pandas DataFrame to a new or existing worksheet in a Google Sheet.
    This will overwrite any existing data in the target worksheet.

    Args:
        df: The Pandas DataFrame to write.
        spreadsheetName: The name of the Google Sheet.
        worksheetName: The name of the worksheet to create/overwrite.
        config: Dictionary containing path to seri
    """
    print(f"Attempting to write DataFrame to worksheet '{worksheetName}' in '{spreadsheetName}'...")
    try:
        gc = gspread.service_account(filename=config['g_service_json_path'])
        spreadsheet = gc.open(spreadsheetName)
        try:
            worksheet = spreadsheet.worksheet(worksheetName)
            print(f"Worksheet '{worksheetName}' already exists. Clearing it before writing new data.")
            worksheet.clear()
        except gspread.exceptions.WorksheetNotFound:
            # If it doesn't exist, create it
            print(f"Worksheet '{worksheetName}' not found. Creating a new one.")
            worksheet = spreadsheet.add_worksheet(title=worksheetName, rows="1000", cols="50")
            
        # Write the DataFrame to the worksheet
        # gspread.utils.dataframe_to_rows returns an iterator, so we convert it to a list
        rows_to_insert = [df.columns.values.tolist()] + df.values.tolist()
        worksheet.update(rows_to_insert, 'A1')
        
        print("DataFrame successfully written to the worksheet.")

    except Exception as e:
        print(f"An error occurred while writing to the sheet: {e}")
