import os
import re
import hashlib
import json
import pandas as pd
import logging
import argparse
import time
import random
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import yfinance as yf
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv('PINECONE_API_KEY')
if not api_key:
    raise ValueError("Please set the PINECONE_API_KEY environment variable.")

# Initialize Pinecone
pinecone_env = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws')  # Make environment configurable
pc = Pinecone(
    api_key=api_key,
    environment=pinecone_env
)

index_name = 'thesis-database2'
dimension = 384  # Embedding dimension for 'all-MiniLM-L6-v2'

# Create the index with correct dimensions if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    logger.info(f"Index '{index_name}' created with dimension {dimension}.")
else:
    logger.info(f"Index '{index_name}' already exists.")

index = pc.Index(index_name)

# Initialize SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("SentenceTransformer 'all-MiniLM-L6-v2' loaded successfully.")

# Load datasets
stock_dataset = load_dataset("jyanimaulik/yahoo_finance_stock_market_news")  # Existing news dataset
tweet_dataset = load_dataset("mjw/stock_market_tweets")
news2_dataset = load_dataset("jyanimaulik/yahoo_finance_stockmarket_news")  # New news2 dataset
logger.info("Datasets loaded successfully.")

# Define folder path for stock data CSVs
folder_path = r'C:\Users\Pham Ty\Desktop\Thesis\Dataset\Dataset-Yahoo'

# Define folder path for forecast data JSONs
forecast_folder_path = r'C:\Users\Pham Ty\Desktop\Thesis\Result_Model\Result_Hydrid'

# List of stock symbols
stock_symbols = [
    'NVDA', 'INTC', 'PLTR', 'TSLA', 'AAPL', 'BBD', 'T', 'SOFI',
    'WBD', 'SNAP', 'NIO', 'BTG', 'F', 'AAL', 'NOK', 'BAC',
    'CCL', 'ORCL', 'AMD', 'PFE', 'KGC', 'MARA', 'SLB', 'NU',
    'MPW', 'MU', 'LCID', 'NCLH', 'RIG', 'AMZN', 'ABEV', 'U',
    'LUMN', 'AGNC', 'VZ', 'WBA', 'WFC', 'RIVN', 'UPST', 'CFE',
    'CSCO', 'VALE', 'AVGO', 'PBR', 'GOOGL', 'SMMT', 'GOLD',
    'NGC', 'BCS', 'UAA'
]

# Define date range
start_date = '2019-09-19'
end_date = '2024-09-19'

def download_stock_data(symbol, folder_path, start_date, end_date):
    """
    Download stock data and save as CSV.
    """
    try:
        logger.info(f"Starting download for {symbol}")
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            logger.warning(f"No data found for {symbol} between {start_date} and {end_date}.")
            return

        # Reset index to have 'Date' as a column
        df.reset_index(inplace=True)

        # Select necessary columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[required_columns]

        # Handle 'Adj Close' if needed
        if 'Adj Close' in df.columns:
            df.rename(columns={'Adj Close': 'Adj Close'}, inplace=True)
        else:
            df['Adj Close'] = df['Close']

        # Reorder columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

        # Save to CSV
        file_path = os.path.join(folder_path, f"{symbol}.csv")
        df.to_csv(file_path, index=False)
        logger.info(f"Data for {symbol} saved to {file_path}")
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}", exc_info=True)

def extract_publication_date(instruction):
    """
    Extracts the publication date from the instruction field.
    Expected format: 'published on DD-MM-YYYY'
    """
    match = re.search(r'published on (\d{2}-\d{2}-\d{4})', instruction)
    if match:
        try:
            # Convert to ISO format for consistency
            date_obj = datetime.strptime(match.group(1), '%d-%m-%Y')
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            logger.warning(f"Invalid date format found: {match.group(1)}")
    return None

def extract_source_url(instruction):
    """
    Extracts the source URL from the instruction field.
    """
    match = re.search(r'source URL is:\s*(https?://\S+)', instruction)
    if match:
        return match.group(1).rstrip('.')
    return None

def extract_headline_content(input_text):
    """
    Extracts Headline and Content from the input field.
    Example Input:
    Headline: New AI ETF Combines AI Stocks with a Jaw-dropping Yield. Content: The new REX AI Equity Premium Income ETF...
    """
    headline_match = re.search(r'Headline:\s*(.*?)\. Content:', input_text)
    content_match = re.search(r'Content:\s*(.*)', input_text, re.DOTALL)
    
    headline = headline_match.group(1).strip() if headline_match else ""
    content = content_match.group(1).strip() if content_match else ""
    
    return headline, content

def generate_unique_id(source_url, publication_date, headline):
    """
    Generates a unique ID based on source_url, publication_date, and headline.
    """
    unique_string = f"{source_url}_{publication_date}_{headline}"
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

def upsert_with_retry(index, vectors, max_retries=3):
    """
    Upsert vectors with retry logic.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            index.upsert(vectors=vectors)
            return True
        except Exception as e:
            logger.error(f"Upsert attempt {attempt + 1} failed: {e}", exc_info=True)
            attempt += 1
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            logger.info(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
    logger.error("All upsert attempts failed.")
    return False

def upsert_news_data(dataset, index, embedding_model):
    """
    Upsert news data from the dataset into the Pinecone index with optimized metadata.
    """
    batch_size = 100
    total_items = len(dataset['train'])

    text_summaries = []
    vector_ids = []
    metadatas = []

    for i, item in enumerate(dataset['train']):
        instruction = item.get('instruction', '').strip()
        input_text = item.get('input', '').strip()

        # Skip if essential fields are missing
        if not instruction or not input_text:
            logger.warning(f"Skipping news item due to missing fields: {item}")
            continue

        # Extract information from instruction and input_text
        publication_date = extract_publication_date(instruction)
        source_url = extract_source_url(instruction)
        headline, content = extract_headline_content(input_text)

        if not publication_date or not source_url:
            logger.warning(f"Skipping news item due to missing publication date or source URL: {item}")
            continue

        text_summary = f"{headline}\n{content}"
        vector_id = generate_unique_id(source_url, publication_date, headline)

        metadata = {
            "type": "news",
            "publication_date": publication_date,
            "source_url": source_url,
            "headline": headline,
            "content": content
        }

        text_summaries.append(text_summary)
        vector_ids.append(vector_id)
        metadatas.append(metadata)

        if len(text_summaries) >= batch_size or (i + 1) == total_items:
            try:
                embeddings = embedding_model.encode(text_summaries, batch_size=batch_size, show_progress_bar=False)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}", exc_info=True)
                text_summaries = []
                vector_ids = []
                metadatas = []
                continue

            batch_vectors = [{
                "id": vid,
                "values": emb.tolist(),
                "metadata": meta
            } for vid, emb, meta in zip(vector_ids, embeddings, metadatas)]

            success = upsert_with_retry(index, batch_vectors)
            if success:
                logger.info(f"Upserted batch of {len(batch_vectors)} news articles.")
            else:
                logger.warning("Attempting to upsert news articles individually due to batch failure.")
                for vector in batch_vectors:
                    if not upsert_with_retry(index, [vector]):
                        logger.error(f"Failed to upsert news article with ID: {vector['id']}")

            text_summaries = []
            vector_ids = []
            metadatas = []

    logger.info("All news data has been uploaded to Pinecone.")

def upsert_news2_data(dataset, index, embedding_model):
    """
    Upsert news2 data from the dataset into the Pinecone index with optimized metadata.
    """
    batch_size = 100
    total_items = len(dataset['train'])

    text_summaries = []
    vector_ids = []
    metadatas = []

    for i, item in enumerate(dataset['train']):
        instruction = item.get('instruction', '').strip()
        input_text = item.get('input', '').strip()

        if not instruction or not input_text:
            logger.warning(f"Skipping news2 item due to missing fields: {item}")
            continue

        publication_date = extract_publication_date(instruction)
        source_url = extract_source_url(instruction)
        headline, content = extract_headline_content(input_text)

        if not publication_date or not source_url:
            logger.warning(f"Skipping news2 item due to missing publication date or source URL: {item}")
            continue

        text_summary = f"{headline}\n{content}"
        vector_id = generate_unique_id(source_url, publication_date, headline)

        metadata = {
            "type": "news2",
            "publication_date": publication_date,
            "source_url": source_url,
            "headline": headline,
            "content": content
        }

        text_summaries.append(text_summary)
        vector_ids.append(vector_id)
        metadatas.append(metadata)

        if len(text_summaries) >= batch_size or (i + 1) == total_items:
            try:
                embeddings = embedding_model.encode(text_summaries, batch_size=batch_size, show_progress_bar=False)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}", exc_info=True)
                text_summaries = []
                vector_ids = []
                metadatas = []
                continue

            batch_vectors = [{
                "id": vid,
                "values": emb.tolist(),
                "metadata": meta
            } for vid, emb, meta in zip(vector_ids, embeddings, metadatas)]

            success = upsert_with_retry(index, batch_vectors)
            if success:
                logger.info(f"Upserted batch of {len(batch_vectors)} news2 articles.")
            else:
                logger.warning("Attempting to upsert news2 articles individually due to batch failure.")
                for vector in batch_vectors:
                    if not upsert_with_retry(index, [vector]):
                        logger.error(f"Failed to upsert news2 article with ID: {vector['id']}")

            text_summaries = []
            vector_ids = []
            metadatas = []

    logger.info("All news2 data has been uploaded to Pinecone.")

def upsert_tweet_data(dataset, index, embedding_model):
    """
    Upsert tweet data from the dataset into the Pinecone index with normalized metadata.
    """
    batch_size = 100
    total_items = len(dataset['train'])

    text_contents = []
    vector_ids = []
    metadatas = []

    for i, item in enumerate(dataset['train']):
        tweet_id = item.get('tweet_id')
        writer = (item.get('writer') or '').strip().lower()
        post_date = (item.get('post_date') or '').strip()
        body = (item.get('body') or '').strip()
        comment_num = item.get('comment_num', 0)
        retweet_num = item.get('retweet_num', 0)
        like_num = item.get('like_num', 0)
        ticker_symbol = (item.get('ticker_symbol') or '').strip().upper()

        if not tweet_id or not writer or not body:
            logger.warning(f"Skipping tweet due to missing fields: {item}")
            continue

        max_text_length = 200
        text_content = body[:max_text_length] if len(body) > max_text_length else body

        vector_id = f"TWEET_{tweet_id}"
        metadata = {
            "type": "tweet",
            "tweet_id": tweet_id,
            "writer": writer,
            "post_date": post_date,
            "comment_num": comment_num,
            "retweet_num": retweet_num,
            "like_num": like_num,
            "ticker_symbol": ticker_symbol,
            "text": text_content
        }

        text_contents.append(text_content)
        vector_ids.append(vector_id)
        metadatas.append(metadata)

        if len(text_contents) >= batch_size or (i + 1) == total_items:
            try:
                embeddings = embedding_model.encode(text_contents, batch_size=batch_size, show_progress_bar=False)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}", exc_info=True)
                text_contents = []
                vector_ids = []
                metadatas = []
                continue

            batch_vectors = [{
                "id": vid,
                "values": emb.tolist(),
                "metadata": meta
            } for vid, emb, meta in zip(vector_ids, embeddings, metadatas)]

            success = upsert_with_retry(index, batch_vectors)
            if success:
                logger.info(f"Upserted batch of {len(batch_vectors)} tweets.")
            else:
                logger.warning("Attempting to upsert tweets individually due to batch failure.")
                for vector in batch_vectors:
                    if not upsert_with_retry(index, [vector]):
                        logger.error(f"Failed to upsert tweet with ID: {vector['id']}")

            text_contents = []
            vector_ids = []
            metadatas = []

    logger.info("All tweet data has been uploaded to Pinecone.")

def upsert_stock_data(csv_file, index, embedding_model):
    """
    Upsert stock data from CSV file into Pinecone index.
    """
    file_path = os.path.join(folder_path, csv_file)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading {csv_file}: {e}", exc_info=True)
        return

    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"{csv_file} is missing one or more required columns: {required_columns}")
        return

    df = df[required_columns].dropna()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    df['Date'] = df['Date'].dt.tz_convert(None)
    invalid_dates = df[df['Date'].isna()]
    if not invalid_dates.empty:
        logger.warning(f"Found {len(invalid_dates)} rows with invalid dates in {csv_file}. They will be dropped.")
        df = df.dropna(subset=['Date'])
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    symbol = csv_file.replace('.csv', '').upper()

    text_summaries = []
    vector_ids = []
    metadatas = []

    for _, row in df.iterrows():
        vector_id = f"{symbol}_{row['Date']}"
        text_summary = (
            f"On {row['Date']}, {symbol} opened at ${row['Open']}, reached a high of ${row['High']}, "
            f"a low of ${row['Low']}, and closed at ${row['Close']}. "
            f"The trading volume was {row['Volume']} shares."
        )

        metadata = {
            "type": "stock",
            "symbol": symbol,
            "date": row['Date'],
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
            "close": float(row['Close']),
            "summary": text_summary
        }

        text_summaries.append(text_summary)
        vector_ids.append(vector_id)
        metadatas.append(metadata)

    batch_size = 100
    total_vectors = len(vector_ids)
    for i in range(0, total_vectors, batch_size):
        batch_texts = text_summaries[i:i + batch_size]
        batch_ids = vector_ids[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]

        try:
            embeddings = embedding_model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error generating embeddings for batch from {csv_file}: {e}", exc_info=True)
            continue

        batch_vectors = [{
            "id": vid,
            "values": emb.tolist(),
            "metadata": meta
        } for vid, emb, meta in zip(batch_ids, embeddings, batch_metadatas)]

        try:
            index.upsert(vectors=batch_vectors)
            logger.info(f"Upserted batch {i//batch_size + 1} with {len(batch_vectors)} vectors from {csv_file}.")
        except Exception as e:
            logger.error(f"Error upserting batch {i//batch_size + 1} from {csv_file}: {e}", exc_info=True)

    logger.info(f"Data from {csv_file} has been uploaded to Pinecone.")

def upsert_forecast_data(folder_path, index, embedding_model):
    """
    Upsert forecast data from JSON files into the Pinecone index.
    """
    batch_size = 100
    text_summaries = []
    vector_ids = []
    metadatas = []
    
    try:
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        if not json_files:
            logger.warning(f"No JSON files found in {folder_path}.")
            return
    except Exception as e:
        logger.error(f"Error accessing forecast folder {folder_path}: {e}", exc_info=True)
        return

    total_files = len(json_files)
    logger.info(f"Found {total_files} JSON files in {folder_path}.")

    for idx, json_file in enumerate(json_files, start=1):
        file_path = os.path.join(folder_path, json_file)
        logger.info(f"Processing file {idx}/{total_files}: {json_file}")
        
        # Chỉnh regex linh hoạt hơn, nếu cần:
        # match = re.search(r'\((Hybrid_)?([A-Z]+)\)', json_file)
        match = re.search(r'\((Hybrid_)?([A-Za-z0-9]+)\)', json_file)
        if match:
            stock_symbol = match.group(2).upper()
        else:
            logger.warning(f"Could not extract stock symbol from file name: {json_file}. Skipping file.")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading {json_file}: {e}", exc_info=True)
            continue

        for item in data:
            instruction = item.get('instruction', '').strip()
            output = item.get('output', '').strip()

            if not instruction or not output:
                logger.warning(f"Skipping forecast item due to missing fields: {item}")
                continue

            # Extract date
            date_match = re.search(r'vào ngày (\d{1,2}/\d{1,2}/\d{4})', instruction)
            if date_match:
                date_str = date_match.group(1)
                try:
                    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                    publication_date = date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    logger.warning(f"Invalid date format in instruction: {instruction}")
                    publication_date = None
            else:
                logger.warning(f"Could not extract date from instruction: {instruction}")
                publication_date = None

            # Extract price
            price_match = re.search(r'là ([\d\.]+)', output)
            if price_match:
                predicted_price = float(price_match.group(1))
            else:
                logger.warning(f"Could not extract price from output: {output}")
                predicted_price = None

            if not publication_date or predicted_price is None:
                logger.warning(f"Skipping forecast item due to missing date or price: {item}")
                continue

            text_summary = f"On {publication_date}, the predicted price of {stock_symbol} is {predicted_price} USD."
            vector_id = f"forecast_{stock_symbol}_{publication_date}"

            metadata = {
                "type": "forecast",
                "stock_symbol": stock_symbol,
                "date": publication_date,
                "predicted_price": predicted_price,
                "source_file": json_file
            }

            text_summaries.append(text_summary)
            vector_ids.append(vector_id)
            metadatas.append(metadata)

            if len(text_summaries) >= batch_size:
                try:
                    embeddings = embedding_model.encode(text_summaries, batch_size=batch_size, show_progress_bar=False)
                except Exception as e:
                    logger.error(f"Error generating embeddings for forecast batch: {e}", exc_info=True)
                    text_summaries = []
                    vector_ids = []
                    metadatas = []
                    continue

                batch_vectors = [{
                    "id": vid,
                    "values": emb.tolist(),
                    "metadata": meta
                } for vid, emb, meta in zip(vector_ids, embeddings, metadatas)]

                success = upsert_with_retry(index, batch_vectors)
                if success:
                    logger.info(f"Upserted batch of {len(batch_vectors)} forecast entries from {json_file}.")
                else:
                    logger.warning("Attempting to upsert forecast entries individually due to batch failure.")
                    for vector in batch_vectors:
                        if not upsert_with_retry(index, [vector]):
                            logger.error(f"Failed to upsert forecast entry with ID: {vector['id']}")

                text_summaries = []
                vector_ids = []
                metadatas = []

    if text_summaries:
        try:
            embeddings = embedding_model.encode(text_summaries, batch_size=batch_size, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error generating embeddings for final forecast batch: {e}", exc_info=True)
            return

        batch_vectors = [{
            "id": vid,
            "values": emb.tolist(),
            "metadata": meta
        } for vid, emb, meta in zip(vector_ids, embeddings, metadatas)]

        success = upsert_with_retry(index, batch_vectors)
        if success:
            logger.info(f"Upserted final batch of {len(batch_vectors)} forecast entries.")
        else:
            logger.warning("Attempting to upsert final forecast entries individually due to batch failure.")
            for vector in batch_vectors:
                if not upsert_with_retry(index, [vector]):
                    logger.error(f"Failed to upsert forecast entry with ID: {vector['id']}")

    logger.info("All forecast data has been uploaded to Pinecone.")

def main(upsert_tweets=False, upsert_stocks=False, upsert_news=False, upsert_news2=False, upsert_forecast=False):
    logger.info("Starting the data upload process to Pinecone.")

    # Step 1: Download and upsert stock data
    if upsert_stocks:
        logger.info("Starting download of stock data.")
        os.makedirs(folder_path, exist_ok=True)  # Ensure directory exists
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(lambda symbol: download_stock_data(symbol, folder_path, start_date, end_date), stock_symbols)
        logger.info("All stock data downloaded successfully.")
        
        logger.info("Starting upsert of stock data into Pinecone.")
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(upsert_stock_data, csv_file, index, embedding_model): csv_file for csv_file in csv_files}
            for future in as_completed(futures):
                csv_file = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing {csv_file}: {e}")
        logger.info("All stock data uploaded to Pinecone successfully.")

    # Step 2: Upsert news data into Pinecone
    if upsert_news:
        logger.info("Starting upsert of news data into Pinecone.")
        upsert_news_data(stock_dataset, index, embedding_model)
        logger.info("All news data uploaded to Pinecone successfully.")

    # Step 3: Upsert tweet data into Pinecone
    if upsert_tweets:
        logger.info("Starting upsert of tweet data into Pinecone.")
        upsert_tweet_data(tweet_dataset, index, embedding_model)
        logger.info("All tweet data uploaded to Pinecone successfully.")

    # Step 4: Upsert news2 data into Pinecone
    if upsert_news2:
        logger.info("Starting upsert of news2 data into Pinecone.")
        upsert_news2_data(news2_dataset, index, embedding_model)
        logger.info("All news2 data uploaded to Pinecone successfully.")

    # Step 5: Upsert forecast data into Pinecone
    if upsert_forecast:
        logger.info("Starting upsert of forecast data into Pinecone.")
        upsert_forecast_data(forecast_folder_path, index, embedding_model)
        logger.info("All forecast data uploaded to Pinecone successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upsert data into Pinecone.")
    parser.add_argument('--tweets', action='store_true', help="Upsert tweet data")
    parser.add_argument('--stocks', action='store_true', help="Upsert stock data")
    parser.add_argument('--news', action='store_true', help="Upsert news data")
    parser.add_argument('--news2', action='store_true', help="Upsert news2 data")
    parser.add_argument('--forecast', action='store_true', help="Upsert forecast data")

    args = parser.parse_args()

    if not any([args.tweets, args.stocks, args.news, args.news2, args.forecast]):
        parser.print_help()
        exit(1)

    main(
        upsert_tweets=args.tweets,
        upsert_stocks=args.stocks,
        upsert_news=args.news,
        upsert_news2=args.news2,
        upsert_forecast=args.forecast
    )
