import os
import re
import hashlib
import pandas as pd
import logging
import argparse
import time
import random
import math
from datetime import datetime, timedelta
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
logger.info("Loading SentenceTransformer 'all-MiniLM-L6-v2'...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("SentenceTransformer 'all-MiniLM-L6-v2' loaded successfully.")

# Load datasets
logger.info("Loading datasets...")
stock_dataset = load_dataset("jyanimaulik/yahoo_finance_stock_market_news")  # Existing news dataset
tweet_dataset = load_dataset("mjw/stock_market_tweets")
news2_dataset = load_dataset("jyanimaulik/yahoo_finance_stockmarket_news")  # New news2 dataset
logger.info("Datasets loaded successfully.")

# Define folder paths for stock data and forecast data
base_dir = os.path.dirname(os.path.abspath(__file__))
stock_folder_path = os.path.join(base_dir, 'Dataset', 'Dataset-Yahoo')
forecast_folder_path = os.path.join(base_dir, 'Forecast_Stock', 'Result_CSV')

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
        # Remove trailing punctuation if present
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

def sanitize_metadata(metadata):
    """
    Removes or replaces any NaN values in the metadata dictionary to ensure JSON compatibility.
    """
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, float) and math.isnan(value):
            # Replace NaN with None
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized

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

def upsert_forecast_data(forecast_folder_path, index, embedding_model):
    """
    Upsert forecast data from CSV files into Pinecone index.
    Only pushes processed data: Symbol, Date, Predicted_Price
    """
    vectors = []
    batch_size = 50  # Adjust batch size if needed
    forecast_files = [f for f in os.listdir(forecast_folder_path) if f.startswith('forecast_summary') and f.endswith('.csv')]

    for file in forecast_files:
        file_path = os.path.join(forecast_folder_path, file)
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully read file {file_path}")
        except Exception as e:
            logger.error(f"Error reading {file}: {e}", exc_info=True)
            continue

        # Check for required columns (excluding 'Date')
        required_columns = ['Symbol', 'RMSE', 'MSE', 'MAPE', 'Predicted_Prices', 'Actual_Prices', 'Future_Price_Predictions', 'Train_Prices']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"{file} is missing one or more required columns: {required_columns}")
            continue

        for _, row in df.iterrows():
            symbol = row['Symbol']
            # Extract model name from filename
            model_name = file.replace('forecast_summary', '').replace('.csv', '').strip('_').lower()
            if not model_name:
                model_name = 'baseline'
            vector_id = f"FORECAST_{symbol}_{model_name}"

            # Get future price predictions
            if isinstance(row['Future_Price_Predictions'], str):
                try:
                    future_prices = [float(item.strip()) for item in row['Future_Price_Predictions'].split(',')]
                except:
                    future_prices = []
            elif isinstance(row['Future_Price_Predictions'], list):
                future_prices = row['Future_Price_Predictions']
            else:
                future_prices = []

            # Get the last date from the stock data or use current date
            stock_data_file = os.path.join(stock_folder_path, f"{symbol}.csv")
            if os.path.exists(stock_data_file):
                try:
                    df_stock = pd.read_csv(stock_data_file)
                    if 'Date' in df_stock.columns:
                        df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')
                        last_date = df_stock['Date'].max()
                        if pd.isnull(last_date):
                            last_date = datetime.now()
                    else:
                        logger.warning(f"'Date' column not found in stock data for {symbol}. Using current date.")
                        last_date = datetime.now()
                except Exception as e:
                    logger.error(f"Error reading stock data for {symbol}: {e}", exc_info=True)
                    last_date = datetime.now()
            else:
                logger.warning(f"Stock data file not found for {symbol}. Using current date.")
                last_date = datetime.now()

            future_dates = [last_date + timedelta(days=i) for i in range(1, len(future_prices)+1)]

            for date, pred_price in zip(future_dates, future_prices):
                # Create unique ID for each forecasted day
                daily_vector_id = f"{vector_id}_{date.strftime('%Y-%m-%d')}"

                # Create text summary for embedding
                text_summary = f"{symbol} predicted price on {date.strftime('%Y-%m-%d')} is {pred_price}."

                # Generate embedding
                try:
                    embedding = embedding_model.encode(text_summary).tolist()
                except Exception as e:
                    logger.error(f"Error generating embedding for {daily_vector_id}: {e}", exc_info=True)
                    continue

                # Prepare metadata
                metadata = {
                    "type": "forecast",
                    "symbol": symbol,
                    "date": date.strftime('%Y-%m-%d'),
                    "predicted_price": pred_price
                }

                # Sanitize metadata to remove NaN values
                metadata = sanitize_metadata(metadata)

                # Append to vectors list
                vectors.append({
                    "id": daily_vector_id,
                    "values": embedding,
                    "metadata": metadata
                })

                # Upsert batch
                if len(vectors) >= batch_size:
                    success = upsert_with_retry(index, vectors)
                    if success:
                        logger.info(f"Upserted batch of {len(vectors)} forecast records from {file}.")
                        vectors = []
                    else:
                        logger.error(f"Failed to upsert forecast batch from {file}.")

    # Upsert any remaining vectors
    if vectors:
        success = upsert_with_retry(index, vectors)
        if success:
            logger.info(f"Upserted final batch of {len(vectors)} forecast records.")
        else:
            logger.error("Failed to upsert final forecast batch.")

    logger.info("All forecast data has been uploaded to Pinecone.")

def upsert_stock_data(csv_file, index, embedding_model):
    """
    Upsert stock data from CSV file into Pinecone index.
    Only pushes processed data: Date, Symbol, Close Price
    """
    # Full path to CSV file
    file_path = os.path.join(stock_folder_path, csv_file)

    # Read data from CSV
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully read file {file_path}")
    except Exception as e:
        logger.error(f"Error reading {csv_file}: {e}", exc_info=True)
        return

    # Select necessary columns and drop NaN
    required_columns = ['Date', 'Close']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"{csv_file} is missing one or more required columns: {required_columns}")
        return

    df = df[required_columns].dropna()

    # Process 'Date' column
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    df['Date'] = df['Date'].dt.tz_convert(None)
    invalid_dates = df[df['Date'].isna()]
    if not invalid_dates.empty:
        logger.warning(f"Found {len(invalid_dates)} rows with invalid dates in {csv_file}. They will be dropped.")
        df = df.dropna(subset=['Date'])
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    # Prepare text summaries and metadata
    symbol = csv_file.replace('.csv', '').upper()

    text_summaries = []
    vector_ids = []
    metadatas = []

    for _, row in df.iterrows():
        # Create unique ID for each vector
        vector_id = f"{symbol}_{row['Date']}"

        # Create text summary
        text_summary = f"{symbol} closing price on {row['Date']} is {row['Close']}."

        # Prepare metadata
        metadata = {
            "type": "stock",
            "symbol": symbol,
            "date": row['Date'],
            "close_price": row['Close']
        }

        # Sanitize metadata to remove NaN values
        metadata = sanitize_metadata(metadata)

        # Append to lists
        text_summaries.append(text_summary)
        vector_ids.append(vector_id)
        metadatas.append(metadata)

    # Batch size for embeddings and upserts
    batch_size = 100
    total_vectors = len(vector_ids)
    for i in range(0, total_vectors, batch_size):
        batch_texts = text_summaries[i:i + batch_size]
        batch_ids = vector_ids[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]

        # Generate embeddings for batch
        try:
            embeddings = embedding_model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error generating embeddings for batch from {csv_file}: {e}", exc_info=True)
            continue

        # Prepare vectors
        batch_vectors = [{
            "id": vid,
            "values": emb.tolist(),
            "metadata": meta
        } for vid, emb, meta in zip(batch_ids, embeddings, batch_metadatas)]

        # Upsert batch
        try:
            index.upsert(vectors=batch_vectors)
            logger.info(f"Upserted batch {i//batch_size + 1} with {len(batch_vectors)} vectors from {csv_file}.")
        except Exception as e:
            logger.error(f"Error upserting batch {i//batch_size + 1} from {csv_file}: {e}", exc_info=True)

    logger.info(f"Data from {csv_file} has been uploaded to Pinecone.")

def main(upsert_tweets=False, upsert_stocks=False, upsert_news=False, upsert_news2=False, upsert_forecast=False):
    logger.info("Starting the data upload process to Pinecone.")

    # Step 1: Download and upsert stock data
    if upsert_stocks:
        logger.info("Starting download of stock data.")
        os.makedirs(stock_folder_path, exist_ok=True)  # Ensure directory exists
        # Utilize concurrency for faster downloads
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(lambda symbol: download_stock_data(symbol, stock_folder_path, start_date, end_date), stock_symbols)
        logger.info("All stock data downloaded successfully.")

        logger.info("Starting upsert of stock data into Pinecone.")
        csv_files = [f for f in os.listdir(stock_folder_path) if f.endswith('.csv')]
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(upsert_stock_data, csv_file, index, embedding_model): csv_file for csv_file in csv_files}
            for future in as_completed(futures):
                csv_file = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing {csv_file}: {e}", exc_info=True)
        logger.info("All stock data uploaded to Pinecone successfully.")

    # Step 2: Upsert news data into Pinecone
    if upsert_news:
        logger.info("Starting upsert of news data into Pinecone.")
        upsert_news_data(stock_dataset, index, embedding_model)
        logger.info("All news data uploaded to Pinecone successfully.")

    # Step 3: Upsert news2 data into Pinecone
    if upsert_news2:
        logger.info("Starting upsert of news2 data into Pinecone.")
        upsert_news2_data(news2_dataset, index, embedding_model)
        logger.info("All news2 data uploaded to Pinecone successfully.")

    # Step 4: Upsert tweet data into Pinecone
    if upsert_tweets:
        logger.info("Starting upsert of tweet data into Pinecone.")
        upsert_tweet_data(tweet_dataset, index, embedding_model)
        logger.info("All tweet data uploaded to Pinecone successfully.")

    # Step 5: Upsert forecast data into Pinecone
    if upsert_forecast:
        logger.info("Starting upsert of forecast data into Pinecone.")
        upsert_forecast_data(forecast_folder_path, index, embedding_model)
        logger.info("All forecast data uploaded to Pinecone successfully.")

    logger.info("Data upload process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upsert data into Pinecone.")
    parser.add_argument('--tweets', action='store_true', help="Upsert tweet data")
    parser.add_argument('--stocks', action='store_true', help="Upsert stock data")
    parser.add_argument('--news', action='store_true', help="Upsert news data")
    parser.add_argument('--news2', action='store_true', help="Upsert news2 data")  # Existing argument
    parser.add_argument('--forecast', action='store_true', help="Upsert forecast data")  # New argument

    args = parser.parse_args()

    if not any([args.tweets, args.stocks, args.news, args.news2, args.forecast]):
        parser.print_help()
        exit(1)

    main(
        upsert_tweets=args.tweets,
        upsert_stocks=args.stocks,
        upsert_news=args.news,
        upsert_news2=args.news2,
        upsert_forecast=args.forecast  # Pass the new argument
    )
