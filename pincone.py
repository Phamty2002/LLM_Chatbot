import os
import pandas as pd
import logging
import argparse
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import yfinance as yf
from datetime import datetime
from datasets import load_dataset
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv('PINECONE_API_KEY')
if not api_key:
    raise ValueError("Please set the PINECONE_API_KEY environment variable.")

# Initialize Pinecone
pc = Pinecone(
    api_key=api_key,
    environment='us-east-1-aws'  # Replace with your environment if different
)

index_name = 'thesis-database1'
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
stock_dataset = load_dataset("jyanimaulik/yahoo_finance_stock_market_news")
tweet_dataset = load_dataset("mjw/stock_market_tweets")
logger.info("Datasets loaded successfully.")

# Define folder path for stock data CSVs
folder_path = r'C:\Users\Pham Ty\Desktop\Thesis\Dataset\Dataset-Yahoo'

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
        logger.error(f"Error downloading data for {symbol}: {e}")

def upsert_stock_data(csv_file, index, embedding_model):
    """
    Upsert stock data from CSV file into Pinecone index.
    """
    # Full path to CSV file
    file_path = os.path.join(folder_path, csv_file)

    # Read data from CSV
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading {csv_file}: {e}")
        return

    # Select necessary columns and drop NaN
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
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

    # Prepare vectors and metadata
    vectors = []
    symbol = csv_file.replace('.csv', '').upper()

    for _, row in df.iterrows():
        # Create unique ID for each vector
        vector_id = f"{symbol}_{row['Date']}"

        # Create text summary
        text_summary = (
            f"On {row['Date']}, {symbol} opened at ${row['Open']}, reached a high of ${row['High']}, "
            f"a low of ${row['Low']}, and closed at ${row['Close']}. "
            f"The trading volume was {row['Volume']} shares."
        )

        # Generate embedding
        embedding = embedding_model.encode(text_summary).tolist()

        # Append to vectors list
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "type": "stock",
                "symbol": symbol,
                "date": row['Date'],
                "close": float(row['Close']),
                "summary": text_summary
            }
        })

    # Upsert in batches
    batch_size = 100
    total_vectors = len(vectors)
    for i in range(0, total_vectors, batch_size):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
            logger.info(f"Upserted batch {i//batch_size + 1} with {len(batch)} vectors from {csv_file}.")
        except Exception as e:
            logger.error(f"Error upserting batch {i//batch_size + 1} from {csv_file}: {e}")

    logger.info(f"Data from {csv_file} has been uploaded to Pinecone.")

def upsert_news_data(dataset, index, embedding_model):
    """
    Upsert news data from the dataset into the Pinecone index.
    """
    vectors = []
    batch_size = 100  # You can adjust the batch size
    total_items = len(dataset['train'])

    for i, item in enumerate(dataset['train']):
        # Extract the necessary fields
        instruction = item.get('instruction', '').strip()
        input_text = item.get('input', '').strip()

        # Skip if essential fields are missing
        if not instruction or not input_text:
            logger.warning(f"Skipping news item due to missing fields: {item}")
            continue

        # Create a text summary
        text_summary = f"{instruction}\n{input_text}"

        # Generate embedding
        try:
            embedding = embedding_model.encode(text_summary).tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for news item {i}: {e}")
            continue

        # Create a unique ID for each news article
        vector_id = f"NEWS_{i}"

        # Append to vectors list
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "type": "news",
                "instruction": instruction,
                "input": input_text
            }
        })

        # Upsert in batches
        if (i + 1) % batch_size == 0 or (i + 1) == total_items:
            try:
                index.upsert(vectors=vectors)
                logger.info(f"Upserted batch of {len(vectors)} news articles.")
                vectors = []
            except Exception as e:
                logger.error(f"Error upserting news data: {e}")

    logger.info("All news data has been uploaded to Pinecone.")

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
            logger.error(f"Upsert attempt {attempt + 1} failed: {e}")
            attempt += 1
            sleep_time = 2 ** attempt
            logger.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
    logger.error("All upsert attempts failed.")
    return False

def upsert_tweet_data(dataset, index, embedding_model):
    """
    Upsert tweet data from the dataset into the Pinecone index with normalized metadata.
    """
    vectors = []
    batch_size = 50  # Adjust as needed
    total_items = len(dataset['train'])

    for i, item in enumerate(dataset['train']):
        # Extract necessary fields with default values
        tweet_id = item.get('tweet_id')
        writer = (item.get('writer') or '').strip().lower()  # Normalize to lowercase and strip spaces
        post_date = (item.get('post_date') or '').strip()
        body = (item.get('body') or '').strip()
        comment_num = item.get('comment_num', 0)
        retweet_num = item.get('retweet_num', 0)
        like_num = item.get('like_num', 0)
        ticker_symbol = (item.get('ticker_symbol') or '').strip().upper()

        # Skip if essential fields are missing
        if not tweet_id or not writer or not body:
            logger.warning(f"Skipping tweet due to missing fields: {item}")
            continue

        # Truncate text to a maximum of 200 characters
        max_text_length = 200
        text_content = body[:max_text_length] if len(body) > max_text_length else body

        # Generate embedding
        try:
            embedding = embedding_model.encode(text_content).tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for tweet ID {tweet_id}: {e}")
            continue

        # Create unique ID
        vector_id = f"TWEET_{tweet_id}"

        # Prepare metadata
        metadata = {
            "type": "tweet",
            "tweet_id": tweet_id,
            "writer": writer,  # Normalized writer name
            "post_date": post_date,
            "comment_num": comment_num,
            "retweet_num": retweet_num,
            "like_num": like_num,
            "ticker_symbol": ticker_symbol,
            "text": text_content  # Truncated text
        }

        # Append to vectors list
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        })

        # Upsert in batches
        if len(vectors) >= batch_size or (i + 1) == total_items:
            # Attempt to upsert with retry logic
            success = upsert_with_retry(index, vectors)
            if success:
                logger.info(f"Upserted batch of {len(vectors)} tweets.")
                vectors = []
            else:
                # If upsert failed after retries, attempt to upsert individually
                logger.warning("Attempting to upsert tweets individually due to batch failure.")
                for vector in vectors:
                    if not upsert_with_retry(index, [vector]):
                        logger.error(f"Failed to upsert tweet with ID: {vector['id']}")
                vectors = []

    logger.info("All tweet data has been uploaded to Pinecone.")

def main(upsert_tweets=False, upsert_stocks=False, upsert_news=False):
    logger.info("Starting the data upload process to Pinecone.")

    # Step 1: Download and upsert stock data
    if upsert_stocks:
        logger.info("Starting download of stock data.")
        os.makedirs(folder_path, exist_ok=True)  # Ensure directory exists
        for symbol in stock_symbols:
            download_stock_data(symbol, folder_path, start_date, end_date)
        logger.info("All stock data downloaded successfully.")
        
        logger.info("Starting upsert of stock data into Pinecone.")
        for csv_file in os.listdir(folder_path):
            if csv_file.endswith('.csv'):
                upsert_stock_data(csv_file, index, embedding_model)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upsert data into Pinecone.")
    parser.add_argument('--tweets', action='store_true', help="Upsert tweet data")
    parser.add_argument('--stocks', action='store_true', help="Upsert stock data")
    parser.add_argument('--news', action='store_true', help="Upsert news data")
    
    args = parser.parse_args()
    
    if not any([args.tweets, args.stocks, args.news]):
        parser.print_help()
        exit(1)
    
    main(upsert_tweets=args.tweets, upsert_stocks=args.stocks, upsert_news=args.news)
