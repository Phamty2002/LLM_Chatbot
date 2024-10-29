import os
import re
import hashlib
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

        # Create text_summary for embedding
        text_summary = f"{headline}\n{content}"

        # Generate unique ID
        vector_id = generate_unique_id(source_url, publication_date, headline)

        # Prepare metadata
        metadata = {
            "type": "news",
            "publication_date": publication_date,  # YYYY-MM-DD
            "source_url": source_url,
            "headline": headline,
            "content": content
        }

        # Append to lists
        text_summaries.append(text_summary)
        vector_ids.append(vector_id)
        metadatas.append(metadata)

        # Process batch
        if len(text_summaries) >= batch_size or (i + 1) == total_items:
            # Generate embeddings for batch
            try:
                embeddings = embedding_model.encode(text_summaries, batch_size=batch_size, show_progress_bar=False)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}", exc_info=True)
                # Skip this batch
                text_summaries = []
                vector_ids = []
                metadatas = []
                continue

            # Prepare vectors
            batch_vectors = [{
                "id": vid,
                "values": emb.tolist(),
                "metadata": meta
            } for vid, emb, meta in zip(vector_ids, embeddings, metadatas)]

            # Upsert batch
            success = upsert_with_retry(index, batch_vectors)
            if success:
                logger.info(f"Upserted batch of {len(batch_vectors)} news articles.")
            else:
                # If upsert failed after retries, attempt to upsert individually
                logger.warning("Attempting to upsert news articles individually due to batch failure.")
                for vector in batch_vectors:
                    if not upsert_with_retry(index, [vector]):
                        logger.error(f"Failed to upsert news article with ID: {vector['id']}")
            # Reset lists
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

        # Skip if essential fields are missing
        if not instruction or not input_text:
            logger.warning(f"Skipping news2 item due to missing fields: {item}")
            continue

        # Extract publication date and source URL from instruction if available
        publication_date = extract_publication_date(instruction)
        source_url = extract_source_url(instruction)

        # Extract headline and content from input_text
        headline, content = extract_headline_content(input_text)

        if not publication_date or not source_url:
            logger.warning(f"Skipping news2 item due to missing publication date or source URL: {item}")
            continue

        # Create text_summary for embedding
        text_summary = f"{headline}\n{content}"

        # Generate unique ID
        vector_id = generate_unique_id(source_url, publication_date, headline)

        # Prepare metadata
        metadata = {
            "type": "news2",
            "publication_date": publication_date,  # YYYY-MM-DD
            "source_url": source_url,
            "headline": headline,
            "content": content
        }

        # Append to lists
        text_summaries.append(text_summary)
        vector_ids.append(vector_id)
        metadatas.append(metadata)

        # Process batch
        if len(text_summaries) >= batch_size or (i + 1) == total_items:
            # Generate embeddings for batch
            try:
                embeddings = embedding_model.encode(text_summaries, batch_size=batch_size, show_progress_bar=False)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}", exc_info=True)
                # Skip this batch
                text_summaries = []
                vector_ids = []
                metadatas = []
                continue

            # Prepare vectors
            batch_vectors = [{
                "id": vid,
                "values": emb.tolist(),
                "metadata": meta
            } for vid, emb, meta in zip(vector_ids, embeddings, metadatas)]

            # Upsert batch
            success = upsert_with_retry(index, batch_vectors)
            if success:
                logger.info(f"Upserted batch of {len(batch_vectors)} news2 articles.")
            else:
                # If upsert failed after retries, attempt to upsert individually
                logger.warning("Attempting to upsert news2 articles individually due to batch failure.")
                for vector in batch_vectors:
                    if not upsert_with_retry(index, [vector]):
                        logger.error(f"Failed to upsert news2 article with ID: {vector['id']}")
            # Reset lists
            text_summaries = []
            vector_ids = []
            metadatas = []

    logger.info("All news2 data has been uploaded to Pinecone.")

def upsert_tweet_data(dataset, index, embedding_model):
    """
    Upsert tweet data from the dataset into the Pinecone index with normalized metadata.
    """
    batch_size = 100  # Adjust as needed
    total_items = len(dataset['train'])

    text_contents = []
    vector_ids = []
    metadatas = []

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

        # Append to lists
        text_contents.append(text_content)
        vector_ids.append(vector_id)
        metadatas.append(metadata)

        # Process batch
        if len(text_contents) >= batch_size or (i + 1) == total_items:
            # Generate embeddings for batch
            try:
                embeddings = embedding_model.encode(text_contents, batch_size=batch_size, show_progress_bar=False)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}", exc_info=True)
                # Skip this batch
                text_contents = []
                vector_ids = []
                metadatas = []
                continue

            # Prepare vectors
            batch_vectors = [{
                "id": vid,
                "values": emb.tolist(),
                "metadata": meta
            } for vid, emb, meta in zip(vector_ids, embeddings, metadatas)]

            # Upsert batch
            success = upsert_with_retry(index, batch_vectors)
            if success:
                logger.info(f"Upserted batch of {len(batch_vectors)} tweets.")
            else:
                # If upsert failed after retries, attempt to upsert individually
                logger.warning("Attempting to upsert tweets individually due to batch failure.")
                for vector in batch_vectors:
                    if not upsert_with_retry(index, [vector]):
                        logger.error(f"Failed to upsert tweet with ID: {vector['id']}")
            # Reset lists
            text_contents = []
            vector_ids = []
            metadatas = []

    logger.info("All tweet data has been uploaded to Pinecone.")

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
        logger.error(f"Error reading {csv_file}: {e}", exc_info=True)
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

    # Prepare text summaries and metadata
    symbol = csv_file.replace('.csv', '').upper()

    text_summaries = []
    vector_ids = []
    metadatas = []

    for _, row in df.iterrows():
        # Create unique ID for each vector
        vector_id = f"{symbol}_{row['Date']}"

        # Create text summary
        text_summary = (
            f"On {row['Date']}, {symbol} opened at ${row['Open']}, reached a high of ${row['High']}, "
            f"a low of ${row['Low']}, and closed at ${row['Close']}. "
            f"The trading volume was {row['Volume']} shares."
        )

        # Prepare metadata
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

def main(upsert_tweets=False, upsert_stocks=False, upsert_news=False, upsert_news2=False):
    logger.info("Starting the data upload process to Pinecone.")

    # Step 1: Download and upsert stock data
    if upsert_stocks:
        logger.info("Starting download of stock data.")
        os.makedirs(folder_path, exist_ok=True)  # Ensure directory exists
        # Utilize concurrency for faster downloads
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upsert data into Pinecone.")
    parser.add_argument('--tweets', action='store_true', help="Upsert tweet data")
    parser.add_argument('--stocks', action='store_true', help="Upsert stock data")
    parser.add_argument('--news', action='store_true', help="Upsert news data")
    parser.add_argument('--news2', action='store_true', help="Upsert news2 data")  # New argument

    args = parser.parse_args()

    if not any([args.tweets, args.stocks, args.news, args.news2]):
        parser.print_help()
        exit(1)

    main(
        upsert_tweets=args.tweets,
        upsert_stocks=args.stocks,
        upsert_news=args.news,
        upsert_news2=args.news2  # Pass the new argument
    )
