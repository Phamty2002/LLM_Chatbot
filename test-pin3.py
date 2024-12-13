import os
import re
import hashlib
import pandas as pd
import logging
import argparse
import time
import random
import json
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
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
pinecone_env = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws')
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

def extract_stock_symbol(filename):
    """
    Extract stock symbol from filename.
    Example: from 'forecast_30_days(Hybrid_AAPL).json' extracts 'AAPL'
    """
    match = re.search(r'Hybrid_([A-Z]+)', filename)
    return match.group(1) if match else None

def generate_unique_id(stock_symbol, date, prediction):
    """
    Generates a unique ID based on stock symbol, date, and prediction.
    """
    unique_string = f"HYBRID_{stock_symbol}_{date}_{prediction}"
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

def upsert_json_data(file_path, index, embedding_model):
    """
    Upsert prediction data from JSON file into Pinecone index.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}", exc_info=True)
        return

    batch_size = 100
    text_summaries = []
    vector_ids = []
    metadatas = []
    
    filename = os.path.basename(file_path)
    stock_symbol = extract_stock_symbol(filename)
    
    if not stock_symbol:
        logger.error(f"Could not extract stock symbol from filename: {filename}")
        return

    for entry in data:
        try:
            # Extract data from JSON
            date = entry.get('Date')
            actual_price = entry.get('Actual_Price')
            predicted_price = entry.get('Predicted_Price')
            mape = entry.get('MAPE')
            rmse = entry.get('RMSE')
            mae = entry.get('MAE')

            if not all([date, actual_price, predicted_price]):
                logger.warning(f"Skipping entry due to missing required fields: {entry}")
                continue

            # Create text summary for embedding
            text_summary = (
                f"For {stock_symbol} on {date}, the hybrid model predicted a price of ${predicted_price:.2f} "
                f"compared to the actual price of ${actual_price:.2f}. "
                f"Model metrics - MAPE: {mape:.2f}%, RMSE: {rmse:.2f}, MAE: {mae:.2f}"
            )

            # Generate unique ID
            vector_id = generate_unique_id(stock_symbol, date, predicted_price)

            # Prepare metadata
            metadata = {
                "type": "hybrid_prediction",
                "stock_symbol": stock_symbol,
                "date": date,
                "actual_price": float(actual_price),
                "predicted_price": float(predicted_price),
                "mape": float(mape) if mape is not None else None,
                "rmse": float(rmse) if rmse is not None else None,
                "mae": float(mae) if mae is not None else None,
                "summary": text_summary
            }

            # Append to lists
            text_summaries.append(text_summary)
            vector_ids.append(vector_id)
            metadatas.append(metadata)

            # Process batch
            if len(text_summaries) >= batch_size:
                try:
                    embeddings = embedding_model.encode(text_summaries, batch_size=batch_size, show_progress_bar=False)
                    
                    batch_vectors = [{
                        "id": vid,
                        "values": emb.tolist(),
                        "metadata": meta
                    } for vid, emb, meta in zip(vector_ids, embeddings, metadatas)]

                    success = upsert_with_retry(index, batch_vectors)
                    if success:
                        logger.info(f"Upserted batch of {len(batch_vectors)} predictions for {stock_symbol}.")
                    
                    # Reset lists
                    text_summaries = []
                    vector_ids = []
                    metadatas = []
                
                except Exception as e:
                    logger.error(f"Error processing batch for {stock_symbol}: {e}", exc_info=True)
                    continue

        except Exception as e:
            logger.error(f"Error processing entry for {stock_symbol}: {e}", exc_info=True)
            continue

    # Process remaining items
    if text_summaries:
        try:
            embeddings = embedding_model.encode(text_summaries, batch_size=batch_size, show_progress_bar=False)
            
            batch_vectors = [{
                "id": vid,
                "values": emb.tolist(),
                "metadata": meta
            } for vid, emb, meta in zip(vector_ids, embeddings, metadatas)]

            success = upsert_with_retry(index, batch_vectors)
            if success:
                logger.info(f"Upserted final batch of {len(batch_vectors)} predictions for {stock_symbol}.")
        
        except Exception as e:
            logger.error(f"Error processing final batch for {stock_symbol}: {e}", exc_info=True)

    logger.info(f"Completed processing {stock_symbol} JSON file.")

def main():
    # List of specific JSON files to process
    json_files = [
        r"C:\Users\Pham Ty\Desktop\Thesis\Result_Model\Result_Hydrid\forecast_30_days(Hybrid_AAPL).json",
        r"C:\Users\Pham Ty\Desktop\Thesis\Result_Model\Result_Hydrid\forecast_30_days(Hybrid_GOOGL).json",
        r"C:\Users\Pham Ty\Desktop\Thesis\Result_Model\Result_Hydrid\forecast_30_days(Hybrid_MSFT).json",
        r"C:\Users\Pham Ty\Desktop\Thesis\Result_Model\Result_Hydrid\forecast_30_days(Hybrid_TSLA).json",
        r"C:\Users\Pham Ty\Desktop\Thesis\Result_Model\Result_Hydrid\forecast_30_days(Hybrid_AMZN).json"
    ]

    logger.info("Starting processing of specific JSON files")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                upsert_json_data, 
                json_file,
                index,
                embedding_model
            ): json_file for json_file in json_files
        }
        
        for future in as_completed(futures):
            json_file = futures[future]
            try:
                future.result()
                logger.info(f"Successfully processed {os.path.basename(json_file)}")
            except Exception as e:
                logger.error(f"Error processing {os.path.basename(json_file)}: {e}")
    
    logger.info("All specified JSON files have been processed.")

if __name__ == "__main__":
    main()