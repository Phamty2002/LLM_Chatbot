import os
import logging
import re
import torch
from datetime import datetime
from functools import lru_cache
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')

if not PINECONE_API_KEY:
    raise ValueError("Please set the PINECONE_API_KEY environment variable.")
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("Please set the HUGGINGFACE_API_TOKEN environment variable.")

# Initialize Pinecone
pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment='us-east-1-aws'
)
index_name = 'thesis-database2'  # Ensure this matches your index name
index = pc.Index(index_name)
logger.info(f"Connected to Pinecone index '{index_name}' successfully.")

# Initialize the language model
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Cập nhật mô hình mới
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_API_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Hoặc torch.float32 nếu gặp vấn đề
        device_map="auto",
        use_auth_token=HUGGINGFACE_API_TOKEN
    )
    logger.info(f"Loaded model '{model_name}'.")
except Exception as e:
    logger.error(f"Error loading model '{model_name}': {e}")
    tokenizer = None
    model = None

# Ensure pad_token_id and eos_token_id are defined
if tokenizer and model:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is None:
        # Set to a default value if still None
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('</s>')

# Initialize the SentenceTransformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Giữ nguyên hoặc cập nhật nếu cần
embedding_model = embedding_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
logger.info("Loaded SentenceTransformer 'all-MiniLM-L6-v2' model.")

# Mapping from query price types to metadata fields
PRICE_TYPE_MAPPING = {
    'opening': 'open',
    'closing': 'close',
    'high': 'high',
    'low': 'low',
    'open': 'open',
    'close': 'close',
    'high price': 'high',
    'low price': 'low'
}

def construct_prompt(query, matches):
    context = ""
    for match in matches:
        metadata = match['metadata']
        if metadata.get('type') == 'stock':
            symbol = metadata.get('symbol', 'Unknown Symbol')
            date = metadata.get('date', 'Unknown Date')
            open_price = metadata.get('open', 'Unknown Open')
            close_price = metadata.get('close', 'Unknown Close')
            high_price = metadata.get('high', 'Unknown High')
            low_price = metadata.get('low', 'Unknown Low')
            summary = metadata.get('summary', '')
            context += (
                f"**Stock Data:**\n"
                f"- **Symbol:** {symbol}\n"
                f"- **Date:** {date}\n"
                f"- **Opening Price:** ${open_price}\n"
                f"- **Closing Price:** ${close_price}\n"
                f"- **High Price:** ${high_price}\n"
                f"- **Low Price:** ${low_price}\n"
                f"- **Summary:** {summary}\n\n"
            )
        elif metadata.get('type') == 'news':
            headline = metadata.get('headline', 'No Headline')
            content = metadata.get('content', 'No Content')
            publication_date = metadata.get('publication_date', 'Unknown Date')
            source_url = metadata.get('source_url', 'No Source URL')
            context += (
                f"**News Article:**\n"
                f"- **Headline:** {headline}\n"
                f"- **Content:** {content}\n"
                f"- **Publication Date:** {publication_date}\n"
                f"- **Source URL:** {source_url}\n\n"
            )
        elif metadata.get('type') == 'tweet':
            writer = metadata.get('writer', 'Unknown')
            post_date = metadata.get('post_date', 'Unknown Date')
            text = metadata.get('text', '')
            context += f"**Tweet:**\n- **Author:** {writer}\n- **Date:** {post_date}\n- **Content:** {text}\n\n"
    prompt = f"Use the following context to answer the user's question.\n\n**Context:**\n{context}\n**Question:** {query}\n**Answer:**"
    return prompt

def parse_stock_query(query):
    # Regex patterns to extract price type, stock symbol and date
    patterns = [
        r"What was the (\w+) price of\s*(\w+)\s*on\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})\??",
        r"(\w+) price of\s*(\w+)\s*on\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})",
        r"How much did\s*(\w+)\s*(\w+) at on\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})\??",
        r"Find the (\w+) price for\s*(\w+)\s*on\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})\??",
        r"Show me the (\w+) price of\s*(\w+)\s*on\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})\??",
        r"What was the (\w+) price of\s*(\w+)\s*on\s*(\d{4}-\d{2}-\d{2})\??",
        r"(\w+) price of\s*(\w+)\s*on\s*(\d{4}-\d{2}-\d{2})",
        r"How much did\s*(\w+)\s*(\w+) at on\s*(\d{4}-\d{2}-\d{2})\??",
        r"Find the (\w+) price for\s*(\w+)\s*on\s*(\d{4}-\d{2}-\d{2})\??",
        r"Show me the (\w+) price of\s*(\w+)\s*on\s*(\d{4}-\d{2}-\d{2})\??"
    ]
    for pattern in patterns:
        match = re.match(pattern, query, re.IGNORECASE)
        if match:
            price_type_raw = match.group(1).lower()
            symbol = match.group(2).upper()
            date_str = match.group(3)
            # Map the raw price_type to the metadata field
            price_type = PRICE_TYPE_MAPPING.get(price_type_raw)
            if not price_type:
                logger.error(f"Unknown price type in query: {price_type_raw}")
                return None, None, None
            try:
                # Parse date in different formats
                try:
                    date_obj = datetime.strptime(date_str, '%B %d, %Y')  # e.g., September 19, 2019
                except ValueError:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')  # e.g., 2019-09-19
                date = date_obj.strftime('%Y-%m-%d')
                return price_type, symbol, date
            except ValueError:
                logger.error(f"Invalid date format in query: {date_str}")
                return None, None, None
    return None, None, None

@lru_cache(maxsize=256)
def get_stock_price(price_type, symbol, date):
    # Define metadata filter
    metadata_filter = {
        "type": {"$eq": "stock"},
        "symbol": {"$eq": symbol},
        "date": {"$eq": date}
    }
    # Perform a metadata-filtered search with a generic vector
    generic_query = "stock price query"
    generic_embedding = embedding_model.encode(generic_query).tolist()
    search_response = index.query(
        vector=generic_embedding,
        top_k=1,
        include_values=False,
        include_metadata=True,
        filter=metadata_filter
    )
    # Check if a match was found
    matches = search_response.get('matches', [])
    if matches:
        metadata = matches[0]['metadata']
        price = metadata.get(price_type, 'Unknown')
        symbol = metadata.get('symbol', 'Unknown')
        date = metadata.get('date', 'Unknown')
        logger.debug(f"Retrieved metadata: {metadata}")
        return f"The {price_type} price of {symbol} on {date} was ${price}."
    else:
        logger.info(f"No stock data found for {symbol} on {date}.")
        return f"No stock data found for {symbol} on {date}."

def chatbot_response(query):
    try:
        logger.info(f"Received query: {query}")

        # Detect if the query is requesting tweets by a specific writer
        tweet_query_match = re.match(r"List of tweets by writer named\s*:\s*(.+)", query, re.IGNORECASE)

        if tweet_query_match:
            # Handle tweet queries
            writer_name = tweet_query_match.group(1).strip().lower()
            logger.info(f"Detected specific query for writer: {writer_name}")

            # Define metadata filter
            metadata_filter = {
                "type": {"$eq": "tweet"},
                "writer": {"$eq": writer_name}
            }

            # Generate a generic embedding vector
            generic_query = "retrieve all tweets"
            generic_embedding = embedding_model.encode(generic_query).tolist()

            # Perform a metadata-filtered search with the generic vector
            search_response = index.query(
                vector=generic_embedding,
                top_k=100,
                include_values=False,
                include_metadata=True,
                filter=metadata_filter
            )

            # Check if any matches were found
            matches = search_response.get('matches', [])
            if not matches:
                logger.info(f"No tweets found for author '{writer_name}'.")
                return f"No tweets found for author '{writer_name}'."

            # Collect tweets from the matches
            tweets = []
            for match in matches:
                metadata = match.get('metadata', {})
                text = metadata.get('text', 'No content')
                post_date = metadata.get('post_date', 'Unknown Date')
                tweets.append(f"- {text} (Posted on {post_date})")

            # Optionally limit the number of tweets displayed
            max_display = 10
            tweets = tweets[:max_display]

            # Construct the response
            tweets_list = "\n".join(tweets)
            response = f"Here are some tweets by {writer_name}:\n{tweets_list}"
            logger.info(f"Responding with tweets for writer '{writer_name}'.")
            return response

        else:
            # Check if the query is asking for a specific stock price type on a specific date
            price_type, symbol, date = parse_stock_query(query)
            if price_type and symbol and date:
                logger.info(f"Detected {price_type} price query for symbol: {symbol} on date: {date}")
                response = get_stock_price(price_type, symbol, date)
                logger.info(f"Responding with stock price: {response}")
                return response
            else:
                # Handle general queries using RAG
                logger.info("Detected general query. Performing vector similarity search.")

                # Generate embedding for the user's query
                query_embedding = embedding_model.encode(query).tolist()
                logger.debug(f"Generated query embedding: {query_embedding}")

                # Perform a similarity search in Pinecone
                search_response = index.query(
                    vector=query_embedding,
                    top_k=10,  # Increased for richer context
                    include_values=False,
                    include_metadata=True
                )
                logger.debug(f"Pinecone search response: {search_response}")

                # Check if any matches were found
                matches = search_response.get('matches', [])
                if not matches:
                    logger.info("No matches found in Pinecone.")
                    return "Sorry, I couldn't find any related information."

                # Construct the prompt using retrieved information
                prompt = construct_prompt(query, matches)
                logger.debug(f"Constructed prompt: {prompt}")

                if not (tokenizer and model):
                    logger.error("Tokenizer or model not initialized.")
                    return "Sorry, the system is currently experiencing issues and cannot generate a response right now."

                # Tokenize the prompt
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,  # Tùy thuộc vào mô hình mới, có thể điều chỉnh
                    padding=True
                ).to(model.device)
                logger.debug(f"Tokenized inputs: {inputs}")

                # Generate response from the model
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=150,  # Có thể điều chỉnh tùy theo yêu cầu
                        do_sample=False,  # Disable sampling for deterministic output
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                logger.debug(f"Model outputs: {outputs}")

                # Extract the generated response
                generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                # Clean up the response
                response = re.sub(r'\s+', ' ', response)
                if not response.endswith('.'):
                    response += '.'
                logger.info(f"Generated response: {response}")

                return response if response else "Sorry, I couldn't generate a response."

    except Exception as e:
        logger.error(f"Error processing query '{query}': {e}")
        return "An error occurred while processing your request. Please try again later."

if __name__ == "__main__":
    logger.info("Chatbot is ready. Type 'exit' or 'quit' to exit.")
    while True:
        query = input("You: ")
        if query.strip().lower() in ["exit", "quit"]:
            logger.info("Exiting chatbot.")
            break
        answer = chatbot_response(query)
        print(f"Chatbot: {answer}")
