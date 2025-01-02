import os
import logging
import re
import torch
from datetime import datetime
from functools import lru_cache
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    logger.info("Pinecone client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Pinecone client: {e}")
    raise

index_name = 'thesis-database2'

# Check if the index exists; if not, create it
try:
    existing_indexes = [index.name for index in pc.list_indexes().indexes]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,  # Ensure this matches your embedding dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        logger.info(f"Index '{index_name}' created successfully.")
    else:
        logger.info(f"Index '{index_name}' already exists.")
except Exception as e:
    logger.error(f"Error checking or creating index '{index_name}': {e}")
    raise

# Connect to the index
try:
    index = pc.Index(index_name)
    logger.info(f"Connected to Pinecone index '{index_name}' successfully.")
except Exception as e:
    logger.error(f"Error connecting to Pinecone index '{index_name}': {e}")
    raise

# Initialize the language model
model_name = "meta-llama/Llama-3.2-1B-Instruct"
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=HUGGINGFACE_API_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
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
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('</s>')

# Initialize the SentenceTransformer model for embeddings
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_model = embedding_model.to(device)
    logger.info("Loaded SentenceTransformer 'all-MiniLM-L6-v2' model.")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {e}")
    raise

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
    context_sections = {
        'stock': [],
        'news': [],
        'tweet': [],
        'forecast': []
    }
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
            stock_info = (
                f"Stock Symbol: {symbol}\n"
                f"Date: {date}\n"
                f"Opening Price: ${open_price}\n"
                f"Closing Price: ${close_price}\n"
                f"High Price: ${high_price}\n"
                f"Low Price: ${low_price}\n"
                f"Summary: {summary}"
            )
            context_sections['stock'].append(stock_info)
        elif metadata.get('type') in ['news', 'news2']:
            headline = metadata.get('headline', 'No Headline')
            content = metadata.get('content', 'No Content')
            publication_date = metadata.get('publication_date', 'Unknown Date')
            source_url = metadata.get('source_url', 'No Source URL')
            news_info = (
                f"Headline: {headline}\n"
                f"Content: {content}\n"
                f"Publication Date: {publication_date}\n"
                f"Source URL: {source_url}"
            )
            context_sections['news'].append(news_info)
        elif metadata.get('type') == 'tweet':
            writer = metadata.get('writer', 'Unknown')
            post_date = metadata.get('post_date', 'Unknown Date')
            text = metadata.get('text', '')
            tweet_info = (
                f"Author: {writer}\n"
                f"Date: {post_date}\n"
                f"Content: {text}"
            )
            context_sections['tweet'].append(tweet_info)
        elif metadata.get('type') == 'forecast':
            symbol = metadata.get('stock_symbol', 'Unknown Symbol')
            date = metadata.get('date', 'Unknown Date')
            predicted_price = metadata.get('predicted_price', 'Unknown')
            forecast_info = (
                f"Stock Symbol: {symbol}\n"
                f"Date: {date}\n"
                f"Predicted Price: ${predicted_price}"
            )
            context_sections['forecast'].append(forecast_info)

    # Build the context string
    context = ""
    for section, infos in context_sections.items():
        if infos:
            context += f"### {section.capitalize()} Information:\n"
            for info in infos:
                context += f"{info}\n\n"

    examples = """
**Examples:**
**Q:** What was the closing price of AAPL on September 19, 2023?
**A:** The closing price of AAPL on September 19, 2023, was $150.25.

**Q:** Summarize the latest news about Tesla (TSLA).
**A:** (Based only on the provided news context) Tesla recently unveiled a new electric truck. Highlights include extended battery life and improved autonomous features.

If the context does not contain the requested information, answer: "I’m sorry, I could not find the information."
"""

    prompt = f"""
You are a highly knowledgeable AI assistant specializing in financial markets, stocks, and related news. Your goal is to provide accurate, concise, and helpful answers based only on the provided context. If the information is not available in the context, respond with "I’m sorry, I could not find the information."

Do not use any information not found in the context.

{examples}

### Context:
{context}

### Question:
{query}

### Answer:
"""
    return prompt.strip()

def parse_stock_query(query):
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
            price_type = PRICE_TYPE_MAPPING.get(price_type_raw)
            if not price_type:
                logger.error(f"Unknown price type in query: {price_type_raw}")
                return None, None, None
            try:
                try:
                    date_obj = datetime.strptime(date_str, '%B %d, %Y')
                except ValueError:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                date = date_obj.strftime('%d/%m/%Y')
                return price_type, symbol, date
            except ValueError:
                logger.error(f"Invalid date format in query: {date_str}")
                return None, None, None
    return None, None, None

def parse_forecast_query(query):
    # Nhận biết câu truy vấn dự báo giá cổ phiếu
    patterns = [
        r"(\w+)\s+Stock Price Prediction on\s+(\d{1,2}\/\d{1,2}\/\d{4})\??",
        r"Predict the price of\s+(\w+)\s+on\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})\??",
        r"Forecast for\s+(\w+)\s+on\s+(\d{1,2}\/\d{1,2}\/\d{4})\??"
    ]

    for pattern in patterns:
        match = re.match(pattern, query, re.IGNORECASE)
        if match:
            symbol = match.group(1).upper()
            date_str = match.group(2)
            try:
                # Thử parse MM/DD/YYYY
                try:
                    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                except ValueError:
                    # Nếu không parse được MM/DD/YYYY, thử 'Month DD, YYYY'
                    try:
                        date_obj = datetime.strptime(date_str, '%B %d, %Y')
                    except ValueError:
                        return None, None
                date_iso = date_obj.strftime('%Y-%m-%d')
                return symbol, date_iso
            except ValueError:
                return None, None

    return None, None

@lru_cache(maxsize=256)
def get_stock_price(price_type, symbol, date):
    metadata_filter = {
        "type": {"$eq": "stock"},
        "symbol": {"$eq": symbol},
        "date": {"$eq": date}
    }
    generic_query = "stock price query"
    generic_embedding = embedding_model.encode(generic_query).tolist()
    try:
        search_response = index.query(
            vector=generic_embedding,
            top_k=1,
            include_values=False,
            include_metadata=True,
            filter=metadata_filter
        )
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return "An error occurred while retrieving stock data."

    matches = search_response.get('matches', [])
    if matches:
        metadata = matches[0]['metadata']
        price = metadata.get(price_type, 'Unknown')
        symbol = metadata.get('symbol', 'Unknown')
        date = metadata.get('date', 'Unknown')
        return f"The {price_type} price of {symbol} on {date} was ${price}."
    else:
        return "I’m sorry, I could not find the information."

@lru_cache(maxsize=256)
def get_forecast_price(symbol, date):
    metadata_filter = {
        "type": {"$eq": "forecast"},
        "stock_symbol": {"$eq": symbol},
        "date": {"$eq": date}
    }
    generic_query = "forecast stock price query"
    generic_embedding = embedding_model.encode(generic_query).tolist()
    try:
        search_response = index.query(
            vector=generic_embedding,
            top_k=1,
            include_values=False,
            include_metadata=True,
            filter=metadata_filter
        )
    except Exception as e:
        logger.error(f"Error querying Pinecone (forecast): {e}")
        return "An error occurred while retrieving forecast data."

    matches = search_response.get('matches', [])
    if matches:
        metadata = matches[0]['metadata']
        predicted_price = metadata.get('predicted_price', 'Unknown')
        symbol = metadata.get('stock_symbol', 'Unknown')
        date = metadata.get('date', 'Unknown')
        return f"The predicted price of {symbol} on {date} is ${predicted_price}."
    else:
        return "I’m sorry, I could not find the information."

def chatbot_response(query):
    try:
        logger.info(f"Received query: {query}")

        # Check tweet query
        tweet_query_match = re.match(r"List of tweets by writer named\s*:\s*(.+)", query, re.IGNORECASE)
        if tweet_query_match:
            writer_name = tweet_query_match.group(1).strip().lower()
            metadata_filter = {
                "type": {"$eq": "tweet"},
                "writer": {"$eq": writer_name}
            }
            generic_query = "retrieve all tweets"
            try:
                generic_embedding = embedding_model.encode(generic_query).tolist()
            except Exception as e:
                logger.error(f"Error generating embedding for tweets: {e}")
                return "An error occurred while processing your request."

            try:
                search_response = index.query(
                    vector=generic_embedding,
                    top_k=100,
                    include_values=False,
                    include_metadata=True,
                    filter=metadata_filter
                )
            except Exception as e:
                logger.error(f"Error querying Pinecone for tweets: {e}")
                return "An error occurred while retrieving tweets."

            matches = search_response.get('matches', [])
            if not matches:
                return f"I’m sorry, I could not find any tweets by {writer_name}."

            tweets = []
            for match in matches:
                metadata = match.get('metadata', {})
                text = metadata.get('text', 'No content')
                post_date = metadata.get('post_date', 'Unknown Date')
                tweets.append(f"- {text} (Posted on {post_date})")

            max_display = 10
            tweets = tweets[:max_display]
            tweets_list = "\n".join(tweets)
            response = f"Here are some tweets by {writer_name.capitalize()}:\n{tweets_list}"
            return response

        # Check stock price query
        price_type, symbol, date = parse_stock_query(query)
        if price_type and symbol and date:
            logger.info(f"Detected stock price query: {price_type}, {symbol}, {date}")
            response = get_stock_price(price_type, symbol, date)
            return response

        # Check forecast query
        forecast_symbol, forecast_date = parse_forecast_query(query)
        if forecast_symbol and forecast_date:
            logger.info(f"Detected forecast price query for {forecast_symbol} on {forecast_date}")
            response = get_forecast_price(forecast_symbol, forecast_date)
            return response

        # General query (RAG)
        logger.info("Detected general query. Performing vector similarity search.")
        try:
            query_embedding = embedding_model.encode(query).tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for query: {e}")
            return "An error occurred while processing your query."

        top_k_results = 10
        try:
            search_response = index.query(
                vector=query_embedding,
                top_k=top_k_results,
                include_values=False,
                include_metadata=True,
                filter={
                    "type": {"$in": ["news", "news2", "stock", "tweet", "forecast"]}
                }
            )
        except Exception as e:
            logger.error(f"Error querying Pinecone for general query: {e}")
            return "An error occurred while retrieving information."

        matches = search_response.get('matches', [])
        if not matches:
            return "I’m sorry, I could not find the information."

        prompt = construct_prompt(query, matches)

        if not (tokenizer and model):
            logger.error("Tokenizer or model not initialized.")
            return "Sorry, the system is currently experiencing issues."

        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
                add_special_tokens=True
            ).to(model.device)
        except Exception as e:
            logger.error(f"Error tokenizing prompt: {e}")
            return "An error occurred while processing your request."

        # Generate response (deterministic)
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except Exception as e:
            logger.error(f"Error generating response from model: {e}")
            return "An error occurred while generating a response."

        try:
            generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            response = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()
        except Exception as e:
            logger.error(f"Error decoding model output: {e}")
            return "An error occurred while processing the model's response."

        response = response.replace('**', '')
        response_lines = response.split('\n')
        cleaned_lines = [line.strip() for line in response_lines if line.strip()]
        response = '\n'.join(cleaned_lines)
        if not response.endswith('.'):
            response += '.'

        if not response:
            return "I’m sorry, I could not find the information."
        else:
            return response

    except Exception as e:
        logger.error(f"An error occurred in chatbot_response: {e}")
        return "An error occurred while processing your request."

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/chatbot', methods=['POST'])
def chatbot_api():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Please provide a query in JSON format.'}), 400
    query = data['query']
    response = chatbot_response(query)
    return jsonify({'response': response})

if __name__ == "__main__":
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
