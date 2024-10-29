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
model_name = "meta-llama/Llama-3.2-3B-Instruct"
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
    embedding_model = embedding_model.to(
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
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
    # Organize context by type for better structure
    context_sections = {
        'stock': [],
        'news': [],
        'tweet': []
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
        elif metadata.get('type') == 'news' or metadata.get('type') == 'news2':
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

    # Build the context string
    context = ""
    for section, infos in context_sections.items():
        if infos:
            context += f"### {section.capitalize()} Information:\n"
            for info in infos:
                context += f"{info}\n\n"

    # Add few-shot examples to the prompt
    examples = """
**Examples:**

**Question:** What was the closing price of AAPL on September 19, 2023?
**Answer:** The closing price of AAPL on 2023-09-19 was $150.25.

**Question:** Summarize the latest news about Tesla (TSLA).
**Answer:** Tesla has recently unveiled its new Model Z, which promises to revolutionize electric vehicles. The car features advanced AI capabilities and longer battery life.

**Question:** What are people saying about Microsoft?
**Answer:** Recent tweets highlight Microsoft's latest software release, focusing on its improved user interface and enhanced security features.

**Question:** List the top 5 performing stocks in Q3 2023.
**Answer:**
1. **Apple Inc. (AAPL):** Increased by 15% due to strong iPhone sales.
2. **Microsoft Corp. (MSFT):** Rose by 12% following the release of new software.
3. **Amazon.com Inc. (AMZN):** Gained 10% driven by e-commerce growth.
4. **Alphabet Inc. (GOOGL):** Surged 8% thanks to advancements in AI technologies.
5. **Tesla Inc. (TSLA):** Boosted by a 7% rise from the launch of Model Z.
"""

    # Construct the prompt with detailed instructions
    prompt = f"""
You are a highly knowledgeable AI assistant specialized in financial markets, stocks, and related news. Your goal is to provide accurate, concise, and helpful answers to the user's questions based on the provided context.

**Guidelines:**
- Use the information in the context to formulate your response.
- If the answer is not in the context, politely inform the user that you don't have that information.
- Format your answer in clear, well-organized paragraphs.
- Use line breaks where appropriate to enhance readability.
- When listing multiple items, use numbered lists for clarity.
- Focus on answering the question directly and thoroughly.

{examples}

### Context:
{context}

### Question:
{query}

### Answer:
"""
    return prompt.strip()

def parse_stock_query(query):
    # Regex patterns to extract price type, stock symbol, and date
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
    try:
        search_response = index.query(
            vector=generic_embedding,
            top_k=1,
            include_values=False,
            include_metadata=True,
            filter=metadata_filter
        )
        logger.debug(f"Pinecone search response: {search_response}")
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return "An error occurred while retrieving stock data."

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
            try:
                generic_embedding = embedding_model.encode(generic_query).tolist()
            except Exception as e:
                logger.error(f"Error generating embedding for generic query: {e}")
                return "An error occurred while processing your request."

            # Perform a metadata-filtered search with the generic vector
            try:
                search_response = index.query(
                    vector=generic_embedding,
                    top_k=100,
                    include_values=False,
                    include_metadata=True,
                    filter=metadata_filter
                )
                logger.debug(f"Pinecone search response for tweets: {search_response}")
            except Exception as e:
                logger.error(f"Error querying Pinecone for tweets: {e}")
                return "An error occurred while retrieving tweets."

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
                try:
                    query_embedding = embedding_model.encode(query).tolist()
                    logger.debug(f"Generated query embedding.")
                except Exception as e:
                    logger.error(f"Error generating embedding for query: {e}")
                    return "An error occurred while processing your query."

                # Perform a similarity search in Pinecone
                try:
                    search_response = index.query(
                        vector=query_embedding,
                        top_k=5,  # Adjusted for better context
                        include_values=False,
                        include_metadata=True,
                        filter={
                            "type": {"$in": ["news", "news2", "stock", "tweet"]},
                            # Additional filters can be added here
                        }
                    )
                    logger.debug(f"Pinecone search response for general query: {search_response}")
                except Exception as e:
                    logger.error(f"Error querying Pinecone for general query: {e}")
                    return "An error occurred while retrieving information."

                # Check if any matches were found
                matches = search_response.get('matches', [])
                if not matches:
                    logger.info("No matches found in Pinecone.")
                    return "Sorry, I couldn't find any related information."

                # Re-rank matches based on score (already sorted by Pinecone)
                # Optionally, we could implement additional re-ranking here

                # Construct the prompt using retrieved information
                prompt = construct_prompt(query, matches)
                logger.debug(f"Constructed prompt for model.")

                if not (tokenizer and model):
                    logger.error("Tokenizer or model not initialized.")
                    return "Sorry, the system is currently experiencing issues and cannot generate a response right now."

                # Tokenize the prompt
                try:
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=2048,
                        padding=True,
                        add_special_tokens=True  # Ensure special tokens are included
                    ).to(model.device)
                    logger.debug(f"Tokenized inputs prepared.")
                except Exception as e:
                    logger.error(f"Error tokenizing prompt: {e}")
                    return "An error occurred while processing your request."

                # Generate response from the model
                try:
                    with torch.no_grad():
                        # Use sampling for more diverse responses
                        outputs = model.generate(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=250,  # Increased to allow for longer responses
                            do_sample=True,
                            top_p=0.85,          # Adjusted for better diversity
                            temperature=0.65,    # Slightly lower for more focused responses
                            num_return_sequences=1,  # Return one response
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    logger.debug(f"Model outputs generated.")
                except Exception as e:
                    logger.error(f"Error generating response from model: {e}")
                    return "An error occurred while generating a response."

                # Extract the generated response
                try:
                    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
                    response = tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=True
                    ).strip()
                    logger.debug(f"Decoded response: {response}")
                except Exception as e:
                    logger.error(f"Error decoding model output: {e}")
                    return "An error occurred while processing the model's response."

                # Clean up the response while preserving line breaks and numbered lists
                response_lines = response.split('\n')
                cleaned_lines = []
                for line in response_lines:
                    stripped_line = line.strip()
                    if stripped_line:
                        cleaned_lines.append(stripped_line)
                response = '\n'.join(cleaned_lines)
                if not response.endswith('.'):
                    response += '.'
                logger.info(f"Generated response: {response}")

                # Fallback mechanism
                if not response:
                    return "I'm sorry, I couldn't generate a response based on the provided information."
                else:
                    return response

    except Exception as e:
        logger.error(f"Error processing query '{query}': {e}")
        return "An error occurred while processing your request. Please try again later."

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
