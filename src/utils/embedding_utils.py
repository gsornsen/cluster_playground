import hashlib
import pickle
import os
import aiohttp
import openai
from tqdm.asyncio import tqdm


# Directory to cache embeddings
CACHE_DIR = "embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to calculate a hash for the text data to check for cache
def calculate_text_hash(texts):
    """
    Calculates a hash for the text data.
    """
    text_concat = ''.join(texts)
    text_hash = hashlib.sha256(text_concat.encode()).hexdigest()
    return text_hash

# Function to save embeddings to disk
def save_embeddings_to_cache(embeddings, text_hash):
    """
    Saves embeddings to disk in a cache file with the text hash.
    """
    cache_path = os.path.join(CACHE_DIR, f"{text_hash}.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)

# Function to load embeddings from disk cache if available
def load_embeddings_from_cache(text_hash):
    """
    Loads cached embeddings from disk if available.
    """
    cache_path = os.path.join(CACHE_DIR, f"{text_hash}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

# Asynchronous function to fetch embeddings from OpenAI's API
async def fetch_embedding(session, text):
    """
    Asynchronous function to fetch embeddings from OpenAI's embedding API.
    """
    try:
        response = await session.post(
            "https://api.openai.com/v1/embeddings",
            json={"input": text, "model": "text-embedding-3-small"},
            headers={"Authorization": f"Bearer {openai.api_key}"}
        )
        result = await response.json()
        if response.status != 200:
            print(f"API Error: {result.get('error', {}).get('message', 'Unknown error')}")
            return None
        if 'data' not in result or not result['data']:
            print(f"Unexpected API response format: {result}")
            return None
        return result["data"][0]["embedding"]
    except Exception as e:
        print(f"Error fetching embedding: {str(e)}")
        return None

# Function to fetch embeddings in parallel with progress tracking
async def get_embeddings_in_parallel(texts):
    """
    Fetches embeddings for all texts in parallel using aiohttp and tqdm for progress.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_embedding(session, text) for text in texts]
        
        # Use tqdm to show the progress of embedding requests
        embeddings = []
        for task in tqdm.as_completed(tasks, total=len(tasks), desc="Embedding texts"):
            embedding = await task
            if embedding is not None:
                print(f"Found empty embedding: {embedding}")
                embeddings.append(embedding)
        
        return embeddings
