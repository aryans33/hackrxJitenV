import os
import pickle
import json
import numpy as np
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from llama_index.embeddings.gemini import GeminiEmbedding
from dotenv import load_dotenv
from utils import get_document_id
import logging
import re
import pinecone

# Pinecone configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'YOUR_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV', 'YOUR_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'your-index-name')

# Pinecone init
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

def get_pinecone_index():
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(PINECONE_INDEX_NAME, dimension=768, metric='cosine')
    return pinecone.Index(PINECONE_INDEX_NAME)

def upsert_vectors_to_pinecone(vectors, ids):
    index = get_pinecone_index()
    items = list(zip(ids, vectors))
    index.upsert(vectors=items)

def query_pinecone(query_vector, top_k=5):
    index = get_pinecone_index()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=os.getenv("GOOGLE_API_KEY"))

VECTOR_STORE_DIR = "vector_store"
METADATA_FILE = "chunk_metadata.pkl"

MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)
EMBEDDING_BATCH_SIZE = 10
CHUNK_BATCH_SIZE = 50

class RAGIndex:
    """Standard RAG Index using dense retrieval with Pinecone and metadata."""
    def __init__(self):
        self.dense_index = get_pinecone_index()
        self.metadata = {}
        self.document_chunks = {}

    def create_or_load_index(self):
        start_time = time.time()
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        metadata_path = os.path.join(VECTOR_STORE_DIR, METADATA_FILE)

        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
            self.metadata = data.get('metadata', {})
            self.document_chunks = data.get('document_chunks', {})
            logger.info(f"Loaded index metadata from {metadata_path}")
        else:
            self.metadata = {}
            self.document_chunks = {}
            logger.info("Created new RAG index (Pinecone for dense vectors)")
        return self

def process_chunk_parallel(chunk_data):
    chunk, idx = chunk_data
    text = chunk['text']
    short_parts = split_into_short_chunks(text, max_chars=300)
    long_parts = split_into_long_chunks(text, max_chars=1000)
    short_chunks, long_chunks = [], []
    for i, short_text in enumerate(short_parts):
        short_chunks.append({**chunk, 'text': short_text, 'chunk_id': f"{chunk['chunk_id']}_short_{i}", 'chunk_type': 'short', 'parent_id': chunk['chunk_id']})
    for i, long_text in enumerate(long_parts):
        long_chunks.append({**chunk, 'text': long_text, 'chunk_id': f"{chunk['chunk_id']}_long_{i}", 'chunk_type': 'long', 'parent_id': chunk['chunk_id']})
    return short_chunks, long_chunks

def create_multi_level_chunks_parallel(chunks):
    all_short_chunks, all_long_chunks = [], []
    chunk_data = [(chunk, idx) for idx, chunk in enumerate(chunks)]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chunk = {executor.submit(process_chunk_parallel, data): data[1] for data in chunk_data}
        for future in as_completed(future_to_chunk):
            try:
                short_chunks, long_chunks = future.result()
                all_short_chunks.extend(short_chunks)
                all_long_chunks.extend(long_chunks)
            except Exception as exc:
                logger.error(f'Chunk {future_to_chunk[future]} exception: {exc}')
    return all_short_chunks, all_long_chunks

def process_embedding_batch(texts_batch):
    embeddings = []
    for text in texts_batch:
        try:
            result = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_document")
            embeddings.append(result['embedding'])
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            embeddings.append([0.0] * 768)
    return embeddings

async def generate_embeddings_parallel(texts):
    if not texts:
        return []
    batches = [texts[i:i + EMBEDDING_BATCH_SIZE] for i in range(0, len(texts), EMBEDDING_BATCH_SIZE)]
    all_embeddings = []
    with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
        future_to_batch = {executor.submit(process_embedding_batch, batch): idx for idx, batch in enumerate(batches)}
        batch_results = [None] * len(batches)
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_embeddings = future.result()
                batch_results[batch_idx] = batch_embeddings
            except Exception as exc:
                logger.error(f'Batch {batch_idx} exception: {exc}')
                batch_size = len(batches[batch_idx])
                batch_results[batch_idx] = [[0.0] * 768] * batch_size
    for batch_embeddings in batch_results:
        if batch_embeddings:
            all_embeddings.extend(batch_embeddings)
    return all_embeddings

async def add_chunks_to_index(rag_index: RAGIndex, chunks: List[Dict[str, Any]], document_path: str = None):
    if not chunks:
        logger.warning("No chunks to add.")
        return rag_index
    document_id = chunks[0]['document_id']

    # Remove existing chunks for this document
    remove_document_from_index(rag_index, document_id)

    # Create multi-level chunks
    short_chunks, long_chunks = create_multi_level_chunks_parallel(chunks)
    all_chunks = short_chunks + long_chunks
    rag_index.document_chunks[document_id] = {
        'original': chunks,
        'short': short_chunks,
        'long': long_chunks
    }

    # Generate embeddings in parallel
    chunk_texts = [chunk['text'] for chunk in all_chunks]
    embeddings = await generate_embeddings_parallel(chunk_texts)

    # Add to Pinecone
    ids = [f"{chunk['document_id']}_{chunk['chunk_id']}" for chunk in all_chunks]
    upsert_vectors_to_pinecone(embeddings, ids)

    for i, chunk in enumerate(all_chunks):
        rag_index.metadata[ids[i]] = chunk

    # Save metadata
    save_index(rag_index)
    return rag_index

def split_into_short_chunks(text: str, max_chars: int = 300) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r'[.!?]+', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def split_into_long_chunks(text: str, max_chars: int = 1000) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = text.split('\n\n')
    chunks, current_chunk = [], ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chars:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def search_index(rag_index: RAGIndex, query: str, document_id: str, top_k: int = 5) -> List[Dict]:
    # Dense (Pinecone) search only
    dense_results = dense_search(rag_index, query, document_id, top_k * 2)
    expanded_results = expand_with_context(rag_index, dense_results)
    return expanded_results

def dense_search(rag_index: RAGIndex, query: str, document_id: str, top_k: int) -> List[Dict]:
    # Get embedding for query
    query_embedding = generate_embeddings([query])[0]
    # Pinecone search
    result = rag_index.dense_index.query(vector=query_embedding, top_k=top_k*2, include_metadata=True)
    results = []
    for item in result['matches']:
        if item['id'] in rag_index.metadata:
            chunk_data = rag_index.metadata[item['id']]
            if chunk_data.get('document_id') == document_id:
                results.append({'score': float(item['score']), 'chunk_data': chunk_data, 'retrieval_type': 'dense'})
            if len(results) >= top_k:
                break
    return results

def expand_with_context(rag_index: RAGIndex, results: List[Dict]):
    expanded_results = []
    for result in results:
        chunk_data = result['chunk_data']
        chunk_type = chunk_data.get('chunk_type', 'original')
        if chunk_type == 'short':
            parent_id = chunk_data.get('parent_id')
            document_id = chunk_data.get('document_id')
            long_context = None
            if document_id in rag_index.document_chunks:
                for long_chunk in rag_index.document_chunks[document_id].get('long', []):
                    if long_chunk.get('parent_id') == parent_id:
                        long_context = long_chunk['text']
                        break
            result['context'] = long_context or chunk_data['text']
            result['short_text'] = chunk_data['text']
        else:
            result['context'] = chunk_data['text']
            result['short_text'] = chunk_data['text']
        expanded_results.append(result)
    return expanded_results

def remove_document_from_index(rag_index: RAGIndex, document_id: str):
    # Remove from metadata
    indices_to_remove = [idx for idx, chunk_data in rag_index.metadata.items() if chunk_data.get('document_id') == document_id]
    for idx in indices_to_remove:
        del rag_index.metadata[idx]
    # Remove from document_chunks
    if document_id in rag_index.document_chunks:
        del rag_index.document_chunks[document_id]
    if indices_to_remove:
        logger.info(f"Removed {len(indices_to_remove)} chunks for document {document_id}")

def save_index(rag_index: RAGIndex):
    metadata_path = os.path.join(VECTOR_STORE_DIR, METADATA_FILE)
    with open(metadata_path, 'wb') as f:
        pickle.dump({'metadata': rag_index.metadata, 'document_chunks': rag_index.document_chunks}, f)

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    embeddings = []
    for text in texts:
        result = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_document")
        embeddings.append(result['embedding'])
    return embeddings

# LEGACY: keep for backward compatibility, but only Pinecone is used.
def create_or_load_index():
    rag_index = RAGIndex()
    return rag_index.create_or_load_index()

def add_chunks_to_index(index, chunks: List[Dict[str, Any]], document_path: str = None):
    if isinstance(index, RAGIndex):
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(add_chunks_to_index(index, chunks, document_path))
    else:
        rag_index = RAGIndex()
        rag_index.dense_index = index
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(add_chunks_to_index(rag_index, chunks, document_path))

def search_document(index, metadata, query: str, document_id: str, top_k: int = 5):
    if isinstance(index, RAGIndex):
        return search_index(index, query, document_id, top_k)
    else:
        rag_index = RAGIndex()
        rag_index.dense_index = index
        rag_index.metadata = metadata
        return dense_search(rag_index, query, document_id, top_k)
