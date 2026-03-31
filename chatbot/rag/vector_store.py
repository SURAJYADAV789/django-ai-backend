import os
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from .document_processor import DocumentChunk

# ChromaDB saves data here - persists between restarts
CHROMA_PATH = 'chatbot/rag/chroma_db'

def get_embeddings():
    '''
    Use AI models embeddings model to convert text -> vector.
    text-embedding-3-small is cheap  ($0.00002 per 1k token) and accurate.

    '''

    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY'),
        model_name='text-embedding-3-small'
    )


def get_collection(collection_name: str = 'documents'):
    '''
    Get or create a ChromaDB collections.
    A Collection =  a named group in vectors (like table in sql)
    '''
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_embeddings(),
        metadata={'hnsw:space': 'cosine'}  # cosine similarity for text
    )

    return collection



def add_chunks(chunks: List[DocumentChunk], collection_name: str = 'documents'):
    '''
    Embed and store chunks in chromaDB.
    Called once when you ingest a document.

    '''
    collection = get_collection(collection_name)

    # Prepare data from chromadb
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        # Id must be unique - use source + chunk index
        chunk_id = f'{chunk.source}_{chunk.chunk_index}'

        ids.append(chunk_id)
        documents.append(chunk.content)
        metadatas.append({
            'source' : chunk.source,
            'chunk_index' : chunk.chunk_index,
            'total_chunks' : chunk.total_chunk
        })

    # Add to chromaDB  - it auto-embeds using our embeddings functions
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas

    )
    print(f'Stored {len(chunks)} chunks in ChromaDB')


def search(
        query: str,
        collection_name: str = 'documents',
        n_result: int = 3  #  return top 3 most relevant chunks,
        ) -> List[dict]:
    '''
    Search for chunks most relevant to the query.
    Return list of {content, source,  score}
    '''

    collection = get_collection(collection_name)

    # check collections is not empty
    if collection.count() == 0:
        print('Vector store is empty. Run ingest_docs first.')
        return []
    
    results = collection.query(
        query_texts=[query],
        n_results=min(n_result, collection.count()), # cant exceed total docs
    )

    # format result cleanly
    chunks = []
    for i, doc in enumerate(results['documents'][0]):
        chunks.append({
            'content': doc,
            'source': results['metadatas'][0][i]['source'],
            'score':  results['distances'][0][i],  # lower = more similiar
        })

    return chunks


def delete_collection(collection_name: str = 'documents'):
    '''Delete all vector - useful for re-ignesting documents'''
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    client.delete_collection(collection_name)
    print(f'Deleted Collection {collection_name}')


def get_stats(collection_name: str = 'documents'):
    '''How many chunks are stored?'''

    collection = get_collection(collection_name)
    return {
        'collection': collection_name,
        'total_chunks': collection.count(),
    }


def semantic_search(
        query: str,
        collection_name: str = 'documents',
        n_results: int = 5,
        min_similarity: float = 0.3  # filter out low quality results
) -> List[dict]:
    
    """
    Enchanced semantic search with similarity filtering
    Only returns results above min_similarity threshold
    """
    collection = get_collection(collection_name)

    if collection.count() == 0:
        return []
    
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    chunks = []
    for i, doc in enumerate(results['documents'][0]):
        distance = results['distances'][0][i]
        similarity = 1 - distance  # convert distance -> similarity

        # filter not low quanlity results
        if similarity < min_similarity:
            continue

        chunks.append({
            "content": doc,
            "source": results['metadatas'][0][i]['source'],
            "chunk_index": results['metadatas'][0][i]['chunk_index'],
            "similarity": round(similarity, 4),  # higher -> more relevant
            "distance": round(distance, 4),    # lower -> more relevant     
        })


    chunks.sort(key=lambda x:x['similarity'], reverse=True)
    return chunks
