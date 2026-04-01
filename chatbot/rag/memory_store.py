import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List


# separate collection for conversations memory
MEMORY_CHROMA_PATH = 'chatbot/rag/memory_db'


def get_memory_collection(session_id: str):
    """
    Each session get its own memory collection.
    session_id = unique identifer per user/conversation
    """

    client = chromadb.PersistentClient(path=MEMORY_CHROMA_PATH)

    embedding_fu = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY'),
        model_name='text-embedding-3-small'
    )

    # Each session = separate collection
    # "memory_abc123" = memory  for session "abc123"
    collection = client.get_or_create_collection(
        name=f"memory_{session_id}",
        embedding_function=embedding_fu,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def save_message_to_memory(
    session_id: str,
    message_id: str,
    question:   str,
    answer:     str,
):
    """
    save a Q&A pair to embedding memory
    stores both question and answer as searchable text
    """

    collection = get_memory_collection(session_id)

    # combines Q+A so search can find by either
    combined_text = f"User: {question}\nAssistant: {answer}"

    collection.add(
        ids=[message_id],
        documents=[combined_text],
        metadatas=[{
            "question": question,
            "answer": answer,
            "session_id": session_id,
        }]
    )


def get_relevant_memory(
    session_id: str,
    current_question: str,
    n_results: int = 3,
    min_similarity: float = 0.1
) -> List[dict]:
    """
    Find past message most relevant to current answer
    Returns relevant Q&A pairs from entire history
    
    """

    collection = get_memory_collection(session_id)

    if collection.count() == 0:
        return []
    
    results = collection.query(
        query_texts=[current_question],
        n_results=min(n_results, collection.count()),
    )

    memories = []
    for i, doc in enumerate(results["documents"][0]):
        distance = results['distances'][0][i]
        similarity = 1 - distance

        if similarity < min_similarity:
            continue


        memories.append({
            "question": results['metadatas'][0][i]['question'],
            "answer": results['metadatas'][0][i]['answer'],
            "similarity": round(similarity, 4),
        })

    # sort by similarity - most relevant answer
    memories.sort(key=lambda x: x['similarity'], reverse=True)
    return memories



def get_memory_stats(session_id: str) -> dict:
    """How many messages stored for this session"""
    collection = get_memory_collection(session_id)
    return {
        "session_id": session_id,
        "total_messages": collection.count()
    }

