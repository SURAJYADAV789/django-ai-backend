import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_embedding(text: str) -> list:
    """Convert text to embedding vector"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Calculate cosine similarity between two vectors
    Returns value between 0 and 1
    1.0 ->  identical meaning
    0.0 ->  completely different 
    """
    import numpy as np
    v1  = np.array(vec1)
    v2  = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


    
def compare_sentences(sentences: list) -> None:
    """
    Compare multiple sentences and show similarity score
    helps visualize which sentences mean similaity things
    """

    print("Getting embeddings...")
    embeddings = {}
    for sentence in sentences:
        embeddings[sentence] = get_embedding(sentence)
        print(f"'{sentence[:40]}....'  {len(embeddings[sentence])} dimensions")

    
    print("\n" + "=" * 60)
    print('Similarity score (1.0 = identical and 0.0 different)')
    print("=" * 60)

    # compare every pair
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            s1 = sentences[i]
            s2 = sentences[j]
            sim = cosine_similarity(embeddings[s1], embeddings[s2])

            # Visual bar
            bar   = "█" * int(sim * 20)
            label = "🟢 similar" if sim > 0.35 else "🟡 related" if sim > 0.15 else "🔴 different"

            print(f"\n{s1[:30]}...")
            print(f"\n{s2[:30]}...")
            print(f"Similarity: {sim:.4f} {bar} {label}")

    print("=" * 60)

