import os
from dataclasses import dataclass
from typing import List
from .vector_store import search
from ..ai_providers.router import get_provider


@dataclass
class RAGResponse:
    '''
    Response from RAG pipeline.
    Includes answer + the source used to generate it.
    '''
    answer: str
    sources: List[str]   # which documents were used
    chunks: List[str]   # actual text chunks used
    provider: str
    model: str

def build_rag_prompt(question: str, chunks: List[dict]) -> str:
    '''
    Builds the prompt that injects retrieved chunks as content.
    
    This is the core of RAG - we tell to AI:
    1. Here is the relevant information from the documents.
    2. Use only this information to answer.
    3. If the answer isn't in the context, say so

    Why "use only this context"?
    -> Prevents AI from making up answer (hallucination)
    -> Forces AI answere from your documents only
    '''
    
    if not chunks:
        return question  # fallback just ask the question directly

    # foramt each chunk with its source
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}: {chunk['source']}]\n{chunk['content']}"
            
        )

    context = '\n\n'.join(context_parts)

    prompt = f'''You are a helpfull assistant that answer questions based on provided context

    CONTEXT FROM DOCUMENTS: {context}

    RULES:
    - Answere only using the context above
    - If the answer is not in the context, say "I don't have information about that in the provided documents"
    - Always mention which source you need
    - Be concise and clear

    QUESTION: {question}
    
    ANSWER:'''

    return prompt


def ask_with_rag(
    question: str,
    collection_name: str ='documents',
    n_chunks: int = 3,  # how many chunks to retrieve
    conversation_history: List[dict] = None,  # Optional for memory

) -> RAGResponse:
    '''
    Main RAG function - retrieves context then generates answer.

    steps: 
    1. Search Vector DB for relevant chunks
    2. Build prompt with chunks as context
    3. Send to AI provider
    4. Return answer + source
    '''

    # step1 - Retrieve relevant chunks
    print(f"Searching for: '{question}")
    chunks = search(question, collection_name, n_result=n_chunks)

    if not chunks:
        # No documents injested yet
        return RAGResponse(
            answer="No documents have been loaded. please ingest documents first.",
            sources=[],
            chunks=[],
            provider='none',
            model='none',
        )
    
    #  step 2 - Build rag prompt with context 
    rag_prompt = build_rag_prompt(question, chunks)

    #  step 3 - Build messages list
    #  if conversation history exists, include in the memory
    messages = []
    if conversation_history:
        for msg in conversation_history:
            messages.append({'role': 'user', 'content': msg['question']})
            messages.append({'role': 'assistant', 'content': msg['answer']})

    # Add current RAG prompt as the latest user messages
    messages.append({'role': 'user', 'content': rag_prompt})

    # Add this debug line
    print(f"Message being sent: {messages}")

    # step 4 - call ths AI with full context
    provider = get_provider()
    result = provider.complete_with_messages(messages)

    # step 5 - Extract unique source lead
    sources = list(set(chunk['source'] for chunk in chunks))

    return RAGResponse(
        answer=result.answer,
        sources=sources,
        chunks=[chunk['content'] for chunk in chunks],
        provider=result.provider,
        model=result.model
    )

