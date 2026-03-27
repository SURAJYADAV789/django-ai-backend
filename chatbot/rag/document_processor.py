import os
from typing import List
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    '''Represent a single chunk of text from a document'''
    content: str
    source : str  # filename
    chunk_index: int  # which chunk number
    total_chunk: int  # total chunk number


def read_pdf(filepath: str) -> str:
    '''Extract all text from pdf file'''
    try:
        import PyPDF2
        text = ""
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    except Exception as e:
        raise Exception(f'Failed to read pdf: {e}')
    

def read_text(filepath: str) -> str:
    '''Read plain text file'''
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()
    
from pathlib import Path
def read_document(filepath: str) -> str:
    '''
    Read any supported document typp.
    Supported: .pdf, .txt
    '''
    ext = Path(filepath).suffix.lower() 
    
    if ext == ".pdf":
        return read_pdf(filepath)
    elif ext == ".txt":
        return read_text(filepath)
    else:
        raise ValueError(f'Unsupported file type {ext}. Use .pdf or .txt')
    

def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,  # word per chunk
    overlap: int = 50, # word shard between chunk

) -> List[DocumentChunk]:
    
    '''
    Split text into overlaping chunks.

    overlap = 50 means:
    Chunk 1: words 1 - 500
    chunk 2: word 451 - 950  <- 50 words overlap with chunk1
    chunk 3: word 901 - 1400 <- 50 words overlap with chunk2

    Why overlap? so context at chunk boundaries isn't lost.
    '''
    words = text.split()
    chunks = []
    start = 0
    index = 0

    while start < len(words):
        end = start + chunk_size
        chunk_word = words[start:end]
        chunk_text = " ".join(chunk_word)

        # skip empty or very short chunks
        if len(chunk_word) > 20:
            chunks.append(
                DocumentChunk(
                    content=chunk_text,
                    source=os.path.basename(source),
                    chunk_index=index,
                    total_chunk=0, # will update below
                )
            )
            index += 1

        
        start += chunk_size - overlap  # Move forwad with overlap

    # update total chunk now we know the final count

    for chunk in chunks:
        chunk.total_chunks = len(chunks)
    return chunks


def process_document(filepath: str) -> List[DocumentChunk]:
    '''
    Main functions - read a documents and return a chunks.
    This is the only functions you need to call from outside 
    '''
    print(f'Reading: {filepath}')
    text = read_document(filepath)

    print(f'Chunking text ({len(text.split())} words...)')
    chunks = chunk_text(text, filepath)

    print(f"Created {len(chunks)} chunks from {os.path.basename(filepath)}")
    return chunks