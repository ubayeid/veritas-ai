"""
Text processing utilities.
"""

import os
import re
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def clean_text(text: str) -> str:
    """
    Clean and normalize text before chunking.
    Handles common PDF extraction artifacts and encoding issues.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove null bytes and other control characters (except newlines/tabs)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    
    # Normalize line breaks (convert all to spaces, will normalize whitespace later)
    text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    
    # Fix common PDF extraction issues
    # Remove page numbers (standalone numbers at start/end of lines)
    text = re.sub(r'^\d+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+\d+$', '', text, flags=re.MULTILINE)
    
    # Fix hyphenation (word- followed by newline/space and word continuation)
    # Pattern: word- space word (where second word likely continues first)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # Remove excessive whitespace (multiple spaces/tabs)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove special characters that are artifacts (but keep punctuation)
    # Remove non-printable unicode characters except common ones
    text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f\ufeff]', '', text)
    
    # Fix common encoding issues
    # Replace common mis-encoded characters
    replacements = {
        '\u2019': "'",  # Right single quotation mark
        '\u2018': "'",  # Left single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2026': '...', # Horizontal ellipsis
        '\u00a0': ' ',  # Non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove URLs (optional - might want to keep them)
    # text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses (optional)
    # text = re.sub(r'\S+@\S+', '', text)
    
    # Final normalization: trim and normalize whitespace
    text = text.strip()
    
    return text


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into overlapping chunks with sentence boundary awareness.
    Tries to split at sentence boundaries to maintain semantic coherence.
    
    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    # Clean text first (handles PDF artifacts, encoding issues, etc.)
    text = clean_text(text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Split into sentences (handle common sentence endings)
    # Pattern matches: . ! ? followed by space or end of string
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$'
    sentences = re.split(sentence_pattern, text)
    
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        # Fallback to character-based chunking if no sentences found
        return _chunk_by_characters(text, chunk_size, overlap)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed chunk_size
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
            # For overlap, include last few sentences from previous chunk
            overlap_sentences = []
            overlap_length = 0
            
            # Work backwards from current_chunk to get overlap
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1  # +1 for space
                else:
                    break
            
            # Start new chunk with overlap sentences
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length + (1 if current_chunk else 0)  # +1 for space
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)
    
    # Filter out empty chunks and ensure minimum size
    chunks = [chunk.strip() for chunk in chunks if chunk.strip() and len(chunk.strip()) > 50]
    
    # If sentence-based chunking produced too few chunks, fall back to character-based
    if len(chunks) < 2 and len(text) > chunk_size:
        return _chunk_by_characters(text, chunk_size, overlap)
    
    return chunks


def _chunk_by_characters(
    text: str,
    chunk_size: int,
    overlap: int
) -> List[str]:
    """
    Fallback: Split text by character count (original method).
    Used when sentence-based chunking fails.
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at word boundary if not at end of text
        if end < len(text):
            # Find last space in chunk
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * 0.7:  # If space is reasonably close to end
                chunk = chunk[:last_space]
                end = start + last_space
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return [chunk for chunk in chunks if chunk]  # Remove empty chunks
