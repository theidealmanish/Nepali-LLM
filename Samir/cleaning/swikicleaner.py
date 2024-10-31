import pandas as pd
import sentencepiece as spm
from datasets import Dataset, load_dataset

def split_text_into_chunks(text, chunk_size=256, stride=128):
    """Split text into overlapping chunks for training."""
    # Clean the text
    cleaned_text = text.replace('------', ' ').strip()
    chunks = []
    num_chars = len(cleaned_text)

    # Split the cleaned text into chunks with overlap (stride)
    for i in range(0, num_chars, chunk_size - stride):
        chunk = cleaned_text[i:i + chunk_size]
        if chunk:  # Ensure the chunk is not empty
            chunks.append(chunk)
            print(f"Created chunk {len(chunks)}: {len(chunk)} characters")  # Debugging statement

    return chunks

def prepare_chunks(dataset, chunk_size=256, stride=128):
    """Prepare chunks from the dataset."""
    text_chunks = []

    for idx, example in enumerate(dataset):
        text = example.get('text', '').strip()
        if not text:
            print(f"Skipping empty text at index {idx}")  # Debugging statement
            continue
        chunks = split_text_into_chunks(text, chunk_size, stride)
        text_chunks.extend(chunks)

    print(f"Total chunks created: {len(text_chunks)}")  # Debugging statement
    return Dataset.from_dict({'text': text_chunks})

def train_sentencepiece_tokenizer(input_file, model_prefix, vocab_size=32000, model_type='bpe'):
    """Train a SentencePiece tokenizer."""
    print("Training SentencePiece tokenizer...")
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.995,
        model_type=model_type,
        add_dummy_prefix=True,
        byte_fallback=False,
        shuffle_input_sentence=True,
        max_sentence_length=8192
    )
    print(f"Tokenizer trained with prefix '{model_prefix}'.")

# File paths and parameters
merged_corpus_path = "C:/Users/L E G I O N/Desktop/Nepali-LLM/data/merged_nepali_corpus.txt"
tokenizer_model_prefix = "C:/Users/L E G I O N/Desktop/Nepali-LLM/data/nepali_tokenizer"
vocab_size = 32000
model_type = 'bpe'

# Load the dataset
print("Loading dataset...")
dataset = load_dataset('text', data_files={'train': merged_corpus_path})

# Check if the dataset is loaded correctly
if 'train' not in dataset:
    raise ValueError("Dataset not loaded correctly. Check your file path and format.")

# Prepare the chunked dataset
chunked_dataset = prepare_chunks(dataset['train'])

# Save the chunked dataset to a file
chunked_file_path = "C:/Users/L E G I O N/Desktop/Nepali-LLM/data/chunked_nepali_corpus.txt"
with open(chunked_file_path, 'w', encoding='utf-8') as f:
    for chunk in chunked_dataset['text']:
        f.write(chunk + '\n')

# Train the SentencePiece tokenizer
train_sentencepiece_tokenizer(chunked_file_path, tokenizer_model_prefix, vocab_size, model_type)
