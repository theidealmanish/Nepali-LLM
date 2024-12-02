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
    for i in range(0,    num_chars, chunk_size - stride):
        chunk = cleaned_text[i:i + chunk_size]
        if chunk:  # Ensure the chunk is not empty
            chunks.append(chunk)
            print(f"Created chunk {len(chunks)}: {len(chunk)} characters")  # Debugging statement

    return chunks

def prepare_chunks(dataset, chunk_size=256, stride=128):
    """Prepare chunks from the dataset."""
    text_chunks = []

    for idx, example in enumerate(dataset):
        # Ensure 'text' key exists in example
        text = example.get('text', '').strip()
        if not text:
            print(f"Skipping empty text at index {idx}")  # Debugging statement
            continue
        chunks = split_text_into_chunks(text, chunk_size, stride)
        text_chunks.extend(chunks)

    print(f"Total chunks created: {len(text_chunks)}")  # Debugging statement
    return Dataset.from_dict({'text': text_chunks})

def train_sentencepiece_tokenizer(input_file, model_prefix, vocab_size=32000, model_type='bpe'):
    """Train a SentencePiece tokenizer with UTF-8 handling."""
    print("Training SentencePiece tokenizer...")
    try:
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=0.98,  # Lowered for Nepali characters
            model_type=model_type,
            add_dummy_prefix=True,
            byte_fallback=False,
            shuffle_input_sentence=True,
            max_sentence_length=8192
        )
        print(f"Tokenizer trained with prefix '{model_prefix}'.")
    except Exception as e:
        print(f"Error during tokenizer training: {e}")

# File paths and parameters
merged_corpus_path = "C:/Users/L E G I O N/Desktop/Nepali-LLM/data/chunked_nepali_corpus.txt"
tokenizer_model_prefix = "C:/Users/L E G I O N/Desktop/Nepali-LLM/data/nepali_tokenizer_hawa"
vocab_size = 32000
model_type = 'bpe'

# Load the dataset
print("Loading dataset...")
try:
    dataset = load_dataset('text', data_files={'train': merged_corpus_path})
    # Ensure dataset is loaded properly
    if 'train' not in dataset:
        raise ValueError("Dataset not loaded correctly. Check your file path and format.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    dataset = None  # Set dataset to None if there's an error

# Prepare the chunked dataset if dataset loaded successfully
if dataset:
    try:
        chunked_dataset = prepare_chunks(dataset['train'])
    except Exception as e:
        print(f"Error during chunk preparation: {e}")
        chunked_dataset = None  # Set to None if there's an error

# Save the chunked dataset to a file
if chunked_dataset:
    chunked_file_path = "C:/Users/L E G I O N/Desktop/Nepali-LLM/data/chunked_nepali_corpus.txt"
    try:
        with open(chunked_file_path, 'w', encoding='utf-8') as f:
            for chunk in chunked_dataset['text']:
                f.write(chunk + '\n')
        print("Chunked data saved successfully.")
    except Exception as e:
        print(f"Error saving chunked data: {e}")

# Train the SentencePiece tokenizer
if chunked_dataset:
    train_sentencepiece_tokenizer(chunked_file_path, tokenizer_model_prefix, vocab_size, model_type)
