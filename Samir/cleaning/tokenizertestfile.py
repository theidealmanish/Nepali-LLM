# tokenizer.py

from datasets import load_dataset, Dataset
import sentencepiece as spm
import re

def split_text_into_chunks(text, chunk_size=256, stride=128):
    cleaned_text = text.replace('------', ' ').strip()
    tokens = cleaned_text.split()
    chunks = []
    num_tokens = len(tokens)
    for i in range(0, num_tokens, chunk_size - stride):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = ' '.join(chunk_tokens)
        if chunk_text:
            chunks.append(chunk_text)
    return chunks

def prepare_chunks(dataset, chunk_size=256, stride=128):
    text_chunks = []
    for example in dataset:
        text = example.get('text', '').strip()
        if text:
            chunks = split_text_into_chunks(text, chunk_size, stride)
            text_chunks.extend(chunks)
    return Dataset.from_dict({'text': text_chunks})

def clean_text(text):
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)  # Keep only Nepali script
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load the merged corpus and prepare chunks
merged_corpus_path = "C:/Users/L E G I O N/Desktop/Nepali-LLM/data/merged_nepali_corpus.txt"
dataset = load_dataset('text', data_files={'train': merged_corpus_path})
chunked_dataset = prepare_chunks(dataset['train'])

# Write cleaned chunks to file
chunked_file_path = "C:/Users/L E G I O N/Desktop/Nepali-LLM/data/cleaned_chunked_nepali_corpus.txt"
with open(chunked_file_path, 'w', encoding='utf-8') as f:
    for chunk in chunked_dataset:
        cleaned_chunk = clean_text(chunk['text'])
        if cleaned_chunk:
            f.write(cleaned_chunk + '\n')

def train_sentencepiece_tokenizer(input_file, model_prefix, vocab_size=50000, model_type='unigram'):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.995,  # Exclude rare characters
        byte_fallback=False,
        add_dummy_prefix=False,
        shuffle_input_sentence=True,
        treat_whitespace_as_suffix=True,
        max_sentence_length=8192
    )
    print(f"Tokenizer trained with prefix '{model_prefix}'.")

# Train the tokenizer
tokenizer_model_prefix = "C:/Users/L E G I O N/Desktop/Nepali-LLM/data/nepali_tokenizer_2"
train_sentencepiece_tokenizer(chunked_file_path, tokenizer_model_prefix)

# Load and test tokenizer
sp = spm.SentencePieceProcessor(model_file=f"{tokenizer_model_prefix}.model")

# Encode and decode sample text
sample_text = "नेपालको राजधानी काठमाडौँ हो।"
encoded_text = sp.encode(sample_text, out_type=int)
decoded_text = sp.decode(encoded_text)

# Debugging output
print(f"Encoded text: {encoded_text}")  # Check encoded token IDs
print(f"Decoded text: {decoded_text}")  # Check decoded text
print("\nToken mappings:")
for token_id in encoded_text:
    print(f"{token_id}: {sp.id_to_piece(token_id)}")  # Show mapping from ID to token

# Check for any negative token IDs
negative_tokens = [token_id for token_id in encoded_text if token_id < 0]
if negative_tokens:
    print(f"Warning: Found negative token IDs: {negative_tokens}")
else:
    print("No negative token IDs found.")
