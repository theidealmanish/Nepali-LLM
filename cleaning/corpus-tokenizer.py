from datasets import load_dataset, Dataset
import sentencepiece as spm

def split_text_into_chunks(text, chunk_size=256, stride=128): # maybe having somewhat uniform length of chunks is better for training the tokenizer.
    
    # Remove '------' delimiters
    cleaned_text = text.replace('------', ' ').strip()
    tokens = cleaned_text.split()
    chunks = []
    num_tokens = len(tokens)
    print(f"Total tokens in text: {num_tokens}")
    
    for i in range(0, num_tokens, chunk_size - stride):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = ' '.join(chunk_tokens)
        if chunk_text:
            chunks.append(chunk_text)
            print(f"Created chunk {len(chunks)}: {len(chunk_tokens)} words")
    
    return chunks

def prepare_chunks(dataset, chunk_size=256, stride=128):
    
    text_chunks = []
    total_examples = len(dataset)
    
    for idx, example in enumerate(dataset):
        text = example.get('text', '').strip()
        if not text:
            print(f"Skipping empty text at index {idx}")
            continue
        chunks = split_text_into_chunks(text, chunk_size, stride)
        text_chunks.extend(chunks)
    
    print(f"Total chunks created: {len(text_chunks)}")
    return Dataset.from_dict({'text': text_chunks})

def train_sentencepiece_tokenizer(input_file, model_prefix, vocab_size=32000, model_type='bpe'):
    """Train a SentencePiece tokenizer.""" # Use bytepiece model type for nepali language.
    spm.SentencePieceTrainer.train(
        treat_whitespace_as_suffix=True,
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.995, 
        model_type=model_type,
        add_dummy_prefix=True,
        byte_fallback=False,
        shuffle_input_sentence=True,
        max_sentence_length=8192 )
        # Common Nepali punctuation marks    )
    print(f"Tokenizer trained with prefix '{model_prefix}'.")


merged_corpus_path = "C:/Users/L E G I O N/Desktop/Nepali-LLM/data/merged_nepali_corpus.txt" 


tokenizer_model_prefix = "C:/Users/L E G I O N/Desktop/Nepali-LLM/data/nepali_tokenizer_2"     
vocab_size = 64000
model_type = 'bpe'


dataset = load_dataset('text', data_files={'train': merged_corpus_path})

chunked_dataset = prepare_chunks(dataset['train'])

chunked_file_path = "C:/Users/L E G I O N/Desktop/Nepali-LLM/data/chunked_nepali_corpus_samir.txt" # ensuring equal length of chunks for training the tokenizer.

with open(chunked_file_path, 'w', encoding='utf-8') as f:
    for chunk in chunked_dataset:
        f.write(chunk['text'] + '\n')

train_sentencepiece_tokenizer(chunked_file_path, tokenizer_model_prefix, vocab_size, model_type)
