import re
import pandas as pd
from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset
import sentencepiece as spm
from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer, pipeline



# THis was entirely ChatGPT , i just gave up midway man. 

def clean_text(text):
    """
    Cleans the input text by removing URLs, HTML tags, and excessive whitespace.
    
    Parameters:
        text (str): The text to clean.
        
    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_wikipedia(wikipedia_csv_path, output_file, delimiter="------", chunksize=1000):
    """
    Processes the Wikipedia CSV in chunks, cleans the text, and writes to the output file.
    
    Parameters:
        wikipedia_csv_path (str): Path to the Wikipedia CSV file.
        output_file (file object): File object to write the cleaned texts.
        delimiter (str): Delimiter to separate articles.
        chunksize (int): Number of rows per chunk.
        
    Returns:
        None
    """
    # Read CSV in chunks
    for chunk in pd.read_csv(wikipedia_csv_path, chunksize=chunksize):
        # Extract and clean 'Text' column
        texts = chunk['Text'].tolist()
        cleaned_texts = [clean_text(text) for text in texts if isinstance(text, str)]
        # Write each cleaned text to the output file with delimiter
        for text in cleaned_texts:
            if text:  # Ensure the text is not empty
                output_file.write(text + f"\n{delimiter}\n")
    print("Processed and wrote Wikipedia articles to the merged corpus.")

def process_compiled(compiled_txt_path, output_file, delimiter="------"):
    """
    Processes the compiled.txt file, cleans the text, and writes to the output file.
    
    Parameters:
        compiled_txt_path (str): Path to the compiled.txt file.
        output_file (file object): File object to write the cleaned texts.
        delimiter (str): Delimiter to separate articles.
        
    Returns:
        None
    """
    with open(compiled_txt_path, "r", encoding='utf-8') as f:
        compiled_data = f.read()
    
    soup = BeautifulSoup(compiled_data, 'html.parser')
    
    # Find all <archive_file> tags
    archive_files = soup.find_all('archive_file')
    
    count = 0
    for archive in archive_files:
        # Extract text from all <p> tags within each <archive_file>
        paragraphs = archive.find_all('p')
        article_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
        # Clean the article text
        article_text = clean_text(article_text)
        if article_text:
            output_file.write(article_text + f"\n{delimiter}\n")
            count += 1
    print(f"Extracted and wrote {count} articles from compiled.txt to the merged corpus.")

def merge_corpora(wikipedia_csv_path, compiled_txt_path, merged_corpus_path, delimiter="------"):
    """
    Merges Wikipedia and compiled articles into a single corpus file with delimiters.
    
    Parameters:
        wikipedia_csv_path (str): Path to the Wikipedia CSV file.
        compiled_txt_path (str): Path to the compiled.txt file.
        merged_corpus_path (str): Path to save the merged corpus.
        delimiter (str): Delimiter to separate articles.
        
    Returns:
        None
    """
    with open(merged_corpus_path, "w", encoding='utf-8') as output_file:
        # Process and write Wikipedia articles
        process_wikipedia(wikipedia_csv_path, output_file, delimiter=delimiter, chunksize=1000)
        
        # Process and write compiled.txt articles
        process_compiled(compiled_txt_path, output_file, delimiter=delimiter)
    
    print(f"Merged corpus saved to '{merged_corpus_path}'.")


    # Define file paths

wikipedia_csv_path = "../data/nepali_wikipedia_articles.csv"
compiled_txt_path = "../data/compiled.txt"
merged_corpus_path = "../data/merged_nepali_corpus.txt"
delimiter = "------"
    
#merge_corpora(wikipedia_csv_path, compiled_txt_path, merged_corpus_path, delimiter=delimiter)





# some entries have long text while some have short so balacing it out by splitting the long text into equal smaller chunks.

def split_text_into_chunks(text, chunk_size=512, stride=128):
    """Split raw text into overlapping chunks based on token count."""
    tokens = text.split()  # Simple word-based split
    chunks = []
    for i in range(0, len(tokens), chunk_size - stride):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = ' '.join(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def prepare_chunks(dataset, chunk_size=512, stride=128):
    """Split each article in the dataset into overlapping chunks."""
    all_chunks = []
    for example in dataset:
        text = example['text']
        if len(text) == 0:
            continue
        chunks = split_text_into_chunks(text, chunk_size, stride)
        for chunk in chunks:
            all_chunks.append({'text': chunk})
    return Dataset.from_dict(all_chunks)

def train_sentencepiece_tokenizer(input_file, model_prefix, vocab_size=32000, model_type='bpe'):
    """Train a SentencePiece tokenizer.""" # Use bytepiece model type for nepali language.
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0, 
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=''
    )
    print(f"Tokenizer trained with prefix '{model_prefix}'.")



merged_corpus_path = "../data/merged_nepali_corpus.txt" 
tokenizer_model_prefix = "../data/nepali_tokenizer"     
vocab_size = 32000
model_type = 'bpe'

train_sentencepiece_tokenizer(merged_corpus_path, tokenizer_model_prefix, vocab_size, model_type)


# Load the trained tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{tokenizer_model_prefix}.model")

# Verify with a sample sentence
sample_sentence = "नेपाल एक सुन्दर देश हो।"
tokens = tokenizer.tokenize(sample_sentence)
print(f"Sample Sentence Tokens: {tokens}")

# Verify with a complex word
complex_word = "अतिसूक्ष्मज्ञान"
tokens_complex = tokenizer.tokenize(complex_word)
print(f"Complex Word Tokens: {tokens_complex}")


# Load the merged corpus
dataset = load_dataset('text', data_files={'train': merged_corpus_path})
print(f"Loaded {len(dataset['train'])} articles for training.")

# Prepare chunks
chunks_dataset = prepare_chunks(dataset['train'], chunk_size=512, stride=128)
print(f"Total chunks created: {len(chunks_dataset)}")

def tokenize_batch(examples):
    """Tokenize the input text."""
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Apply tokenization
tokenized_dataset = chunks_dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
print("Tokenization of chunks completed.")



# Define GPT-2 configuration
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,  # Match tokenizer vocab size
    n_positions=512,                   # Max sequence length
    n_ctx=512,                         # Context size
    n_embd=768,                        # Embedding dimensions
    n_layer=12,                        # Number of transformer layers
    n_head=12,                         # Number of attention heads
    bos_token_id=2,
    eos_token_id=3,
    pad_token_id=0
)

# Initialize the GPT-2 model
model = GPT2LMHeadModel(config)
print("GPT-2 model initialized.")



# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_nepali_model",
    overwrite_output_dir=True,
    num_train_epochs=3,                # Number of epochs
    per_device_train_batch_size=4,     # Batch size per device
    save_steps=10000,                   # Save checkpoint every 10k steps
    save_total_limit=2,                 # Keep only the last 2 checkpoints
    logging_dir="./logs",
    logging_steps=500,                  # Log every 500 steps
    prediction_loss_only=True,
    fp16=True,                          # Use mixed precision
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
print("Trainer initialized.")

# Start training
trainer.train()

# Save the final model
trainer.save_model("./gpt2_nepali_model_final")
print("Training completed and model saved.")



# Initialize the text generation pipeline
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Define a Nepali prompt
prompt = "नेपाल एक सुन्दर देश हो"

# Generate text
generated_text = text_generator(prompt, max_length=50, num_return_sequences=1)
print("Generated Text:")
print(generated_text[0]['generated_text'])
