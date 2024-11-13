from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import codecs
import os

def read_nepali_text_file(filename):
    """Read Nepali text from a file."""
    with codecs.open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def train_tokenizer(text, vocab_size, output_dir):
    """Train a tokenizer on the given text and save the model and vocab."""
    # Initialize a tokenizer with BPE (Byte Pair Encoding) model
    tokenizer = Tokenizer(models.BPE())
    
    # Use a pre-tokenizer to split text into words
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Create a trainer with the specified vocabulary size
    trainer = trainers.BpeTrainer(vocab_size=vocab_size)
    
    # Train the tokenizer
    tokenizer.train_from_iterator([text], trainer=trainer)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model and vocab in .vocab and .model format
    tokenizer.model.save(output_dir)
    
    # Rename the files to the required extensions
    os.rename(f"{output_dir}/vocab.json", f"{output_dir}/your_model.vocab")
    os.rename(f"{output_dir}/merges.txt", f"{output_dir}/your_model.model")
    
    print(f"Tokenizer trained and saved as .vocab and .model in {output_dir}")

def main():
    # Specify the path to your Nepali text file
    filename = 'C:/Users/L E G I O N/Desktop/Nepali-LLM/data/merged_nepali_corpus.txt'  # Your Nepali text file
    output_dir = 'C:/Users/L E G I O N/Desktop/Nepali-LLM/data'  # Directory to save the vocab and model files
    vocab_size = 10000  # Adjust the vocabulary size as needed

    # Read the text from the file
    nepali_text = read_nepali_text_file(filename)
    
    # Train and save the tokenizer
    train_tokenizer(nepali_text, vocab_size, output_dir)

if __name__ == "__main__":
    main()
