from transformers import BertTokenizerFast, BertModel
import os
import codecs

def read_nepali_text_file(filename):
    """Read Nepali text from a file."""
    if filename is None or not os.path.isfile(filename):
        raise ValueError("The provided filename is invalid or the file does not exist.")
    
    with codecs.open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def train_bert_tokenizer(filename, vocab_size, output_dir):
    """Train a WordPiece tokenizer for BERT on the given text file and save the tokenizer."""
    # Initialize the tokenizer with a pre-trained model
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

    # Read the text from the file
    nepali_text = read_nepali_text_file(filename)
    
    # Split text into sentences for training
    nepali_texts = nepali_text.splitlines()  # Adjust this based on your text structure

    # Train the tokenizer on the text
    tokenizer.train_new_from_iterator(nepali_texts, vocab_size=vocab_size)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the trained tokenizer in the specified output directory
    tokenizer.save_pretrained(output_dir)
    
    print(f"Tokenizer trained and saved to {output_dir}")

    # Display some of the most common subwords
    display_common_subwords(tokenizer)

def display_common_subwords(tokenizer, top_n=100):
    """Display the top N common subwords from the trained vocabulary."""
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])  # Sort by frequency
    print(f"\nTop {top_n} common subwords in the vocabulary:")
    for token, _ in sorted_vocab[:top_n]:
        print(f"'{token}'")

def main():
    # Specify the path to your single Nepali text file
    filename = 'C:/Users/L E G I O N/Desktop/Nepali-LLM/data/merged_nepali_corpus.txt'  # Your Nepali text file
    output_dir = 'C:/Users/L E G I O N/Desktop/Nepali-LLM/data'  # Directory to save the tokenizer files
    vocab_size = 10000  # Adjust the vocabulary size as needed

    # Check if the file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")

    # Train and save the BERT tokenizer
    train_bert_tokenizer(filename, vocab_size, output_dir)

    # Load the trained tokenizer to verify it's compatible with BertModel
    tokenizer = BertTokenizerFast.from_pretrained(output_dir)
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
    
    # Test the tokenizer with a sample sentence
    test_sentence = "नेपालको राजधानी काठमाडौँ हो।"
    encoding = tokenizer(test_sentence, return_tensors="pt")
    output = model(**encoding)

    print("Sample Encoding:", encoding)
    print("Model Output:", output)

if __name__ == "__main__":
    main()
