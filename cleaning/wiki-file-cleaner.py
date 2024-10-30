import re
import pandas as pd
from bs4 import BeautifulSoup
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

