from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Example Nepali text data - replace this with loading your .txt files
nepali_text_data = [
    "तपाईंलाई कस्तो लाग्यो?",
    "यो कुरा मैले पहिले सुनेको थिएँ।",
    "यसको उत्तर के हो?"
]

# Tokenize each sentence and get input IDs
tokenized_data = [tokenizer(sentence, return_tensors="pt", padding=True, truncation=True) for sentence in nepali_text_data]

# Display tokenized results
for i, data in enumerate(tokenized_data):
    print(f"Original Text: {nepali_text_data[i]}")
    print(f"Token IDs: {data['input_ids']}")
    print(f"Attention Mask: {data['attention_mask']}\n")

# Load pre-trained BERT model
model = BertModel.from_pretrained("bert-base-multilingual-cased")

# Generate embeddings for each tokenized input
embeddings = []
for data in tokenized_data:
    with torch.no_grad():
        output = model(**data)
        embeddings.append(output.last_hidden_state)  # Last hidden layer representations

# Print vocabulary and embedding example
vocab = tokenizer.get_vocab()
print("Vocabulary Size:", len(vocab))
print("\nSample Token Embeddings for the first sentence:")

# Display embeddings for each token in the first sentence
for idx, token_id in enumerate(tokenized_data[0]["input_ids"][0]):
    token = tokenizer.convert_ids_to_tokens(token_id.item())  # Convert token_id to integer
    embedding = embeddings[0][0][idx]
    print(f"Token: {token}, Embedding: {embedding[:5]}")  # Show first 5 values in embedding for brevity


