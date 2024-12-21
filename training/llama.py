import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_scheduler
import numpy as np

# -------------------------------
# Dataset Preparation
# -------------------------------
class TokenizedDataset(Dataset):
    def __init__(self, tokenized_data, seq_length):
        self.data = tokenized_data
        self.seq_length = seq_length

    def __len__(self):
        # Ensure len is non-negative
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx):
        input_tokens = self.data[idx:idx + self.seq_length]
        target_tokens = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)

# -------------------------------
# Model Definition (LLaMA-Style)
# -------------------------------
class LLaMA(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, seq_length, word2vec_embeddings=None):
        super(LLaMA, self).__init__()
        if word2vec_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(word2vec_embeddings, dtype=torch.float32))
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, embedding_dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                activation="gelu"
            ) for _ in range(num_layers)
        ])
        self.transformer_encoder = nn.TransformerEncoder(nn.Sequential(*self.layers), num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        transformer_output = self.transformer_encoder(embeddings.permute(1, 0, 2))
        logits = self.fc_out(transformer_output.permute(1, 0, 2))
        return logits

# -------------------------------
# Training Configuration
# -------------------------------
def train(model, dataloader, optimizer, scheduler, device, epochs=3):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss / len(dataloader):.4f}")

# -------------------------------
# Main Training Script
# -------------------------------
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 30000  # Adjust based on your tokenizer's vocabulary size
    embedding_dim = 512
    num_heads = 8
    num_layers = 6
    hidden_dim = 2048
    seq_length = 128
    batch_size = 32
    learning_rate = 5e-4
    num_epochs = 3

    # Load your data
    tokenized_data = np.load("C:/Users/L E G I O N/Desktop/Nepali-LLM/Preprocessing/sentencepiece_tokenized.pkl", allow_pickle=True)  # Replace with your data path
    dataset = TokenizedDataset(tokenized_data, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load pre-trained embeddings if available
    word2vec_embeddings = np.load("processed_normalized.word2vec.wv.vectors.npy")  # Replace with your embedding path

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLaMA(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        seq_length=seq_length,
        word2vec_embeddings=word2vec_embeddings
    ).to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(dataloader) * num_epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

    # Train the model
    train(model, dataloader, optimizer, scheduler, device, epochs=num_epochs)

    # Save the model
    torch.save(model.state_dict(), "llama_model.pth")
