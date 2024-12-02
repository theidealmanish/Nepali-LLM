from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, pipeline
from datasets import load_dataset

# Step 1: Load the .txt file dataset
# Replace "your_text_file.txt" with the path to your .txt file
dataset = load_dataset("text", data_files={"train": "C:/Users/L E G I O N/Desktop/Nepali-LLM/data/cleaned_chunked_nepali_corpus.txt"})

# Step 2: Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained("Shushant/NepNewsBERT")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 3: Set up data collator and training arguments
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

training_args = TrainingArguments(
    output_dir="./finetuned_model",
    evaluation_strategy="no",  # Disable evaluation
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    weight_decay=0.01,
    report_to="none",
)

# Step 4: Load the pre-trained model
model = AutoModelForMaskedLM.from_pretrained("Shushant/NepNewsBERT")

# Step 5: Initialize and train using Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Step 6: Save the fine-tuned model
trainer.save_model("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")

# Step 7: Use the fine-tuned model for masked language modeling
fine_tuned_model = AutoModelForMaskedLM.from_pretrained("./finetuned_model")
fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./finetuned_model")

# Fill-mask pipeline
fill_mask = pipeline("fill-mask", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)

# Example usage
input_text = f"नेपालको राजधानी {fine_tuned_tokenizer.mask_token} हो।"
print("\nMasked Language Modeling Results:")
print(fill_mask(input_text))

# Step 8: Generate word embeddings using the fine-tuned model
def generate_embeddings(text):
    """
    Generates embeddings for the provided Nepali text using the fine-tuned model.

    :param text: The input Nepali text.
    :return: The embeddings as a tensor.
    """
    inputs = fine_tuned_tokenizer(text, return_tensors="pt")
    outputs = fine_tuned_model(**inputs, output_hidden_states=True)
    embeddings = outputs.hidden_states[-1]
    return embeddings

# Example for generating embeddings
embedding_input = "नेपाल एक सुन्दर देश हो।"
embeddings = generate_embeddings(embedding_input)
print("\nGenerated Embeddings for the text:")
print(embeddings)
