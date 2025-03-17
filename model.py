from textblob import TextBlob

def analyze_sentiment(sentence):
    blob = TextBlob(sentence)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"
# import pandas as pd
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# from datasets import Dataset



# # Load dataset

# df = pd.read_csv("train.csv", encoding="ISO-8859-1")
#  # Replace with actual file path

# # Select required columns
# df = df[['text', 'sentiment']]

# # Map sentiment labels to numerical values
# label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
# df['label'] = df['sentiment'].map(label_map)

# # Load BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # Tokenization function
# def tokenize_function(examples):
#     if "text" not in examples:
#         raise ValueError("Column 'text' not found in dataset.")

#     texts = examples["text"]
#     if isinstance(texts, str):  # Single string case
#         texts = [texts]
#     elif not isinstance(texts, list):
#         raise ValueError("Unexpected format for 'text' column.")

#     return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

# # Convert to Hugging Face dataset
# df = df.dropna()  # Remove rows with missing text

# dataset = Dataset.from_pandas(df)
# dataset = dataset.map(tokenize_function, batched=True)

# # Split into train and test sets
# train_test_split = dataset.train_test_split(test_size=0.2)
# train_dataset = train_test_split["train"]
# test_dataset = train_test_split["test"]

# # Load pre-trained BERT model
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     save_strategy="epoch",
#     logging_dir="./logs"
# )

# # Define Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset
# )

# # Train the model
# trainer.train()


# def analyze_sentiment(sentence):
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     model = BertForSequenceClassification.from_pretrained("results/checkpoint-3")
#     model.eval()

#     inputs = tokenizer(sentence, return_tensors="pt")
    
#     # Ensure the model runs without an unnecessary label
#     with torch.no_grad():
#         outputs = model(**inputs)

#     logits = outputs.logits
#     predicted_class = torch.argmax(logits, dim=-1).item()

#     label_map = {0: "negative", 1: "neutral", 2: "positive"}
#     return label_map.get(predicted_class, "unknown")  # Safe dictionary lookup

