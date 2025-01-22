import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler

# Load the data
data = pd.read_csv('fff.csv')

# Step 1: Dynamically extract question and answer columns
question_columns = [col for col in data.columns if "question" in col]
answer_columns = [col for col in data.columns if "answer" in col]

# Sort columns to maintain a consistent pairing
question_columns.sort()
answer_columns.sort()

# Check if 'user_response' column exists
if 'user_response' not in data.columns:
    data['user_response'] = ''

# Step 2: Combine questions, answers, and user response into a single text
def combine_features(row):
    combined_text = []
    for q_col, a_col in zip(question_columns, answer_columns):
        question = row[q_col] if not pd.isna(row[q_col]) else ""
        answer = row[a_col] if not pd.isna(row[a_col]) else ""
        combined_text.append(f"Q: {question} A: {answer}")
    return " ".join(combined_text) + f" User Response: {row['user_response']}"

data['combined_text'] = data.apply(combine_features, axis=1)

# Step 3: Create score categories based on bins
bins = [50, 125, 200, 250]
labels = ['Low', 'Moderate', 'High']
data['score_category'] = pd.cut(data['score'], bins=bins, labels=labels, right=False, include_lowest=True)

# Drop rows with missing score categories
data = data.dropna(subset=['score_category'])

# Encode score categories
label_encoder = LabelEncoder()
data['score_category_encoded'] = label_encoder.fit_transform(data['score_category'])

# Step 4: Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['combined_text'], data['score_category_encoded'], test_size=0.2, random_state=42
)

# Step 5: Tokenizer and Dataset preparation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

train_dataset = CustomDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
val_dataset = CustomDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Step 6: Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * 3  # Assuming 3 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Step 7: Train the model
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs=3):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1}: Train loss = {train_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
                val_loss += outputs.loss.item()
        print(f"Epoch {epoch + 1}: Validation loss = {val_loss / len(val_loader):.4f}")

train_model(model, train_loader, val_loader, optimizer, lr_scheduler, device)

# Step 8: Save the trained model
model.save_pretrained('./bert_score_model')
tokenizer.save_pretrained('./bert_score_model')
np.save('./label_classes.npy', label_encoder.classes_)

# Step 9: Predict score category
def predict_score(input_text, model, tokenizer, device):
    model.eval()
    encoding = tokenizer(
        input_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = probabilities.argmax().item()
        predicted_category = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_category

# Interactive Prediction Loop
while True:
    user_input = input("Enter your response (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the prediction loop. Goodbye!")
        break
    try:
        predicted_category = predict_score(user_input, model, tokenizer, device)
        print(f"Predicted Score Category: {predicted_category}")
    except Exception as e:
        print(f"Error: {e}")