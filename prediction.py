import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model and tokenizer
model_path = './bert_score_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Load the label encoder classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('./label_classes.npy', allow_pickle=True)

# Define the 10 important questions and their respective choices
important_questions = [
    "Usually, I don't feel tired, worn out, used up, or exhausted.",
    "The thought of an accident doesn't affect me.",
    "Tension in life doesn't affect my health.",
    "I always keep committed and involved.",
    "I believe that people are essentially good and can be trusted.",
    "In my daily life I get a chance to show how capable I am.",
    "I like to do any task at the right place and right time.",
    "I find it easy to make decisions.",
    "I have happy memories of the past.",
    "I am living the kind of life I wanted to."
]

choices = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]

# Function to predict score category
def predict_score(input_responses, model, tokenizer, device):
    combined_text = []
    for i, response in enumerate(input_responses):
        question = f"Q: {important_questions[i]}"
        combined_text.append(f"{question} A: {response}")
    combined_text = " ".join(combined_text)

    model.eval()
    encoding = tokenizer(
        combined_text,
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
def interactive_prediction():
    print("Please answer the following questions with one of the following options:")
    for idx, choice in enumerate(choices):
        print(f"{idx + 1}. {choice}")
    
    user_responses = []
    for i in range(len(important_questions)):
        while True:
            try:
                print(f"{i + 1}. {important_questions[i]}")
                user_response_idx = int(input("Enter the number corresponding to your choice: ")) - 1
                if 0 <= user_response_idx < len(choices):
                    user_responses.append(choices[user_response_idx])
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    predicted_category = predict_score(user_responses, model, tokenizer, device)
    print(f"Predicted Score Category: {predicted_category}")

# Start the interactive prediction loop
interactive_prediction()