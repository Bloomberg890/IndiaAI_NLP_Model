import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download("stopwords")
nltk.download('punkt')

nltk.download('punkt_tab')

# Initialize stop words and stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
train_data = pd.read_csv("~/Downloads/train.csv")

# Combine text columns (adjust column names based on your dataset)
train_data['text'] = train_data['crimeaditionalinfo']
# Ensure all text data is valid during preprocessing
train_data['text'] = train_data['text'].fillna("").astype(str)

# Apply stop word removal and stemming
def preprocess_text(text):
    # Tokenize the text
    words = nltk.word_tokenize(text)
    # Remove stop words and apply stemming
    filtered_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
    # Join the processed words back into a single string
    return " ".join(filtered_words)

train_data['text'] = train_data['text'].apply(preprocess_text)

# Map labels to integers
category_encoder = LabelEncoder()
subcategory_encoder = LabelEncoder()

train_data['category_label'] = category_encoder.fit_transform(train_data['category'])
train_data['subcategory_label'] = subcategory_encoder.fit_transform(
    train_data['sub_category'].fillna("Unknown")
)

# Save mappings for later use
category_mapping = dict(enumerate(category_encoder.classes_))
subcategory_mapping = dict(enumerate(subcategory_encoder.classes_))

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    train_data['text'],
    train_data[['category_label', 'subcategory_label']],
    test_size=0.2,
    random_state=42
)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create a custom dataset
class TextDataset(Dataset):
    def __init__(self, texts, category_labels, subcategory_labels, tokenizer, max_length=256):
        self.texts = texts
        self.category_labels = category_labels
        self.subcategory_labels = subcategory_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        category_label = self.category_labels.iloc[idx]
        subcategory_label = self.subcategory_labels.iloc[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            **{key: val.squeeze(0) for key, val in encoding.items()},
            "category_label": torch.tensor(category_label, dtype=torch.long),
            "subcategory_label": torch.tensor(subcategory_label, dtype=torch.long),
        }

# Prepare datasets and dataloaders
train_dataset = TextDataset(X_train, y_train['category_label'], y_train['subcategory_label'], tokenizer)
val_dataset = TextDataset(X_val, y_val['category_label'], y_val['subcategory_label'], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Define the model for multi-label classification
num_category_labels = len(category_mapping)
num_subcategory_labels = len(subcategory_mapping)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_category_labels + num_subcategory_labels
)
model.to(device)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_batches = len(train_loader)
    for batch_idx, batch in enumerate(train_loader):
        # Move inputs to device
        inputs = {key: val.to(device) for key, val in batch.items() if key != "category_label" and key != "subcategory_label"}
        category_labels = batch["category_label"].to(device)
        subcategory_labels = batch["subcategory_label"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Separate logits for category and subcategory
        category_logits = logits[:, :num_category_labels]
        subcategory_logits = logits[:, num_category_labels:]

        # Compute loss for category and subcategory
        category_loss = torch.nn.CrossEntropyLoss()(category_logits, category_labels)
        subcategory_loss = torch.nn.CrossEntropyLoss()(subcategory_logits, subcategory_labels)

        # Total loss
        loss = category_loss + subcategory_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{total_batches}, Loss: {total_loss / (batch_idx + 1):.4f}")

    print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / total_batches:.4f}")

# Save the trained model and tokenizer
model.save_pretrained("category_subcategory_model")
tokenizer.save_pretrained("category_subcategory_model")
print("Model and tokenizer saved to 'category_subcategory_model'")

# Prediction function
def predict_category_and_subcategory(text):
    model.eval()
    # Preprocess input text
    preprocessed_text = preprocess_text(text)
    inputs = tokenizer(
        preprocessed_text,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        category_logits = logits[:, :num_category_labels]
        subcategory_logits = logits[:, num_category_labels:]

        predicted_category = torch.argmax(category_logits, dim=1).item()
        predicted_subcategory = torch.argmax(subcategory_logits, dim=1).item()

    return category_mapping[predicted_category], subcategory_mapping[predicted_subcategory]

# Example usage
sample_text = "Sample input text for prediction"
predicted_category, predicted_subcategory = predict_category_and_subcategory(sample_text)
print(f"Predicted Category: {predicted_category}")
print(f"Predicted Subcategory: {predicted_subcategory}")

