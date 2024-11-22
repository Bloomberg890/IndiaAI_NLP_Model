import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, recall_score, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import autocast  # Import autocast for mixed precision
# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Initialize stop words and stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define text preprocessing function
def preprocess_text(text):
    # Tokenize the text
    words = nltk.word_tokenize(text)
    # Remove stop words and apply stemming
    filtered_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
    # Join the processed words back into a single string
    return " ".join(filtered_words)

# Load the saved model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("category_subcategory_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("category_subcategory_model")

# Load test data
test_data = pd.read_csv("~/Downloads/test.csv")

# Combine text columns (adjust column names based on your dataset)
test_data['text'] = test_data['crimeaditionalinfo'].fillna("").astype(str)

# Preprocess text data
test_data['text'] = test_data['text'].apply(preprocess_text)

# Load label encoders (ensure these match those used during training)
category_encoder = LabelEncoder()
subcategory_encoder = LabelEncoder()
# Load the training data
train_data = pd.read_csv("~/Downloads/train.csv")

# Handle missing values (if any)
train_data["category"] = train_data["category"].fillna("Unknown")
train_data["sub_category"] = train_data["sub_category"].fillna("Unknown")

# Reinitialize the LabelEncoders
category_encoder = LabelEncoder()
subcategory_encoder = LabelEncoder()

# Fit the encoders on the training data
train_data['category_label'] = category_encoder.fit_transform(train_data['category'])
train_data['subcategory_label'] = subcategory_encoder.fit_transform(train_data['sub_category'])

# Recreate the mappings
category_mapping = dict(enumerate(category_encoder.classes_))
subcategory_mapping = dict(enumerate(subcategory_encoder.classes_))

# Check for unseen categories and map them to a default value
test_data['category_label'] = test_data['category'].apply(
    lambda x: category_encoder.transform([x])[0] if x in category_encoder.classes_ else -1
)

test_data['subcategory_label'] = test_data['sub_category'].apply(
    lambda x: subcategory_encoder.transform([x])[0] if x in subcategory_encoder.classes_ else -1
)

# Tokenize test data
def tokenize_texts(texts, tokenizer, max_length=256):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

test_encodings = tokenize_texts(test_data['text'], tokenizer).to(device)
true_category_labels = test_data['category_label'].values
true_subcategory_labels = test_data['subcategory_label'].values


# Gradient accumulation parameters
accumulation_steps = 30 # Adjust based on your memory constraints

# Predict function for the test dataset with gradient accumulation and GPU memory management
def predict(model, encodings, num_category_labels, device, accumulation_steps):
    model.eval()
    predicted_category_labels = []
    predicted_subcategory_labels = []

    # Split inputs for gradient accumulation
    total_steps = len(encodings['input_ids'])
    step_size = total_steps // accumulation_steps
    for i in range(0, total_steps, step_size):
        # Slice the input for this step
        batch_input = {key: val[i:i + step_size].to(device) for key, val in encodings.items()}

        with torch.no_grad():
            with autocast():  # Enable mixed precision
                outputs = model(**batch_input)
                logits = outputs.logits
                category_logits = logits[:, :num_category_labels]
                subcategory_logits = logits[:, num_category_labels:]
                predicted_category_labels.append(torch.argmax(category_logits, dim=1).cpu().numpy())
                predicted_subcategory_labels.append(torch.argmax(subcategory_logits, dim=1).cpu().numpy())

        # Clear GPU cache after each batch
        torch.cuda.empty_cache()

    # Flatten predictions
    predicted_category_labels = [label for batch in predicted_category_labels for label in batch]
    predicted_subcategory_labels = [label for batch in predicted_subcategory_labels for label in batch]
    return predicted_category_labels, predicted_subcategory_labels

# Number of labels (ensure this matches training setup)
num_category_labels = len(category_encoder.classes_)
num_subcategory_labels = len(subcategory_encoder.classes_)

# Make predictions
predicted_category_labels, predicted_subcategory_labels = predict(
    model, test_encodings, num_category_labels, device, accumulation_steps
)

# Compute metrics
category_accuracy = accuracy_score(true_category_labels, predicted_category_labels)
subcategory_accuracy = accuracy_score(true_subcategory_labels, predicted_subcategory_labels)
category_recall = recall_score(true_category_labels, predicted_category_labels, average="macro")
subcategory_recall = recall_score(true_subcategory_labels, predicted_subcategory_labels, average="macro")

# Print metrics
print(f"Category Accuracy: {category_accuracy:.4f}")
print(f"Subcategory Accuracy: {subcategory_accuracy:.4f}")
print(f"Category Recall: {category_recall:.4f}")
print(f"Subcategory Recall: {subcategory_recall:.4f}")


# Calculate F1 score for category and subcategory
category_f1 = f1_score(true_category_labels, predicted_category_labels, average="macro")
subcategory_f1 = f1_score(true_subcategory_labels, predicted_subcategory_labels, average="macro")

# Print F1 scores
print(f"Category F1 Score: {category_f1:.4f}")
print(f"Subcategory F1 Score: {subcategory_f1:.4f}")

# Plot for category
plt.scatter(true_category_labels, predicted_category_labels, alpha=0.5, color='blue')
plt.title("Regression Line for Category Predictions")
plt.xlabel("True Labels")
plt.ylabel("Predicted Labels")
plt.plot([min(true_category_labels), max(true_category_labels)],
         [min(true_category_labels), max(true_category_labels)], color="red")
plt.savefig("category_reg_line.png")

# Plot for subcategory
plt.scatter(true_subcategory_labels, predicted_subcategory_labels, alpha=0.5, color='green')
plt.title("Regression Line for Subcategory Predictions")
plt.xlabel("True Labels")
plt.ylabel("Predicted Labels")
plt.plot([min(true_subcategory_labels), max(true_subcategory_labels)],
         [min(true_subcategory_labels), max(true_subcategory_labels)], color="red")
plt.savefig("subcategory_reg_line.png")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(true_category_labels, predicted_category_labels)
# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(16, 12))  # Increase figure size for better readability
disp = ConfusionMatrixDisplay(
    cm,
    display_labels=category_encoder.classes_  # Use subcategory labels
)
disp.plot(cmap="viridis", ax=ax, xticks_rotation=60)  # Rotate x-axis labels for clarity
ax.set_title("Confusion Matrix: Category", fontsize=18)
ax.tick_params(axis="x", labelsize=10)  # Adjust font size for x-axis labels
ax.tick_params(axis="y", labelsize=10)  # Adjust font size for y-axis labels
plt.tight_layout() 

plt.savefig("confusion_matrix_category.png")

# Plot the confusion matrix for subcategories with enhanced readability
# Compute confusion matrix
subcategory_cm = confusion_matrix(true_subcategory_labels, predicted_subcategory_labels)

fig, ax = plt.subplots(figsize=(16, 12))  # Increase figure size for better readability
disp = ConfusionMatrixDisplay(
    subcategory_cm,
    display_labels=subcategory_encoder.classes_
)
disp.plot(cmap="viridis", ax=ax, xticks_rotation=60)  # Rotate x-axis labels for clarity

# Customize title and label sizes
ax.set_title("Confusion Matrix: Subcategory", fontsize=18)
ax.tick_params(axis="x", labelsize=12)  # Increase font size for x-axis labels
ax.tick_params(axis="y", labelsize=12)  # Increase font size for y-axis labels

# Tight layout to prevent overlapping
plt.tight_layout()
plt.savefig("confusion_matrix_subcategory.png")  # Save the plot
plt.show()

# F1 scores for comparison
metrics = ["Accuracy", "F1 Score"]
category_scores = [category_accuracy, category_f1]
subcategory_scores = [subcategory_accuracy, subcategory_f1]

# Plot the scores
x = np.arange(len(metrics))  # Label locations
width = 0.35  # Width of bars

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, category_scores, width, label="Category")
ax.bar(x + width/2, subcategory_scores, width, label="Subcategory")

# Add labels, title, and legend
ax.set_xlabel("Metrics")
ax.set_ylabel("Scores")
ax.set_title("Comparison of Category and Subcategory Metrics")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc="best")

# Display the plot
plt.tight_layout()
plt.savefig("metrics_comparison.png")



# Example metrics (replace these with actual calculated values)
category_accuracy = 0.82  # Example accuracy for categories
subcategory_accuracy = 0.76  # Example accuracy for subcategories

# Calculate misclassification rates
category_misclassification = 1 - category_accuracy
subcategory_misclassification = 1 - subcategory_accuracy

# Data for plotting
labels = ["Category", "Subcategory"]
accuracies = [category_accuracy, subcategory_accuracy]
misclassifications = [category_misclassification, subcategory_misclassification]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(labels))  # Label positions
bar_width = 0.4

# Bar plot
ax.bar(x - bar_width / 2, accuracies, bar_width, label="Accuracy", color="green")
ax.bar(x + bar_width / 2, misclassifications, bar_width, label="Misclassification", color="red")

# Add labels, title, and legend
ax.set_ylabel("Metric Values", fontsize=12)
ax.set_title("Scope for Improvement: Accuracy vs Misclassification", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.legend(fontsize=10)

plt.savefig("accuracy_improvement_scope.png")
print("Graph saved as 'accuracy_improvement_scope.png'")

