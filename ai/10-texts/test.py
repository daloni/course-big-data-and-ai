import torch
import pandas as pd
from transformers import AutoTokenizer
from SentimentModel import SentimentModel

MODEL_PATH = "model.pth"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

embed_size = 100
hidden_size = 128
num_classes = 2
vocab_size = tokenizer.vocab_size

model = SentimentModel(vocab_size, embed_size, hidden_size, num_classes)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def load_reviews_from_csv(file_path):
    df = pd.read_csv(file_path)
    positive_reviews = df[df['sentiment'] == 'positive'].head(5)
    negative_reviews = df[df['sentiment'] == 'negative'].head(5)
    filtered_reviews = pd.concat([positive_reviews, negative_reviews])
    reviews = filtered_reviews['review'].tolist()
    labels = filtered_reviews['sentiment'].tolist()
    return reviews, labels

def predict_sentiment(review):
    encoding = tokenizer(
        review,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

if __name__ == "__main__":
    csv_path = "./data/IMDB_Dataset.csv"
    reviews, labels = load_reviews_from_csv(csv_path)

    correct = 0
    num_tests = len(reviews)

    for i in range(num_tests):
        review = reviews[i]
        true_label = labels[i]
        true_label_binary = 1 if true_label == 'positive' else 0
        predicted_label = predict_sentiment(review)

        print(f"Review: {review}")
        print(f"Predicted Sentiment: {'Positive' if predicted_label == 1 else 'Negative'}")
        print(f"True Sentiment: {'Positive' if true_label_binary == 1 else 'Negative'}")
        print("Correct!" if predicted_label == true_label_binary else "Wrong!", end="\n\n")

        if predicted_label == true_label_binary:
            correct += 1

    print(f"Accuracy: {correct / num_tests * 100:.2f}%")
