import warnings
import pandas as pd
import torch
from torch import device
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def clean_text(text):
    # Suppress the BeautifulSoup warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=MarkupResemblesLocatorWarning)
        text = BeautifulSoup(text, 'html.parser').get_text()

    # Remove non-alphabetic characters
    text = ''.join([char.lower() for char in text if char.isalpha() or char.isspace()])

    return text


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)


def load_sentiment_data_csv(dataset_path):
    df = pd.read_csv(dataset_path)

    # Cleaning and preprocessing
    df['review'] = df['review'].apply(clean_text)
    df['review'] = df['review'].apply(preprocess_text)

    # Convert string labels to numeric values
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    texts = df['review'].tolist()
    labels = df['sentiment'].tolist()
    return texts, labels


def tokenize_and_prepare_data(texts, labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize and encode the data
    tokenized_data = tokenizer.batch_encode_plus(
        texts,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Prepare DataLoader
    dataset = TensorDataset(tokenized_data['input_ids'], tokenized_data['attention_mask'], torch.tensor(labels))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return train_dataloader, test_dataloader

def train(model, train_dataloader, num_epochs=3, learning_rate=2e-5, gradient_accumulation_steps=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    total_loss = 0
    accumulation_steps = 0

    for epoch in range(num_epochs):
        model.train()

        for batch in train_dataloader:
            input_ids, attention_mask, labels_batch = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()

            # Accumulate gradients for a certain number of batches
            accumulation_steps += 1
            if accumulation_steps == gradient_accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()
                accumulation_steps = 0

        average_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}')



def evaluate(model, test_dataloader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels_batch = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'Accuracy: {accuracy}')
    print(classification_report(all_labels, all_predictions))


def save_model(model, model_path='fine_tuned_bert_model'):
    model.save_pretrained(model_path)


def main():
    dataset_path = 'a1_IMDB_Dataset.csv'
    texts, labels = load_sentiment_data_csv(dataset_path)
    train_dataloader, test_dataloader = tokenize_and_prepare_data(texts, labels)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    train(model, train_dataloader)
    evaluate(model, test_dataloader)
    save_model(model)


if __name__ == "__main__":
    main()
