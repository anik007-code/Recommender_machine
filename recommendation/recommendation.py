# recommendation.py
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from data_loader.data_loader import load_data_from_json


class Recommendation:
    def __init__(self):
        self.data = self.load_data()
        self.job_embeddings = self.create_job_embeddings()

    def load_data(self):
        return load_data_from_json()

    def initialize_bert_model(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        return tokenizer, model

    def get_bert_embeddings(self, text):
        tokenizer, model = self.initialize_bert_model()
        input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids)
            embeddings = outputs[0][:, 0, :]
        return embeddings

    def create_job_embeddings(self, max_seq_length=512):
        tokenizer, model = self.initialize_bert_model()
        job_embeddings = {}

        for entry in self.data:
            title = entry["JobTitle"]
            content = entry["CleanContent"]

            # Tokenize the content
            tokens = tokenizer.tokenize(content)

            # Split the tokens into smaller chunks of max_seq_length
            token_chunks = [tokens[i:i + max_seq_length] for i in range(0, len(tokens), max_seq_length)]

            # Initialize embeddings for the title
            title_embedding = self.get_bert_embeddings(title)

            # Initialize embeddings for the content chunks
            content_embeddings = []
            for chunk in token_chunks:
                content_chunk = " ".join(chunk)
                content_embeddings.append(self.get_bert_embeddings(content_chunk))

            # Combine content embeddings if there are multiple chunks
            if len(content_embeddings) > 1:
                content_embedding = torch.cat(content_embeddings, dim=0)
            else:
                content_embedding = content_embeddings[0]

            # Combine title and content embeddings
            combined_embedding = torch.cat([title_embedding, content_embedding], dim=0)

            job_embeddings[title] = combined_embedding

        return job_embeddings

    def compute_similarity(self, embedding1, embedding2):
        return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

    def get_recommendations(self, input_job_title):
        input_embedding = self.job_embeddings.get(input_job_title)
        if input_embedding is None:
            return []

        similarities = []
        for title, embedding in self.job_embeddings.items():
            if title != input_job_title:
                similarity = self.compute_similarity(input_embedding, embedding)
                similarities.append((title, similarity))

        recommendations = sorted(similarities, key=lambda x: x[1], reverse=True)
        return recommendations[:5]

    def save_recommender_model(self, filename):
        joblib.dump(self.job_embeddings, filename)
