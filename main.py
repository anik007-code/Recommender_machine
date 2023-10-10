# main.py
from recommendation.recommendation import Recommendation

if __name__ == "__main__":
    recommender = Recommendation()
    input_job_title = "Freelance Writer"
    recommendations = recommender.get_recommendations(input_job_title)

    for title, score in recommendations:
        print(f"Job Title: {title}, Similarity Score: {score:.2f}")

    recommender.save_recommender_model("bert_recommender_model.joblib")
