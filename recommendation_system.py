import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load and clean the data
df = pd.read_csv("qatar_airways_reviews.csv")

# Drop rows with critical missing data
df.dropna(subset=["Review Body", "Rating", "Author", "Route"], inplace=True)

# Clean and format columns
df["Author"] = df["Author"].str.strip().str.lower()
df["Review Body"] = df["Review Body"].astype(str)
df["Route"] = df["Route"].astype(str)
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df = df.dropna(subset=["Rating"])

# Create pivot table: rows = authors, columns = routes, values = ratings
pivot_table = df.pivot_table(index="Author", columns="Route", values="Rating")

# Filter the main df to only authors present in the pivot table (ensures alignment)
common_authors = pivot_table.index
df_filtered = df[df["Author"].isin(common_authors)]

# Reindex pivot table to maintain alignment
pivot_table = pivot_table.loc[common_authors]

# Collaborative filtering similarity
collab_similarity = cosine_similarity(pivot_table.fillna(0))

# Text-based similarity using TF-IDF of Review Body
grouped_reviews = df_filtered.groupby("Author")["Review Body"].apply(lambda texts: " ".join(texts)).reindex(common_authors)
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(grouped_reviews)
text_similarity = cosine_similarity(tfidf_matrix)

# Combine both (hybrid similarity)
hybrid_similarity = 0.4 * text_similarity + 0.5 * collab_similarity

# Recommendation function
def get_recommendations(name, top_n=5):
    name = name.lower().strip()
    if name not in common_authors:
        print(f"Reviewer '{name}' not found.")
        return

    idx = list(common_authors).index(name)
    sim_scores = list(enumerate(hybrid_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [x for x in sim_scores if x[0] != idx]
    top_users = [common_authors[i[0]] for i in sim_scores[:top_n]]

    print(f"\nTop {top_n} reviewers similar to '{name.title()}':\n")
    all_recommendations = []

    for i, user in enumerate(top_users, start=1):
        user_reviews = df[df["Author"] == user][["Review Body", "Route", "Rating"]].dropna()
        for _, row in user_reviews.iterrows():
            review = row["Review Body"]
            rating = row["Rating"]
            route = row["Route"]
            print(f"#{i}: '{user.title()}' on {route} - Rating: {rating}/10")
            print(f"Review: {review[:300]}{'...' if len(review) > 300 else ''}\n")

        # Generate simple recommendation based on review sentiment keywords
        sentiments = {
            "positive": ["good", "excellent", "great", "amazing", "pleasant", "comfortable"],
            "negative": ["bad", "poor", "delay", "terrible", "uncomfortable", "worst"]
        }

        user_text = " ".join(user_reviews["Review Body"].astype(str)).lower()
        pos_count = sum(user_text.count(w) for w in sentiments["positive"])
        neg_count = sum(user_text.count(w) for w in sentiments["negative"])

        if pos_count > neg_count:
            summary = "Likely a positive experience. Recommend similar route/class."
        elif neg_count > pos_count:
            summary = "Likely a negative experience. Suggest improving service on this route."
        else:
            summary = "Mixed experience. Further feedback may help."

        print(f"Summary recommendation for '{user.title()}': {summary}\n{'-'*60}")
        all_recommendations.append((user, summary))

# Example usage:
user_input = input("Enter the reviewer's name you want to check (e.g., 'mary lee'): ")
get_recommendations(user_input, top_n=3)
