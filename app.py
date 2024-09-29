from flask import Flask, render_template, request
import pandas as pd
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data (assuming 'overview' column contains movie descriptions)
movies_data = pd.read_csv('./movies.csv')
list_of_all_titles = movies_data['title'].tolist()
movie_descriptions = movies_data['overview'].tolist()  # Assuming descriptions

# ... (previous code)

# Handle missing values (choose one of the options)
movies_data = movies_data.dropna(subset=['overview'])  # Option 1: Remove rows with missing values

movie_descriptions = movies_data['overview'].tolist()

# Ensure correct data type
if not pd.api.types.is_string_dtype(movies_data['overview']):
    movies_data['overview'] = movies_data['overview'].astype('str')


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Generate TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(movie_descriptions)

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')  # Render the index.html template

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']  # Get movie name from form in index.html

    find_close_match = get_close_matches(movie_name, list_of_all_titles)
    close_match = find_close_match[0]

    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    movie_vector = tfidf_matrix[index_of_the_movie]  # Get vector for the close match

    # Calculate cosine similarity for all movies
    cosine_similarities = cosine_similarity(movie_vector.reshape(1, -1), tfidf_matrix)
    similarity_scores = list(enumerate(cosine_similarities.flatten()))  # Flatten and enumerate

    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for movie in sorted_similar_movies[1:31]:  # Limit to top 30 (exclude the close match itself)
        index = movie[0]
        title = movies_data[movies_data.index == index]['title'].values[0]
        recommended_movies.append(title)

    return render_template('index.html', recommended_movies=recommended_movies)


if __name__ == '__main__':
    app.run(debug=False)  # Run in debug mode for development


