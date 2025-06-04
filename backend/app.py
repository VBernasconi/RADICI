# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import ast
import os
import re
import faiss
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from image_search import build_faiss_index, load_embeddings, find_similar_images
from collections import defaultdict
from sentence_transformers import SentenceTransformer


app = Flask(__name__)
CORS(app)  # allow frontend calls

INDEX_PATH = "index.faiss"
INDEX_KW_PATH = "index_kw.faiss"

CSV_PATH = "dataset_with_embeddings.csv"
EMBEDDINGS_PATH = "dataset_with_embeddings.pkl" #"dataset_with_parsed_embeddings.csv"

KEYWORDS_PATH = "df_with_category_28_05_2025_02.csv" #"dataset_with_parsed_embeddings.csv"

# Load embedding model
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Map each keyword to the dataframe row indices where it appears
keyword_to_df_indices = defaultdict(list)

# Download stopwords and tokenizer resources once
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
nltk.download('stopwords')
# Define stopwords set
stop_words = set(stopwords.words('italian')) | set(stopwords.words('english'))  # if your text can be multilingual


def extract_keywords_from_title(title):
    if pd.isna(title):
        return []
    # Normalize and tokenize
    title_clean = re.sub(r'[^\w\s]', '', title.lower())
    tokens = word_tokenize(title_clean)
    # Filter out stopwords and short words
    keywords = [word for word in tokens if word not in stop_words and len(word) > 2]
    return keywords

def extract_keywords_from_fields(row):
    fields = ['title', 'author', 'description', 'origin', 'singer']
    all_text = ' '.join(str(row[field]) for field in fields if pd.notna(row[field]))
    return extract_keywords_from_title(all_text)

# Check if index exists
if os.path.exists(INDEX_PATH):
    print("Loading existing FAISS index and metadata...")
    df = pd.read_pickle(EMBEDDINGS_PATH)
    index = faiss.read_index(INDEX_PATH)
else:
    print("Building FAISS index...")
    df, embedding_matrix = load_embeddings(CSV_PATH)
    df.to_pickle(EMBEDDINGS_PATH)

    # Build the index
    index = build_faiss_index(embedding_matrix)
    faiss.write_index(index, INDEX_PATH)

# Check if index for keywords exists
if os.path.exists(INDEX_KW_PATH):
    print("Loading existing keywords FAISS index...")
    index_keywords = faiss.read_index(INDEX_KW_PATH)
    df_keywords = pd.read_csv(KEYWORDS_PATH)
    # Load keyword mapping
    with open("keyword_mapping.pkl", "rb") as f:
        keyword_to_index = pickle.load(f)
    with open("keyword_to_df_indices.pkl", "rb") as f:
        keyword_to_df_indices = pickle.load(f)
else:
    print("Building FAISS index for keywords...")
    df_keywords = pd.read_csv(KEYWORDS_PATH)

    # Flatten keyword list and keep mapping
    df_keywords["keywords"] = df_keywords.apply(extract_keywords_from_fields, axis=1)

    df_keywords.to_csv('df_keywords_28_05_2025.csv', index=False)

    all_keywords = list(set([kw.strip() for sublist in df_keywords["keywords"] for kw in sublist]))
    # Generate embeddings
    embeddings = MODEL.encode(all_keywords)
    # Create FAISS index
    index_keywords = build_faiss_index(embeddings)
    faiss.write_index(index_keywords, INDEX_KW_PATH)

    # Save mapping
    keyword_to_index = {i: kw for i, kw in enumerate(all_keywords)}
    with open("keyword_mapping.pkl", "wb") as f:
        pickle.dump(keyword_to_index, f)

    for idx, row in df_keywords.iterrows():
        for kw in row["keywords"]:
            kw = kw.strip()
            keyword_to_df_indices[kw].append(idx)

    with open("keyword_to_df_indices.pkl", "wb") as f:
        pickle.dump(keyword_to_df_indices, f)

@app.route('/images/<image_file>')
def get_image(image_file):
    if not image_file or image_file == "default":
        image_file = "default.jpeg"
    
    image_path = os.path.join("downloaded_images", image_file)
    
    if os.path.exists(image_path):
        return send_from_directory("downloaded_images", image_file)
    else:
        # Optionally serve default even if a wrong file was passed
        return send_from_directory("downloaded_images", "default.jpeg")


# Route for requests to /images/ with no file specified
@app.route('/images/')
def get_default_image():
    return send_from_directory("downloaded_images", "default.jpeg")
    
#def get_image(image_file):
#    return send_from_directory("downloaded_images", image_file)

@app.route('/search', methods=['POST'])
def search():
    json_data = request.get_json()

    if not json_data or 'image_id' not in json_data:
        print('No image_id provided')
        return jsonify({'error': 'No image_id provided'}), 400
    
    image_id = json_data['image_id']

    try:
        query_vec = df.loc[df['id'] == image_id, 'parsed_embedding'].values[0]
        
        results = find_similar_images(index, df, query_vec)
        if isinstance(results, str) and results == "NULL":
            return jsonify("ISSUE")
        else:
            cleaned_results = results.where(pd.notnull(results), None)  # replaces NaN with None
            id_list = cleaned_results.to_dict(orient='records')
            return jsonify(id_list)

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route("/search_keywords", methods=["GET"])
def search_keywords():
    global MODEL

    query = request.args.get("q", "").strip().lower()

    query_embedding = MODEL.encode([query]).astype('float32')
    D, I = index_keywords.search(query_embedding, k=5)

    matched_keywords = [keyword_to_index[i] for i in I[0]]
    print("Matched keywords:", matched_keywords)

    matched_row_indices = set()
    for kw in matched_keywords:
        matched_row_indices.update(keyword_to_df_indices.get(kw, []))

    # Get the rows corresponding to matched keywords ONLY THE FIRST 10!
    matched_rows = df_keywords.loc[list(matched_row_indices)].head(10)

    # Convert all to strings to ensure JSON serializable
    matched_rows = matched_rows.astype(str)

    # Optionally convert to dict/json
    results = matched_rows.to_dict(orient="records")

    # Show top results
    return jsonify(results)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5030)