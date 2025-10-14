# app.py

import sys
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

import numpy.core.numeric as numeric
sys.modules['numpy._core.numeric'] = numeric

app = Flask(__name__)
CORS(app)  # allow frontend calls

INDEX_PATH = "index_14_10_2025.faiss"
INDEX_KW_PATH = "index_kw_14_10_2025.faiss"

CSV_PATH = "redis_export.csv" #"dataset_with_embeddings.csv"
EMBEDDINGS_PATH = "redis_export.pkl" #"dataset_with_parsed_embeddings.csv"

#KEYWORDS_PATH = "db_archives_18_09_2025_CATEGORIES_CLEANED.csv" #"df_with_category_28_05_2025_02.csv" #"dataset_with_parsed_embeddings.csv"
KEYWORDS_PATH = "df_keywords_14_10_2025.csv"
KEYWORDS_MAPPING = "keyword_mapping_14_10_2025.pkl"
KEYWORDS_INDICES = "keyword_to_df_indices_14_10_2025.pkl"
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
    fields = ['title', 'author', 'model_base', 'origin', 'singer']
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
    with open("keyword_mapping_14_10_2025.pkl", "rb") as f:
        keyword_to_index = pickle.load(f)
    with open("keyword_to_df_indices_14_10_2025.pkl", "rb") as f:
        keyword_to_df_indices = pickle.load(f)
else:
    print("Building FAISS index for keywords...")
    df_keywords = pd.read_csv(CSV_PATH)

    # Flatten keyword list and keep mapping
    df_keywords["keywords"] = df_keywords.apply(extract_keywords_from_fields, axis=1)

    #df_keywords.to_csv('df_keywords_28_05_2025.csv', index=False)
    df_keywords.to_csv(KEYWORDS_PATH, index=False)

    all_keywords = list(set([kw.strip() for sublist in df_keywords["keywords"] for kw in sublist]))
    # Generate embeddings
    embeddings = MODEL.encode(all_keywords)
    # Create FAISS index
    index_keywords = build_faiss_index(embeddings)
    faiss.write_index(index_keywords, INDEX_KW_PATH)

    # Save mapping
    keyword_to_index = {i: kw for i, kw in enumerate(all_keywords)}
    with open(KEYWORDS_MAPPING, "wb") as f:
        pickle.dump(keyword_to_index, f)

    for idx, row in df_keywords.iterrows():
        for kw in row["keywords"]:
            kw = kw.strip()
            keyword_to_df_indices[kw].append(idx)

    with open(KEYWORDS_INDICES, "wb") as f:
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
        app.logger.info(f"DataFrame columns: {list(df.columns)}")
        query_vec = df.loc[df['id'] == image_id, 'parsed_embedding'].values[0] #'parsed_embedding'].values[0]
        
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


# Assuming `get_images` (above) handles listing, and search (now named `gallery`) used for querying.
# @app.route('/gallery', methods=['GET', 'POST'])
# def gallery():
#     if request.method == 'GET':
#         # Now mirror /images with filtering.
#         return get_images()
#     elif request.method == 'POST':
#         try:
#             json_data = request.get_json()
#             if not json_data or 'query' not in json_data:
#                 return jsonify({"error": "Query must be provided as 'query': 'keyword' (or similar)"}), 400
#             query = json_data['query']
#             page = json_data.get('page', 1)
#             items_per_page = json_data.get('items_per_page', 20)
            
#             if not isinstance(page, int) or not isinstance(items_per_page, int):
#                 return jsonify({"error": "page and items_per_page must be integers"}), 400
            
#             if page < 1 or items_per_page < 1:
#                 return jsonify({"error": "page/item_per_page must be >=1"}), 400
            
#             # Now, integrate the filter logic here (either DF or index-based)
#             # Example using DF:
#             if 'category' in df.columns:  # Update if you have a populated 'category'
#                 if query:
#                     if isinstance(query, str):
#                         # Simple example, you need actual logic here
#                         filtered_ids = [i for i, row in df.iterrows() if query in row['category'].lower()]
#                     elif isinstance(query, list):
#                         # Multiple categories
#                         # (handling list queries here)
#                         # If 'category' is a column in your DF with multiple matches per row (e.g., tagging)
#                         # You'd typically want to check each category for at least one match:
#                         valid_ids = set()
#                         for cat in query:
#                             if cat in df['category'].values and any(df['category'].str.lower().str.contains(cat, case=False)):
#                                 valid_ids.update(df[df['category'].str.lower() == cat]['id'])
#                         if len(valid_ids) == 0:
#                             return jsonify({"message": "No matches found for any category"}), 404
#                         # Now SELECTing ids and handling pagination
#                         selected = valid_ids  # ideally with proper (start, end)
#                         # For clarity, if possible, say 'category' filtering is partial:
#                         selected = [all but filtered]  # if not all categories strictly match all
#                     else:
#                         return jsonify({"error": "Invalid query format (must be string or list of strings)"})
#             else:
#                 # Assume no category, fall back to full search (via index) but this would be slow
#                 # Possible merge of get_images logic here?

#             if not selected:
#                 return jsonify({"message": "No matching query found (check query validity)"}), 404

#             # Mock pagination: for simplicity
#             # If using actual pairwise ID list (if your filter is on IDs), just slice:
#             start = (page - 1) * items_per_page
#             end = start + items_per_page
#             if (start > (len(selected) - 1) or end > len(selected)):
#                 return jsonify({"message": "Pagination out of bounds for your results"}, 404)

#             images = [f"downloaded_images/{f}" for f in selected[start:end]]
#             return jsonify({
#                 "images": images,
#                 "total": len(selected),
#                 "page": page,
#                 "items_per_page": items_per_page
#             }), 200
#         except Exception as e:
#             return jsonify({"error": str(e)}), 500
#     else:
#         return jsonify({"error": "Method not allowed"}), 405
# Example POST query /gallery?query=architecture&page=2&items_per_page=5


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5030)
