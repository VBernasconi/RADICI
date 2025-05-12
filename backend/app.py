# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
import ast
import os
import faiss
from image_search import build_faiss_index, load_embeddings, find_similar_images


app = Flask(__name__)
CORS(app)  # allow frontend calls

INDEX_PATH = "index.faiss"
CSV_FILE = "dataset_with_embeddings.csv"
EMBEDDINGS_PATH = "dataset_with_embeddings.pkl" #"dataset_with_parsed_embeddings.csv"

REPO_ID = "VBernasconi/RADICI"
HF_TOKEN = "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"

# Download the file securely
CSV_PATH = hf_hub_download(
                           repo_id = REPO_ID,
                           filename = CSV_FILE,
                           repo_type = "dataset",
                           token = HF_TOKEN
)

# Check if index exists
if os.path.exists(INDEX_PATH):
    print("Loading existing FAISS index and metadata...")
    #df = pd.read_csv(EMBEDDINGS_PATH)
    df = pd.read_pickle(EMBEDDINGS_PATH)
    #df['parsed_embedding'] = df['parsed_embedding'].apply(lambda x: np.array(x, dtype=np.float32))
    index = faiss.read_index(INDEX_PATH)
else:
    print("Building FAISS index...")
    df, embedding_matrix = load_embeddings(CSV_PATH)
    df.to_pickle(EMBEDDINGS_PATH)
    #df['parsed_embedding'] = df['parsed_embedding'].apply(lambda x: str(list(x)))
    #df.to_csv(EMBEDDINGS_PATH, index=False)
    # Build the index
    index = build_faiss_index(embedding_matrix)
    faiss.write_index(index, INDEX_PATH)


@app.route('/images/<image_file>')
def get_image(image_file):
    return send_from_directory("downloaded_images", image_file)

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
            #id_list = results['id'].tolist()
            cleaned_results = results.where(pd.notnull(results), None)  # replaces NaN with None
            id_list = cleaned_results.to_dict(orient='records')
            return jsonify(id_list)

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5030)

