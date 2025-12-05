# app_redis.py

import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import redis
import json
import os
import re
import faiss
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sentence_transformers import SentenceTransformer

import numpy.core.numeric as numeric
sys.modules['numpy._core.numeric'] = numeric

# ---------------------------------------------------------
# Redis + Flask Setup
# ---------------------------------------------------------
r = redis.Redis(host='192.168.249.170', port=6379, db=10, decode_responses=True)
app = Flask(__name__)
CORS(app)

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------
# NLP Setup
# ---------------------------------------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.download('stopwords')

stop_words = set(stopwords.words('english') + stopwords.words('italian'))

import re

def get_next_object_id():
    keys = r.keys("*")  # all keys
    numeric_keys = [int(k) for k in keys if k.isdigit()]  # only numeric keys
    if not numeric_keys:
        return 1
    return max(numeric_keys) + 1

def extract_keywords(text):
    """Extract sanitized keywords from a text."""
    if not text:
        return []

    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = word_tokenize(text)

    keywords = [t for t in tokens if t not in stop_words and len(t) > 2]
    return keywords

# ---------------------------------------------------------
# Load FAISS index from Redis hashes
# ---------------------------------------------------------
def load_faiss_from_redis():
    keys = r.keys("*")
    objects = []
    embeddings = []

    for key in keys:
        obj = r.hgetall(key)
        if not obj:
            continue
        emb = obj.get("embeddings")
        if not emb:
            continue

        emb = np.array(json.loads(emb), dtype=np.float32)
        embeddings.append(emb)
        objects.append(obj)

    dim = 384  # embedding dimension of all-MiniLM-L6-v2
    if len(embeddings) == 0:
        return [], faiss.IndexFlatL2(dim)

    embeddings = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return objects, index


def load_keyword_index_from_redis():
    keys = r.keys("*")
    keyword_map = defaultdict(list)
    keyword_list = []

    for key in keys:
        obj = r.hgetall(key)
        if not obj:
            continue
        obj_id = obj["id"]  # keep as string

        text = " ".join([
            obj.get("title", ""),
            obj.get("description", ""),
            obj.get("author", ""),
            obj.get("origin", ""),
            obj.get("model_base", ""),
            obj.get("singer", "")
        ])

        kws = extract_keywords(text)
        for kw in kws:
            if kw not in keyword_map:
                keyword_list.append(kw)
            keyword_map[kw].append(obj_id)

    if len(keyword_list) == 0:
        dim = 384
        return {}, [], np.zeros((0, dim), dtype="float32"), faiss.IndexFlatL2(dim)

    keyword_embeddings = MODEL.encode(keyword_list).astype("float32")
    keyword_faiss = faiss.IndexFlatL2(keyword_embeddings.shape[1])
    keyword_faiss.add(keyword_embeddings)

    return keyword_map, keyword_list, keyword_embeddings, keyword_faiss

print("Loading Redis FAISS embeddings…")
redis_objects, redis_faiss = load_faiss_from_redis()

print("Loading keyword FAISS index…")
keyword_map, keyword_list, keyword_embeddings, keyword_faiss = load_keyword_index_from_redis()

# ---------------------------------------------------------
# UPLOAD IMAGES
# ---------------------------------------------------------
@app.route("/upload_image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = request.files["image"]
    filename = img.filename

    save_path = os.path.join("downloaded_images", filename)
    os.makedirs("downloaded_images", exist_ok=True)
    img.save(save_path)

    return jsonify({"filename": filename})

# ---------------------------------------------------------
# ADD OBJECT
# ---------------------------------------------------------
@app.route("/add_object", methods=["POST"])
def add_object():
    global redis_objects, redis_faiss
    global keyword_map, keyword_list, keyword_embeddings, keyword_faiss

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Assign string ID
    obj_id = str(get_next_object_id())
    print("NEW OBJECT ID INSERTED: ", obj_id)

    data["id"] = obj_id

    # Build combined text
    text = " ".join([
        data.get("title", ""),
        data.get("description", ""),
        data.get("author", ""),
        data.get("origin", ""),
        data.get("model_base", ""),
        data.get("singer", "")
    ])

    # EMBEDDING
    emb = MODEL.encode([text]).astype("float32")[0]
    data["embeddings"] = json.dumps(emb.tolist())

    # Convert list fields to JSON strings for Redis
    if isinstance(data.get("tags"), list):
        data["tags"] = json.dumps(data["tags"])

    # Save to Redis as hash
    r.hset(f"{obj_id}", mapping=data)

    # --- FAISS Safe Add ---
    if redis_faiss is None or redis_faiss.ntotal == 0:
        # First vector, create index with correct dimension
        redis_faiss = faiss.IndexFlatL2(emb.shape[0])
        
    elif redis_faiss.d != emb.shape[0]:
        # Dimension mismatch, recreate index (rare)
        print(f"[FAISS] Dimension mismatch, recreating index: {redis_faiss.d} -> {emb.shape[0]}")
        redis_faiss = faiss.IndexFlatL2(emb.shape[0])
        # Re-add all existing embeddings
        for o in redis_objects:
            e = np.array(json.loads(o["embeddings"]), dtype="float32").reshape(1, -1)
            redis_faiss.add(e)

    redis_faiss.add(emb.reshape(1, -1))
    redis_objects.append(data)

    # ---- Keyword extraction ----
    new_keywords = extract_keywords(text)
    for kw in new_keywords:
        if kw not in keyword_map:
            keyword_map[kw] = []
            keyword_list.append(kw)

            kw_emb = MODEL.encode([kw]).astype("float32")
            if len(keyword_embeddings) == 0:
                keyword_embeddings = kw_emb
                keyword_faiss = faiss.IndexFlatL2(kw_emb.shape[1])
                keyword_faiss.add(kw_emb)
            else:
                keyword_embeddings = np.vstack([keyword_embeddings, kw_emb])
                keyword_faiss.add(kw_emb)

        keyword_map[kw].append(obj_id)

    return jsonify({
        "object_id": obj_id,
        "added_keywords": new_keywords
    }), 201

# ---------------------------------------------------------
# IMAGE ROUTES
# ---------------------------------------------------------
@app.route('/images/<filename>')
def get_image(filename):
    if not filename or filename == "default":
        filename = "default.jpeg"
    img_path = os.path.join("downloaded_images", filename)
    if os.path.exists(img_path):
        return send_from_directory("downloaded_images", filename)
    return send_from_directory("downloaded_images", "default.jpeg")

@app.route('/images/')
def get_default_image():
    return send_from_directory("downloaded_images", "default.jpeg")

# ---------------------------------------------------------
# SEARCH (semantic)
# ---------------------------------------------------------
@app.route("/search", methods=["POST"])
def search():
    json_data = request.get_json()
    image_id = json_data.get("image_id", "").strip()  # <-- grab image_id
    k = json_data.get("k", 10)
    print("Received search request:", json_data)

    if not image_id:
        return jsonify([])

    # Find the object with this image_id in redis_objects
    obj = next((o for o in redis_objects if o.get("id") == image_id), None)
    if not obj:
        print(f"Image ID {image_id} not found")
        return jsonify([])

    # Fetch embedding (already stored as JSON string)
    emb_str = obj.get("embeddings")
    if not emb_str:
        print(f"No embedding for image ID {image_id}")
        return jsonify([])

    query_emb = np.array(json.loads(emb_str), dtype="float32").reshape(1, -1)

    # Make sure dimension matches FAISS index
    if query_emb.shape[1] != redis_faiss.d:
        print(f"Embedding dimension {query_emb.shape[1]} does not match FAISS index {redis_faiss.d}")
        return jsonify([])

    # Perform FAISS search
    D, I = redis_faiss.search(query_emb, min(k, len(redis_objects)))

    # Get the matching objects
    results = [redis_objects[idx] for idx in I[0]]

    return jsonify(results)

# ---------------------------------------------------------
# SEARCH KEYWORDS
# ---------------------------------------------------------
@app.route("/search_keywords", methods=["GET"])
def search_keywords():
    query = request.args.get("q", "").strip().lower()
    if not query or len(keyword_list) == 0:
        return jsonify({"results": []})

    # Embed the query text
    query_emb = MODEL.encode([query]).astype("float32")

    # Limit number of keyword matches
    k = min(10, len(keyword_list))
    D, I = keyword_faiss.search(query_emb, k)

    # Get matched keywords
    matched_keywords = [keyword_list[i] for i in I[0]]

    # Collect unique object IDs for matched keywords
    matched_ids = set()
    for kw in matched_keywords:
        matched_ids.update(keyword_map.get(kw, []))

    # Fetch the objects from Redis
    results = []
    for obj_id in matched_ids:
        obj = r.hgetall(f"{obj_id}")
        if obj:
            # Make sure each object includes an 'id' field for the frontend
            results.append({
                "id": obj.get("id", obj_id),
                "title": obj.get("title", ""),
                "description": obj.get("model_base", ""),
                "author": obj.get("author", ""),
                "origin": obj.get("origin", ""),
                "model_base": obj.get("model_base", ""),
                "singer": obj.get("singer", ""),
                "img_path": obj.get("img_path", "default.jpeg"),
                "fondo": obj.get("fondo", ""),
                "date": obj.get("date", ""),
                "image": obj.get("image", ""),
                "coordinates": obj.get("coordinates", "")
            })

    return jsonify({
        "matched_keywords": matched_keywords,
        "results": results[:10]  # limit to 10 results
    })

# ---------------------------------------------------------
# RUN APP
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5030)
