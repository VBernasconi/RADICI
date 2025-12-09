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

# Default fields for every object
DEFAULT_FIELDS = {
    "image": "",
    "url": "",
    "tags": "",
    "model_base": "",
    "place": "",
    "date": "",
    "fondo": "",
    "coordinates": "[]",
    "img_path": "",
    "type": "",
    "archive": "",
    "origin": "",
    "embeddings": "",
    "title": "",
    "id": "",
    "singer": "",
    "author": "",
    "model_lora": ""
}

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

        # Skip the cache
        if key == "place_coordinates_cache":
            continue
            
        obj = r.hgetall(key)
        if not obj:
            continue
        #obj_id = obj["id"]  # keep as string
        obj_id = obj.get("id")
        if not obj_id:
            continue  # skip entries without an id


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

    # Frontend will store this directly into Redis
    return jsonify({
        "filename": filename,
        "img_path": f"/images/{filename}"
    }), 200

@app.route("/add_object", methods=["POST"])
def add_object():
    global keyword_map, keyword_list, keyword_embeddings, keyword_faiss

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Assign numeric ID
    obj_id = str(get_next_object_id())

    # Full container path for the image
    filename = data.get("img_path", "").replace("/images/", "")
    full_img_path = f"/app/data/downloaded_images/{filename}" if filename else ""

    # ----------------------------
    # FIELD MAPPING (frontend → Redis)
    # ----------------------------
    mapped = {
        "id": obj_id,
        "title": data.get("title", ""),
        "model_base": data.get("description", ""),     # description → model_base
        "author": data.get("author", ""),
        "singer": data.get("singer", ""),
        "date": data.get("date", ""),
        "fondo": data.get("fondo", ""),
        "archive": data.get("archive", ""),
        "place": data.get("place", ""),
        "type": data.get("type", ""),
        "origin": data.get("origin", ""),
        "url": data.get("url", ""),
        "model_lora": data.get("model_lora", ""),
        "coordinates": json.dumps(data.get("coordinates", [])),
        "tags": json.dumps(data.get("tags", [])),
        "img_path": full_img_path,
        "image": filename,  # keep just the filename
        "embeddings": ""  # FILLED LATER BY YOUR EMBEDDING PIPELINE
    }

    # --------------------------------
    # Save object into Redis
    # --------------------------------
    r.hset(obj_id, mapping=mapped)

    # --------------------------------
    # Build text for keyword index
    # --------------------------------
    text = " ".join([
        mapped["title"],
        mapped["model_base"],
        mapped["author"],
        mapped["place"],
        mapped["origin"],
        mapped["singer"],
        mapped["type"]
    ])

    # Extract keywords
    new_keywords = extract_keywords(text)

    # --------------------------------
    # Update keyword FAISS index
    # --------------------------------
    for kw in new_keywords:
        if kw not in keyword_map:
            keyword_map[kw] = []
            keyword_list.append(kw)

            kw_emb = MODEL.encode([kw]).astype(np.float32)

            if keyword_embeddings is None or len(keyword_embeddings) == 0:
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
# ADD OBJECT
# ---------------------------------------------------------
# @app.route("/add_object", methods=["POST"])
# def add_object():
#     global keyword_map, keyword_list, keyword_embeddings, keyword_faiss

#     data = request.get_json()
#     if not data:
#         return jsonify({"error": "No data provided"}), 400

#     obj_id = str(get_next_object_id())
#     data["id"] = obj_id

#     # ensure all default fields exist
#     for f, default_val in DEFAULT_FIELDS.items():
#         if f not in data:
#             data[f] = default_val

#     # --- FIX: Convert tags list -> JSON string ---
#     if isinstance(data.get("tags"), list):
#         data["tags"] = json.dumps(data["tags"])
    
#     if isinstance(data.get("coordinates"), list):
#         data["coordinates"] = json.dumps(data["coordinates"])

#     # --- FIX: ensure img_path is stored, even if empty ---
#     if not data.get("img_path"):
#         data["img_path"] = ""

#     # --- Keyword embedding ---
#     text = " ".join([
#         data.get("title", ""),
#         data.get("description", ""),
#         data.get("author", ""),
#         data.get("place", ""),
#         data.get("origin", ""),
#         data.get("model_base", ""),
#         data.get("singer", "")
#     ])
#     text_emb = MODEL.encode([text]).astype(np.float32)[0]
#     data["text_embedding"] = json.dumps(text_emb.tolist())
    
#     # Save to Redis
#     r.hset(obj_id, mapping=data)

#     # Update keyword FAISS index
#     new_keywords = extract_keywords(text)
#     for kw in new_keywords:
#         if kw not in keyword_map:
#             keyword_map[kw] = []
#             keyword_list.append(kw)
#             kw_emb = MODEL.encode([kw]).astype(np.float32)
#             if keyword_embeddings is None or len(keyword_embeddings) == 0:
#                 keyword_embeddings = kw_emb
#                 keyword_faiss = faiss.IndexFlatL2(kw_emb.shape[1])
#                 keyword_faiss.add(kw_emb)
#             else:
#                 keyword_embeddings = np.vstack([keyword_embeddings, kw_emb])
#                 keyword_faiss.add(kw_emb)
#         keyword_map[kw].append(obj_id)

#     return jsonify({"object_id": obj_id, "added_keywords": new_keywords}), 201

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
