# app_redis.py

import sys
from flask import Flask, request, jsonify, send_from_directory
from flask import session, redirect, url_for, render_template, jsonify
from flask_cors import CORS
import numpy as np
import redis
import json
import os
import re
import faiss
import nltk
import hashlib
import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from datetime import timedelta

import numpy.core.numeric as numeric
sys.modules['numpy._core.numeric'] = numeric

# ---------------------------------------------------------
# Redis + Flask Setup
# ---------------------------------------------------------
r = redis.Redis(host='192.168.249.170', port=6379, db=10, decode_responses=True)
r_users = redis.Redis(host='192.168.249.170', port=6379, db=12, decode_responses=True)
app = Flask(__name__)
# CORS(app, supports_credentials=True, origins=["http://localhost:8000"])

# app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change_this_secret")  # session secret
# app.permanent_session_lifetime = datetime.timedelta(days=7)  # optional: session lifetime

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "secret_key")
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

# REQUIRED for cookies to work CROSS-ORIGIN
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,  # set to True if using HTTPS
    SESSION_COOKIE_HTTPONLY=True
)

CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {"origins": [
        "http://localhost:8000",
        "http://192.168.249.170:8000",
        "http://127.0.0.1:8000"
    ]}}
)

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = "/home/valentine.bernasconi/RADICI/downloaded_images"#os.path.join(BASE_DIR, "downloaded_images")
print(IMAGES_DIR)

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

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

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

    save_path = os.path.join(IMAGES_DIR, filename)
    img.save(save_path)

    # Frontend will store this directly into Redis
    return jsonify({
        "filename": filename,
        "img_path": f"/images/{filename}"
    }), 200

@app.route("/add_object", methods=["POST"])
@login_required
def add_object():
    global keyword_map, keyword_list, keyword_embeddings, keyword_faiss

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Assign numeric ID
    obj_id = str(get_next_object_id())

    # Full container path for the image
    filename = data.get("img_path", "").replace("/images/", "")
    full_img_path = f"/images/{filename}" if filename else "/images/default.jpeg"


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
# HANDLE ROUTES
# ---------------------------------------------------------
@app.route("/")
@app.route("/esplorare")
def esplorare():
    return render_template("grid.html")

@app.route("/documentarsi")
def documentarsi():
    return render_template("documentarsi.html")

@app.route("/sperimentare")
def sperimentare():
    return render_template("sperimentare.html")

@app.route("/about")
def about():
    return render_template("about.html")
# ---------------------------------------------------------
# IMAGE ROUTES
# ---------------------------------------------------------
@app.route('/images/<filename>')
def get_image(filename):
    if not filename or filename == "default":
        filename = "default.jpeg"

    img_path = os.path.join(IMAGES_DIR, filename)
    if os.path.exists(img_path):
        return send_from_directory(IMAGES_DIR, filename)

    # fallback to default
    default_path = os.path.join(IMAGES_DIR, "default.jpeg")
    return send_from_directory(IMAGES_DIR, "default.jpeg")

@app.route('/images/')
def get_default_image():
    return send_from_directory(IMAGES_DIR, "default.jpeg")

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
# LOGIN
# ---------------------------------------------------------
@app.route("/user/status", methods=["GET"])
def user_status():
    if "user" in session:
        return jsonify({"logged_in": True, "username": session["user"]})
    return jsonify({"logged_in": False})

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    email = data.get("email", "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    # Check if user exists
    if r_users.exists(f"user:{username}:data"):
        return jsonify({"error": "Username already exists"}), 400

    # Hash password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    # Store user
    r_users.hset(f"user:{username}:data", mapping={
        "password": hashed_password,
        "email": email,
        "created_at": str(datetime.datetime.utcnow())
    })

    return jsonify({"success": True, "message": "User registered successfully"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user_data = r_users.hgetall(f"user:{username}:data")
    if not user_data:
        return jsonify({"error": "Invalid username or password"}), 401

    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    if hashed_password != user_data.get("password"):
        return jsonify({"error": "Invalid username or password"}), 401

    # Login success
    session.permanent = True  # <— this makes the session last as per PERMANENT_SESSION_LIFETIME
    session["user"] = username
    session.permanent = True  # optional, respects permanent_session_lifetime
    return jsonify({"success": True, "message": "Logged in successfully"})


@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user", None)
    return jsonify({"success": True, "message": "Logged out successfully"})


@app.route("/user/collections", methods=["GET"])
@login_required
def list_collections():
    username = session["user"]
    collection_names = r_users.smembers(f"user:{username}:collections") or set()
    collections = []

    for name in collection_names:
        print(f"user:{username}:collection:{name}")
        # Get object IDs in this collection
        object_ids = r_users.smembers(f"user:{username}:collection:{name}") or set()
        print(object_ids)
        objects = []
        
        for obj_id in object_ids:
            #obj = r.hgetall(f"{obj_id}")
            # Find the object with this image_id in redis_objects
            obj = next((o for o in redis_objects if o.get("id") == obj_id), None)
            print(obj)
            if obj:
                objects.append({
                    "id": obj.get("id", obj_id),
                    "title": obj.get("title", ""),
                    "image": obj.get("image", ""),
                    "description": obj.get("model_base", "")
                })

        collections.append({
            "name": name,
            "objects": objects
        })

    return jsonify({"success": True, "collections": collections})


@app.route("/user/collections/create", methods=["POST"])
@login_required
def create_collection():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    username = session["user"]
    data = request.get_json()
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Collection name required"}), 400

    # Add collection name to user's collections set
    r_users.sadd(f"user:{username}:collections", name)
    return jsonify({"success": True, "message": f"Collection '{name}' created."})

@app.route("/user/collections/add", methods=["POST"])
@login_required
def add_to_collection():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    username = session["user"]
    data = request.get_json()
    obj_id = str(data.get("object_id"))
    collection_name = data.get("collection_name", "").strip()
    if not obj_id or not collection_name:
        return jsonify({"error": "Missing object_id or collection_name"}), 400

    # Make sure collection exists
    if not r_users.sismember(f"user:{username}:collections", collection_name):
        return jsonify({"error": "Collection does not exist"}), 404

    # Add object to collection (as Redis set)
    r_users.sadd(f"user:{username}:collection:{collection_name}", obj_id)
    return jsonify({"success": True, "message": f"Object {obj_id} added to '{collection_name}'"})

@app.route("/api/collection/<collection_name>")
@login_required
def get_collection(collection_name):
    username = session["user"]

    key = f"user:{username}:collection:{collection_name}"

    if not r_users.exists(key):
        return jsonify({"success": False, "error": "Collection not found"}), 404

    object_ids = r_users.smembers(key)
    objects = []

    for obj_id in object_ids:
        obj = next((o for o in redis_objects if o.get("id") == obj_id), None)
        if obj:
            objects.append({
                "id": obj.get("id"),
                "title": obj.get("title", ""),
                "img_path": obj.get("img_path", ""),
                "image": obj.get("image", ""),
                "description": obj.get("model_base", "")
            })

    return jsonify({
        "success": True,
        "collection": collection_name,
        "objects": objects
    })

@app.route("/collection/<collection_name>")
@login_required
def collection_page(collection_name):
    username = session["user"]

    # check if collection exists for this user
    if not r_users.sismember(f"user:{username}:collections", collection_name):
        return "Collection not found or unauthorized", 404

    return render_template(
        "collection.html",
        collection_name=collection_name
    )

# ---------------------------------------------------------
# RUN APP
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5030)
