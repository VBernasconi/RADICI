import pandas as pd
import numpy as np
import faiss
import ast

# ======= Load and Parse Embeddings =======
def load_embeddings(csv_path):
    df = pd.read_csv(csv_path)

    def parse_embedding(emb_str):
        return np.array(ast.literal_eval(emb_str), dtype=np.float32)

    df["parsed_embedding"] = df["embeddings"].apply(parse_embedding)
    embedding_matrix = np.stack(df["parsed_embedding"].values)
    return df, embedding_matrix

# ======= Build FAISS Index =======
def build_faiss_index(embedding_matrix):
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    return index

# ======= Search Function =======
def search(index, df, query_vec, k=5):
    print("IN SEARCH FUNCTION")
    query_vec = np.array(query_vec, dtype=np.float32).reshape(1, -1)
    D, I = index.search(query_vec, k)
    return df.iloc[I[0]][["title", "author", "date", "img_path", "url", "id", "archive"]]

def find_similar_images(index, df, query_vec, top_k=5):#image_id, top_k=5):
    try:
        embedding = np.array(query_vec, dtype=np.float32).reshape(1, -1)
        k = min(top_k, index.ntotal)
        D, I = index.search(embedding, k)
        
        return df.iloc[I[0]][["title", "author", "date", "img_path", "url", "id", "archive"]]
    
    except Exception as e:
        print("ERROR")
        print(str(e))

    return "NULL"


# ======= Example Usage =======
if __name__ == "__main__":
    # Load embeddings from CSV
    df, embedding_matrix = load_embeddings("dataset_with_embeddings.csv")

    # Build the index
    index = build_faiss_index(embedding_matrix)

    # Example: use the first embedding as a query
    example_query = df.iloc[0]["parsed_embedding"]
    results = search(index, df, example_query, k=5)

    print("\nTop 5 similar items:\n")
    print(results.to_string(index=False))
