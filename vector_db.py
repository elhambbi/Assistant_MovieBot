from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores.utils import DistanceStrategy
from datasets import load_dataset
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def preprocess_data(data_path):

    ds = load_dataset(data_path)
    df = pd.DataFrame(ds["train"])

    df = df.drop_duplicates(subset='Title', keep='first')
    df = df[["Release Year", "Title", "Director", "Cast", "Genre", "Plot"]].dropna()

    df["Release Year"] = pd.to_numeric(df["Release Year"], errors='coerce')
    df = df.dropna(subset=["Release Year"])
    df["Release Year"] = df["Release Year"].astype(int)

    mask = df.map(lambda x: str(x).lower().strip() == 'unknown').any(axis=1)
    df = df[~mask]
    df = df[df["Release Year"] >= 1990].reset_index(drop= True)

    print(f"{df.shape[0]} movie documents are used as RAG docs")
    return df

def build_vector_store(df, embedding_model, vector_db_dir):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= 512,
        chunk_overlap= 50
    )
    
    texts = []
    for _, row in df.iterrows():
        chunks = text_splitter.split_text(row["Plot"])
        for chunk in chunks:
            texts.append({
                "title": row["Title"],
                "release_year": row["Release Year"],
                "genre": row["Genre"],
                "director": row["Director"],
                "cast": row["Cast"],
                "text": chunk
            })

    embeddings = HuggingFaceEmbeddings(model_name= embedding_model)
    docs = [
        Document(
            page_content=(
                f"Title: {item['title']}\n"
                f"Release Year: {item['release_year']}\n" 
                f"Genre: {item['genre']}\n"
                f"Director: {item['director']}\n"
                f"Cast: {item['cast']}\n"
                f"Plot: {item['text']}"
            ),
        ) for item in texts
    ]
    print(f"Creating FAISS index with {len(docs)} chunked documents...")
    start = time.time()
    vector_db = FAISS.from_documents(docs, embeddings, distance_strategy= DistanceStrategy.MAX_INNER_PRODUCT)  # equivalent to cosine similarity
    vector_db.save_local(vector_db_dir)
    print(f"  Index created in {round((time.time() - start)/60, 2)} minutes.\n")
    
    return vector_db