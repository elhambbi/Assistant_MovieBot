from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores.utils import DistanceStrategy
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def preprocess_data(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Download dataset and place in data_dir")
    df = pd.read_csv(data_dir)
    df = df.drop_duplicates(subset='Title', keep='first')
    df = df[["Release Year", "Title", "Director", "Cast", "Genre", "Plot"]].dropna().head(1000)
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
    vector_db = FAISS.from_documents(docs, embeddings, distance_strategy= DistanceStrategy.MAX_INNER_PRODUCT)  # equivalent to cosine similarity
    vector_db.save_local(vector_db_dir)
    
    return vector_db