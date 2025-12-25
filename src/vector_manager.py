import os
import sys
from pathlib import Path
import pandas as pd

from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_PATH, VECTOR_STORE_PATH, EMBEDDING_MODEL, logger


class VectorManager:
    def __init__(
        self, 
        vector_path: str = None, 
        embedding_model: str = None
    ):

        self.vector_path = vector_path or VECTOR_STORE_PATH
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.data_path = DATA_PATH
        
        logger.info(f"VectorManager initialized:")
        logger.info(f"  Vector store path: {self.vector_path}")
        logger.info(f"  Embedding model: {self.embedding_model}")
        logger.info(f"  Data path: {self.data_path}")
        
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vectorstore = None

    def build_or_load(self) -> FAISS:
        if os.path.exists(self.vector_path):
            logger.info(f"Loading existing FAISS index from {self.vector_path}")
            try:
                self.vectorstore = FAISS.load_local(
                    self.vector_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS index loaded successfully")
                
                return self.vectorstore
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                logger.warning("Rebuilding index from CSV data...")
        
        logger.info(f"Building FAISS index from CSV data in {self.data_path}")
        self.vectorstore = self._build_from_csv()
        return self.vectorstore

    def _build_from_csv(self) -> FAISS:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data folder not found: {self.data_path}")
        
        csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")
        
        logger.info(f"Found CSV files: {csv_files}")
        
        dfs = []
        for csv_file in csv_files:
            csv_path = os.path.join(self.data_path, csv_file)
            logger.info(f"Loading {csv_file}...")
            df = pd.read_csv(csv_path)
            dfs.append(df)
            logger.info(f"  Loaded {len(df)} rows")
        
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total rows: {len(df)}")
    
        df = df.drop_duplicates(subset=['Title'], keep='first').reset_index(drop=True)
        logger.info(f"After removing duplicates: {len(df)} rows")
        
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        
        logger.info("Creating document chunks...")
        for idx, row in df.iterrows():
            if idx % 500 == 0:
                logger.info(f"  Processed {idx}/{len(df)} movies")
            
            plot_text = str(row.get('Plot', ''))
            if not plot_text or plot_text.lower() == 'nan':
                continue
            
            chunks = text_splitter.split_text(plot_text)
            
            for chunk in chunks:
                # Build page_content with metadata embedded (so LLM sees it in context)
                title = str(row.get('Title', 'Unknown'))
                
                release_year = 'Unknown'
                if 'Release Year' in row:
                    try:
                        release_year = str(int(row['Release Year']))
                    except (ValueError, TypeError):
                        release_year = str(row.get('Release Year', 'Unknown'))
                
                genre = str(row.get('Genre', 'Unknown'))
                director = str(row.get('Director', 'Unknown'))
                cast = str(row.get('Cast', 'Unknown'))
                
                page_content = f"""Title: {title}
Release Year: {release_year}
Genre: {genre}
Director: {director}
Cast: {cast}
Plot: {chunk}"""
                
                # Also store metadata separately for filtering/search purposes
                metadata = {
                    'title': title,
                    'genre': genre,
                    'director': director,
                    'cast': cast
                }

                try:
                    metadata['release_year'] = int(release_year) if release_year != 'Unknown' else None
                except (ValueError, TypeError):
                    metadata['release_year'] = None
                
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)
        
        logger.info(f"Created {len(documents)} document chunks")
        
        # Build FAISS index
        logger.info("Building FAISS index (this may take a few minutes)...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        os.makedirs(self.vector_path, exist_ok=True)
        logger.info(f"Saving FAISS index to {self.vector_path}")
        self.vectorstore.save_local(self.vector_path)
        logger.info("FAISS index saved successfully")
        
        return self.vectorstore

    def get_vectorstore(self) -> FAISS:
        if self.vectorstore is None:
            self.build_or_load()
        return self.vectorstore