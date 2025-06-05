
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from vector_db import preprocess_data, build_vector_store

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class MovieBot:
    def __init__(self):
        embedding_model = "all-MiniLM-L6-v2"
        model_name = "gemma2:9b"
        top_k = 2
        score_threshold = 0.8       # at least 80% similarity between query and retrieved doc
        vector_db_dir = "faiss_movie_index"
        data_path = "vishnupriyavr/wiki-movie-plots-with-summaries" # on HuggingFace

        if os.path.exists(vector_db_dir):
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            print("Loading existing FAISS index...")
            vector_db = FAISS.load_local(vector_db_dir, embeddings, allow_dangerous_deserialization=True)
        else:
            df = preprocess_data(data_path)
            vector_db = build_vector_store(df, embedding_model, vector_db_dir)

        retriever = vector_db.as_retriever(search_kwargs={"k": top_k, "score_threshold": score_threshold})
        llm = OllamaLLM(model = model_name, temperature = 0)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

        custom_prompt = PromptTemplate(
            input_variables=["chat_history", "question", "context"],
            template=
            """System:
            You are a helpful movie assistant. Answer questions using ONLY the provided Context from movie plots and metadata.
            If you don't know the answer, say so. Keep answers concise. Include relevant movie titles and years.
            If the question is about something other than movies, politely remind that you are a movie assistant and you can only help with this.
            Pretend you do not have access to the provided Context and treat it as your own knowledge.
            Do not mention the Context in your answer.

            Context:
            {context}

            Current conversation:
            {chat_history}

            Human:
            {question}

            """
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm= llm,
            retriever= retriever,
            memory= memory,
            return_source_documents= True,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            verbose= False
        )

    def chat(self, query):
        response = self.qa_chain({"question": query})
        return response["answer"]
