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

embedding_model = "all-MiniLM-L6-v2"
model_name = "gemma2:9b"
top_k = 2
vector_db_dir = "faiss_movie_index"
data_dir = "data/wiki_movie_plots_deduped.csv"

def main():
    df = preprocess_data(data_dir)
    if os.path.exists(vector_db_dir):
        embeddings = HuggingFaceEmbeddings(model_name= embedding_model)
        # Example embedding vector - to test if the embeddings are already normalized for cosine similarity
        # embedding = embeddings.embed_query("this is a test...")
        #print(np.linalg.norm(embedding))  # Should be ~1.0 (normalized)
        print("Loading existing FAISS index...")
        vector_db = FAISS.load_local(vector_db_dir, embeddings, allow_dangerous_deserialization=True) # to allow loading pickle files
    else:
        vector_db = build_vector_store(df, embedding_model, vector_db_dir)

  
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    llm = OllamaLLM(model= model_name, temperature= 0)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    
    custom_prompt = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template=
        """<start_of_turn>system
        You are a helpful movie expert assistant. Answer questions using ONLY the provided context from movie plots and metadata.
        If you don't know the answer, say so. Keep answers concise. Include relevant movie titles and years.
            
        Context:
        {context}
            
        Current conversation:
        {chat_history}<end_of_turn>
            
        <start_of_turn>user
        {question}<end_of_turn>
            
        <start_of_turn>model
        """)
    
    # conversational RAG chain with custom prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        verbose= False  # True to see prompts and RAG chain steps
    )
    print("\nMovieBot: Hi! I'm your movie expert. Ask me about movies or their plots. Type 'exit' to quit...")
    
    while True:
        query = input("\n>> You: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        response = qa_chain({"question": query})
        answer = response["answer"]
        print(f"\n>> MovieBot: {answer}")

if __name__ == "__main__":
    main()
