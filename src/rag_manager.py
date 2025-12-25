import sys
from pathlib import Path

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import logger


class PromptLoggingCallback(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        if 'question' in inputs:
            user_question = inputs.get('question', '')
            logger.info("\n" + "=" * 80)
            logger.info(f"USER QUESTION: {user_question}")
    
    def on_retriever_end(self, documents, **kwargs):
        logger.info(f"\nRETRIEVED DOCUMENTS ({len(documents)} documents):")
        logger.info("-" * 80)
        for idx, doc in enumerate(documents, 1):
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            logger.info(f"[{idx}] {content_preview}")
        logger.info("-" * 80)


class RAGManager:
    def __init__(self, llm, retriever, memory: ConversationBufferMemory = None):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self._chain = None
        
        logger.info("RAGManager initialized with LLM and retriever")
        
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "question", "context"],
            template="""Use the following pieces of context (retrieved movie information) to answer the question. Each movie entry includes Title, Release Year, Genre, Director, Cast, and Plot information.

IMPORTANT: 
- Use the Title, Genre, Director, Cast, and other metadata from the Context to answer questions
- If asked about a movie's title, genre, director, or cast, extract it from the Context
- If you don't know the answer based on the Context, say so clearly
- Do not make up information
- Do not mention that you are using the Context - treat retrieved information as your own knowledge.
- Give short and friendly answers, use emojis when appropriate.

Context (retrieved movie information):
{context}

Previous conversation:
{chat_history}

Question: {question}

Answer based on the Context above. Include specific details like movie titles, years, genres, directors, and cast when relevant:"""
        )

    def build_chain(self, memory: ConversationBufferMemory = None, chain_type: str = "stuff"):
        """
        Build a ConversationalRetrievalChain that includes:
        - Retriever for fetching relevant docs
        - Memory for conversation history
        - Custom prompt for movie context
        - Callback to log final prompt sent to LLM
        
        Args:
            memory: Optional ConversationBufferMemory to use. If None, uses instance memory.
        """
        # Use provided memory (per-session) or fall back to instance memory
        # This allows each session to have its own conversation history, ensuring
        # users don't see each other's conversations
        chain_memory = memory or self.memory
        
        logger.info("Building ConversationalRetrievalChain...")
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=chain_memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            callbacks=[PromptLoggingCallback()],
            verbose=True
        )
        logger.info("RAG chain built successfully")
        return chain

    def get_memory(self):
        memory_vars = self.memory.load_memory_variables({})
        history_length = len(memory_vars.get("chat_history", []))
        logger.debug(f"Memory accessed: {history_length} messages in history")
        return self.memory