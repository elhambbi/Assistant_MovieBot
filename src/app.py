import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from .vector_manager import VectorManager
from .llm_manager import LLMManager
from .rag_manager import RAGManager
from .memory import Memory
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from config.settings import VECTOR_STORE_PATH, EMBEDDING_MODEL, TOP_K_RETRIEVAL, logger


llm_manager: LLMManager = None
vector_manager: VectorManager = None
vectorstore = None
retriever = None
rag: RAGManager = None
memory: Memory = None
# Store LangChain ConversationBufferMemory per session for persistent conversation history
session_memories: dict[str, ConversationBufferMemory] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_manager, vector_manager, vectorstore, retriever, rag, memory, session_memories
    
    logger.info("Starting up MovieBot API...")
    
    llm_manager = LLMManager()
    vector_manager = VectorManager(
        vector_path=VECTOR_STORE_PATH,
        embedding_model=EMBEDDING_MODEL
    )
    
    # Load or build vectorstore (blocking, but only once at startup)
    logger.info("Loading/building vectorstore...")
    vectorstore = vector_manager.build_or_load()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
    
    memory = Memory()
    session_memories = {}
    rag = RAGManager(llm_manager.llm, retriever)
    
    logger.info("MovieBot API ready.")
    yield
    logger.info("Shutting down MovieBot API.")


app = FastAPI(
    title="Advanced MovieBot (LangChain + RAG)",
    description="Movie Q&A with retrieval-augmented generation and conversation history",
    lifespan=lifespan
)


class Query(BaseModel):
    question: str
    session_id: str = "default"


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    history: list = []


@app.get("/", response_class=HTMLResponse)
async def get_chat_page():
    html_path = os.path.join(os.path.dirname(__file__), "chat_ui.html")
    with open(html_path, "r") as f:
        html_content = f.read()
    return html_content


@app.post("/query", response_model=QueryResponse)
async def query_bot(query: Query) -> QueryResponse:
    """
    Main query endpoint. Supports both sync and async LLM backends.
    """
    if not rag or not retriever:
        logger.error("Service not initialized")
        return QueryResponse(
            answer="Service not initialized. Please try again.",
            session_id=query.session_id
        )
    
    logger.info(f"Processing query for session: {query.session_id}")
    logger.debug(f"Question: {query.question[:100]}...")
    
    # Get or create LangChain memory for this session (persists across queries)
    if query.session_id not in session_memories:
        session_memories[query.session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        logger.info(f"Created new LangChain memory for session: {query.session_id}")
    else:
        # Log existing conversation history before adding new message
        existing_memory_vars = session_memories[query.session_id].load_memory_variables({})
        existing_history = existing_memory_vars.get("chat_history", [])
        if existing_history:
            logger.info(f"\n--- Existing conversation history for session {query.session_id} ({len(existing_history)} messages) ---")
            for idx, msg in enumerate(existing_history, 1):
                if hasattr(msg, 'content'):
                    role = "USER" if isinstance(msg, HumanMessage) else "ASSISTANT"
                    content = msg.content
                    logger.info(f"  [{idx}] {role}: {content[:200]}{'...' if len(content) > 200 else ''}")
            logger.info("--- End of existing history ---\n")
    
    session_memory = session_memories[query.session_id]
    
    # Build the RAG chain with session-specific memory (memory persists between queries)
    logger.debug("Building RAG chain with session memory")
    chain = rag.build_chain(memory=session_memory)
    
    try:
        # LangChain chains handle async/sync internally based on the LLM
        # The chain automatically adds the assistant response to session_memory
        logger.debug(f"Invoking chain (backend: {llm_manager.backend})")
        result = chain.invoke({"question": query.question})
        answer = result["answer"]
        
        if "source_documents" in result:
            source_docs = result["source_documents"]
            logger.info(f"\nRETRIEVED DOCUMENTS FROM RESULT ({len(source_docs)} documents):")
            logger.info("-" * 80)
            for idx, doc in enumerate(source_docs, 1):
                page_content = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                logger.info(f"[{idx}] {page_content}")
                if metadata:
                    logger.info(f"     Metadata: {metadata}")
            logger.info("-" * 80)
        
        # Ensure assistant response is in memory (chain should add it automatically, but verify)
        # Check if the last message in memory is the assistant response
        memory_vars_check = session_memory.load_memory_variables({})
        chat_history_check = memory_vars_check.get("chat_history", [])
        if not chat_history_check or not isinstance(chat_history_check[-1], AIMessage):
            # If not automatically added, add it manually
            session_memory.chat_memory.add_ai_message(answer)
        
        logger.info(f"Query answered successfully for session {query.session_id}")
        logger.debug(f"Answer preview: {answer[:100]}...")
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        answer = f"Error processing query: {str(e)}"
        # Add error response to LangChain memory
        session_memory.chat_memory.add_ai_message(answer)
    
    # Sync LangChain memory to our Memory class for API response
    # Extract all messages from LangChain memory and store in simple format
    memory_vars = session_memory.load_memory_variables({})
    chat_history = memory_vars.get("chat_history", [])
    
    # Update our Memory class with the full conversation from LangChain
    if query.session_id not in memory.sessions:
        memory.sessions[query.session_id] = []
    
    # Clear and rebuild from LangChain memory (single source of truth)
    memory.sessions[query.session_id] = []
    for msg in chat_history:
        if hasattr(msg, 'content'):
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user" if "human" in msg.__class__.__name__.lower() else "assistant"
            memory.sessions[query.session_id].append({"role": role, "content": msg.content})
    
    history = memory.get_history(query.session_id)
    logger.info(f"Session {query.session_id} now has {len(history)} total messages")
    
    # Log full conversation history after processing
    logger.info("\n" + "=" * 80)
    logger.info(f"COMPLETE CONVERSATION HISTORY - Session: {query.session_id} ({len(history)} messages)")
    for idx, msg in enumerate(history, 1):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if len(content) > 1000:
            logger.info(f"[{idx}] {role}:")
            logger.info(f"     {content[:1000]}...")
            logger.info(f"     (... {len(content) - 1000} more characters)")
        else:
            # For shorter messages, show on one line or multiple lines if needed
            if '\n' in content or len(content) > 200:
                logger.info(f"[{idx}] {role}:")
                for line in content.split('\n'):
                    logger.info(f"     {line}")
            else:
                logger.info(f"[{idx}] {role}: {content}")
    logger.info("\n" + "=" * 80)
    
    return QueryResponse(
        answer=answer,
        session_id=query.session_id,
        history=history
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "Advanced MovieBot"}
