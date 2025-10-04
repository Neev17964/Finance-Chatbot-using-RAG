from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import List, Dict
import uuid
import os

# For Render deployment - use PORT environment variable
port = int(os.environ.get("PORT", 8000))

load_dotenv()

# 🎯 Initialize FastAPI
app = FastAPI(
    title="Finance Chatbot API",
    description="AI-powered Indian Income Tax Assistant",
    version="1.0.0"
)

# 🌐 Enable CORS (so web developer can call from browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📝 Request/Response Models
class ChatRequest(BaseModel):
    question: str
    session_id: str = None  # Optional: for maintaining user sessions

class ChatResponse(BaseModel):
    answer: str
    session_id: str

# 💾 Session storage (in-memory, use Redis for production)
sessions: Dict[str, List[tuple]] = {}

# 🚀 Initialize RAG components (load once at startup)
print("🔄 Loading models and vector store...")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1000
)

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Load knowledge base
with open('data.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()

headers_to_split_on = [
    ("#", "Header_1"),
    ("##", "Header_2"),
    ("###", "Header_3"),
    ("####", "Header_4"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, 
    strip_headers=False
)
header_split_data = markdown_splitter.split_text(text_data)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)

documents = []
for doc in header_split_data:
    if len(doc.page_content) > 800:
        for sub_chunk in recursive_splitter.split_text(doc.page_content):
            documents.append(Document(page_content=sub_chunk, metadata=doc.metadata))
    else:
        documents.append(Document(page_content=doc.page_content, metadata=doc.metadata))

vector_store = Chroma(persist_directory="Database", embedding_function=embedding)

prompt_template = """You are an expert Indian Income Tax consultant. Answer using ONLY the provided context.

Context:
{context}

Question: {question}

Instructions:
- Provide clear, readable answers (8-12 lines)
- Use line breaks between different points for better readability
- Add relevant emojis (💰📊✅❌) to make it engaging
- Mention key deductions with section numbers
- End with the final result clearly stated
- Use simple language, avoid jargon

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False
)

print("✅ API Ready!")

# 🔗 API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Finance Chatbot API is running! 💼🤖",
        "endpoints": {
            "POST /chat": "Send questions to the chatbot",
            "DELETE /session/{session_id}": "Clear conversation history",
            "GET /health": "Check API health"
        }
    }

@app.get("/health")
async def health():
    """Health check for monitoring"""
    return {"status": "healthy", "model": "llama-3.3-70b-versatile"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chatbot endpoint
    
    Example request:
    {
        "question": "I earn ₹10 lakh. What's my tax?",
        "session_id": "user123"  // optional
    }
    """
    try:
        # Generate or use existing session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get conversation history for this session
        conversation_history = sessions.get(session_id, [])
        
        # Build history context
        history_context = ""
        if conversation_history:
            history_context = "\n\nPrevious Conversation:\n"
            for i, (q, a) in enumerate(conversation_history[-5:], 1):
                history_context += f"Q{i}: {q}\nA{i}: {a}\n\n"
        
        # Add history to query
        query_with_history = f"{history_context}Current Question: {request.question}"
        
        # Get answer from RAG
        result = rag.invoke(query_with_history)
        answer = result["result"].strip()
        
        # Store in session history
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append((request.question, answer))
        
        # Keep only last 5 exchanges
        if len(sessions[session_id]) > 5:
            sessions[session_id].pop(0)
        
        return ChatResponse(answer=answer, session_id=session_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared successfully"}
    return {"message": "Session not found"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions (for debugging)"""
    return {
        "active_sessions": len(sessions),
        "session_ids": list(sessions.keys())
    }