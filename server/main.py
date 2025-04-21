import os
import docx
import PyPDF2
import logging
import time
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import gc
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
print("GROQ API Key present:", bool(os.getenv("GROQ_API_KEY")))

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://aichatbot-six-tau.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Global variables
llm = None
qa_chain = None
chat_chain = None
memory = None

# Add these constants at the top
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB limit
CHUNK_SIZE = 1000  # Reduced chunk size
CHUNK_OVERLAP = 100  # Reduced overlap
MAX_CHUNKS = 30  # Reduced max chunks


def initialize_llm():
    """Initialize the LLM with proper error handling"""
    global llm
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        llm = ChatGroq(
            temperature=0.7,
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile"
        )

        # Test if LLM is properly initialized
        if not llm:
            raise ValueError("LLM initialization failed")

    except Exception as e:
        logger.error(f"LLM initialization error: {str(e)}")
        raise


def initialize_general_chat():
    """Initialize the general chat chain without document context"""
    global chat_chain, llm, memory

    if llm is None:
        initialize_llm()

    # Create new memory instance with a limited number of messages
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    # Create chat prompt template with clear instructions
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a direct and efficient AI assistant. Provide brief, accurate responses without unnecessary explanations."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Initialize the chat chain with the prompt
    chat_chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        memory=memory,
        verbose=True
    )


def extract_text_from_pdf(file_contents: bytes) -> str:
    # Use PyPDF2 to extract text from PDF bytes.
    reader = PyPDF2.PdfReader(file_contents)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""  # Extract text from each page.
    return text


def extract_text_from_docx(file_contents: bytes) -> str:
    # Write the bytes to a temporary file for python-docx to open.
    # Alternatively, if you already have a temporary file wrapper method, use that.
    tmp_filename = "temp_doc.docx"
    with open(tmp_filename, "wb") as tmp_file:
        tmp_file.write(file_contents)
    doc = docx.Document(tmp_filename)
    text = "\n".join([para.text for para in doc.paragraphs])
    os.remove(tmp_filename)  # Clean up the temporary file.
    return text


def initialize_memory():
    global memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global qa_chain, chat_chain, memory, llm

    try:
        # First ensure LLM is initialized
        if llm is None:
            initialize_llm()

        # Initialize memory if not already done
        if memory is None:
            initialize_memory()

        # Stream file in chunks instead of loading entirely into memory
        file_size = 0
        text_content = ""
        chunk = await file.read(8192)
        while chunk:
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large")
            if file.filename.endswith('.txt'):
                text_content += chunk.decode('utf-8')
            chunk = await file.read(8192)

        # Use smaller chunks and batch processing
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        texts = splitter.split_text(text_content)

        if len(texts) > MAX_CHUNKS:
            texts = texts[:MAX_CHUNKS]

        # Process embeddings in smaller batches
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Create vector store with smaller batch size
        texts_batches = [texts[i:i + 16]
                         # Create batches of 16
                         for i in range(0, len(texts), 16)]
        vector_store = None

        for batch in texts_batches:
            batch_vectorstore = FAISS.from_texts(
                batch,
                embedding_model
            )

            if vector_store is None:
                vector_store = batch_vectorstore
            else:
                vector_store.merge_from(batch_vectorstore)

        # Make sure LLM is properly initialized before creating QA chain
        if not llm:
            raise ValueError("LLM initialization failed")

        # Configure QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"  # Specify the output key
            ),
            return_source_documents=True,
            verbose=False  # Reduce logging
        )

        return {"message": "File processed successfully", "mode": "document"}

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    question: str


def get_realtime_search_results(query: str, max_results: int = 3, max_retries: int = 3) -> str:
    """Get real-time search results from DuckDuckGo with retry mechanism"""
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    max_results=max_results,
                    backend="api"  # Explicitly use API backend
                ))

                if not results:
                    logger.info(f"No results found for query: {query}")
                    return "No real-time information found."

                # Format the results
                formatted_results = []
                for result in results:
                    try:
                        formatted_results.append(
                            f"Source: {result.get('link', 'No link')}\n"
                            f"{result.get('body', 'No content available')}"
                        )
                    except Exception as e:
                        logger.error(f"Error formatting result: {e}")
                        continue

                return "\n\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Search attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retrying
            continue

    return "Unable to fetch real-time information at the moment."


def get_current_mode() -> str:
    """Return the current mode of operation"""
    if qa_chain is not None:
        return "document"
    elif chat_chain is not None:
        return "chat"
    return "none"


@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    try:
        if not chat_request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        if not llm:
            initialize_llm()

        if qa_chain:
            # Document mode
            try:
                result = qa_chain.invoke({
                    "question": chat_request.question,
                    # Keep only last 4 messages
                    "chat_history": memory.chat_memory.messages[-4:] if memory else []
                })

                # Now we know it's specifically "answer"
                answer = result["answer"]
                source_docs = result.get("source_documents", [])

                # Only search real-time if needed
                if "don't find this information" in answer.lower():
                    realtime_info = get_realtime_search_results(
                        chat_request.question,
                        max_results=2  # Reduce number of results
                    )
                    if realtime_info and realtime_info != "Unable to fetch real-time information at the moment.":
                        answer += f"\n\nHowever, here is some relevant information from real-time sources:\n{realtime_info}"

                # Clear some memory
                gc.collect()

                return {
                    "answer": answer,
                    "mode": "document",
                    "realtime_search": "don't find this information" in answer.lower()
                }

            except Exception as e:
                logger.error(f"Error in QA chain: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail="Error processing document-based response"
                )

        else:
            # General chat mode
            if chat_chain is None:
                initialize_general_chat()

            realtime_info = get_realtime_search_results(
                chat_request.question,
                max_results=2  # Reduce number of results
            )

            prompt = f"""Question: {chat_request.question}

Real-time information: {realtime_info}

Please provide a concise answer incorporating both your knowledge and the real-time information when relevant."""

            result = chat_chain.invoke({"input": prompt})

            # Clear some memory
            gc.collect()

            return {
                "answer": result["text"],
                "mode": "chat",
                "realtime_search": True
            }

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>AI Chat Backend</title>
        </head>
        <body style="font-family: Arial, sans-serif; background-color: #f5f5f5; text-align: center; padding-top: 50px;">
            <h1>AI Chat Backend</h1>
            <p>Your FastAPI server with LangChain is up and running.</p>
        </body>
    </html>
    """
    return html_content

# Add a new endpoint to clear the current mode


@app.post("/clear")
async def clear_mode():
    """Clear all chat modes, memory, and document context"""
    global qa_chain, chat_chain, memory, llm
    qa_chain = None
    chat_chain = None
    memory = None
    llm = None
    # Also reset the LLM instance
    return {"message": "Cleared all chat modes, memory, and document context"}

# Add a new endpoint to check current mode


@app.get("/mode")
async def get_mode():
    """Return the current mode of operation"""
    if qa_chain is not None:
        return {"mode": "document"}
    elif chat_chain is not None:
        return {"mode": "chat"}
    return {"mode": "none"}

# Add memory cleanup endpoint


@app.post("/cleanup")
async def cleanup_memory():
    """Manually trigger memory cleanup"""
    global qa_chain, chat_chain, memory, llm
    qa_chain = None
    chat_chain = None
    memory = None
    llm = None
    gc.collect()
    return {"message": "Memory cleaned up"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "answer": "I apologize, but I encountered a temporary issue. Please try asking your question again.",
            "mode": "chat",
            "error": str(exc.detail)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "answer": "I apologize, but I encountered a temporary issue. Please try asking your question again.",
            "mode": "chat",
            "error": "Internal server error"
        }
    )
