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


def initialize_llm():
    global llm
    if llm is None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        llm = ChatGroq(
            temperature=0.7,  # Higher temperature for more creative responses
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile"
        )


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
        ("system", "You are a helpful AI assistant. Provide concise and accurate responses."),
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
        # Clear existing chains and memory first
        qa_chain = None
        chat_chain = None
        memory = None

        filename = file.filename.lower()
        ext = os.path.splitext(filename)[1]
        allowed_extensions = [".txt", ".pdf", ".docx"]

        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {allowed_extensions}"
            )

        file_bytes = await file.read()

        # Extract text based on file type
        if ext == ".txt":
            file_contents = file_bytes.decode("utf-8")
        elif ext == ".pdf":
            from io import BytesIO
            pdf_buffer = BytesIO(file_bytes)
            file_contents = extract_text_from_pdf(pdf_buffer)
        elif ext == ".docx":
            file_contents = extract_text_from_docx(file_bytes)

        # Initialize LLM if not already initialized
        if llm is None:
            initialize_llm()

        # Initialize new memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = splitter.split_text(file_contents)

        if not texts:
            raise HTTPException(
                status_code=400,
                detail="No text content could be extracted from the file"
            )

        # Create embeddings and vector store
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(texts, embedding_model)

        # Create QA chain with proper memory initialization
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            ),
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template="""Use the following pieces of context to answer the question. Base your answer ONLY on the provided context. If you don't find the answer in the context, say "I don't find this information in the provided document."

                    Context: {context}

                    Question: {question}

                    Answer: """,
                    input_variables=["context", "question"]
                )
            }
        )

        # Disable chat_chain when document mode is active
        chat_chain = None

        return {"message": "File processed and QA chain is ready", "mode": "document"}

    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


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
    global qa_chain, chat_chain, llm

    try:
        if not llm:
            initialize_llm()

        # Check if we're in document mode
        if qa_chain:
            try:
                # Use document QA chain
                result = qa_chain.invoke({
                    "question": chat_request.question,
                    "chat_history": memory.chat_memory.messages if memory else []
                })

                answer = result.get("answer", "")
                if "don't find this information" in answer.lower():
                    realtime_info = get_realtime_search_results(
                        chat_request.question)
                    if realtime_info and realtime_info != "Unable to fetch real-time information at the moment.":
                        answer += f"\n\nHowever, here is some relevant information from real-time sources:\n{realtime_info}"

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
            # Initialize general chat if no document is loaded
            if chat_chain is None:
                initialize_general_chat()

            # Limit chat history to last 5 messages before processing
            if memory and len(memory.chat_memory.messages) > 10:
                memory.chat_memory.messages = memory.chat_memory.messages[-5:]

            # Include chat history context in the prompt
            chat_history = memory.chat_memory.messages if memory else []
            realtime_info = get_realtime_search_results(chat_request.question)

            result = chat_chain.invoke({
                "input": f"Previous context: {chat_history}\n\nCurrent question: {chat_request.question}\n\nAvailable real-time information: {realtime_info}"
            })

            return {
                "answer": result["text"],
                "mode": "chat",
                "realtime_search": True
            }

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
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
