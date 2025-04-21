# AI Chatbot with Document Processing

<div align="center">
  <video width="800" autoplay loop muted>
    <source src="https://firebasestorage.googleapis.com/v0/b/portfolio-c5c0a.appspot.com/o/AI%20Assistant.mp4?alt=media&token=546923f1-2556-4e4d-816c-632ce9ef3519" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

A full-stack AI chatbot application that combines document analysis and real-time web search capabilities. Built with FastAPI, React, and powered by Groq's LLM API.

## Features

- 📄 **Document Processing**: Upload and analyze PDF, DOCX, and TXT files
- 💬 **Dual Chat Modes**:
  - Document Q&A: Ask questions about uploaded documents
  - General Chat: Get responses enhanced with real-time web search
- 🔍 **Semantic Search**: Powered by FAISS and HuggingFace embeddings
- 🌐 **Real-time Information**: Integration with DuckDuckGo search
- 🚀 **Modern Stack**: FastAPI backend + React frontend

## Tech Stack

### Backend

- FastAPI
- LangChain
- Groq LLM (llama-3.3-70b-versatile)
- FAISS Vector Store
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- DuckDuckGo Search API

### Frontend

- React (Create React App)
- Modern UI/UX design

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 14+
- Groq API key

### Backend Setup

1. Navigate to server directory:

```bash
cd server
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create `.env` file:

```
GROQ_API_KEY=your-groq-api-key-here
ALLOWED_ORIGINS=http://localhost:3000,your-production-domain
```

5. Run the server:

```bash
uvicorn main:app --reload
```

### Frontend Setup

1. Navigate to client directory:

```bash
cd client
```

2. Install dependencies:

```bash
npm install
```

3. Start development server:

```bash
npm start
```

## Deployment

### Backend

- Deployed on Render
- Configuration in `render.yaml`

### Frontend

- Deployed on Vercel
- Production build: `npm run build`

## Features in Detail

### Document Processing

- File size limit: 5MB
- Supported formats: PDF, DOCX, TXT
- Efficient chunking for large documents
- Semantic search capabilities

### Chat Modes

1. **Document Mode**

   - Upload and analyze documents
   - Ask questions about document content
   - Semantic search for relevant answers

2. **General Chat Mode**
   - Natural conversation
   - Real-time web search integration
   - Enhanced responses with current information

### Memory Management

- Conversation history tracking
- Efficient memory cleanup
- Batch processing for large documents

## API Endpoints

- `/chat`: Process chat messages
- `/upload`: Handle document uploads
- `/mode`: Check current chat mode
- `/clear`: Reset chat context
- `/cleanup`: Memory management

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
