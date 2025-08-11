# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Dependencies
**IMPORTANT: Always use `uv` for dependency management. Do not use `pip` directly.**

```bash
# Install dependencies
uv sync

# Add new dependencies
uv add <package-name>

# Run any Python commands through uv
uv run <command>
```

### Environment Setup
Required environment variable in `.env`:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) system** for querying course materials. The architecture follows a modular design with clear separation between document processing, vector search, and AI generation.

### Core RAG Flow
1. **Document Ingestion**: Course documents (txt, pdf, docx) in `/docs/` are processed into chunks
2. **Vector Storage**: Text chunks are embedded using SentenceTransformers and stored in ChromaDB
3. **Query Processing**: User queries trigger semantic search through the vector store
4. **AI Generation**: Claude generates responses using retrieved context and tool-based search
5. **Session Management**: Conversation history is maintained for context

### Key Components Architecture

**RAGSystem (`rag_system.py`)** - Main orchestrator that coordinates all components:
- Manages document processing workflow
- Orchestrates vector storage operations  
- Handles tool-based search integration with AI generation
- Provides session management for conversation context

**VectorStore (`vector_store.py`)** - ChromaDB integration layer:
- Two collections: `course_content` (text chunks) and `course_metadata` (course info)
- Uses SentenceTransformer embeddings (`all-MiniLM-L6-v2`)
- Handles semantic search with configurable result limits
- Supports incremental document addition (avoids re-processing existing courses)

**AIGenerator (`ai_generator.py`)** - Claude API integration:
- Uses tool-based approach where Claude can call search functions
- Configured with specialized system prompt for educational content
- Manages conversation history and tool execution
- Temperature set to 0 for consistent responses

**Search Tools (`search_tools.py`)** - Tool interface for Claude:
- `CourseSearchTool` provides semantic search capability to the AI
- Tool manager handles registration and execution of search tools
- Sources tracking for citation purposes

### Configuration (`config.py`)
All system parameters are centralized:
- **Chunking**: 800 chars with 100 char overlap for context preservation  
- **Search**: Max 5 results, 2-message conversation history
- **Models**: Claude Sonnet 4, MiniLM-L6-v2 embeddings
- **Storage**: ChromaDB persisted to `./chroma_db`

### Data Flow
- Documents are chunked with metadata (course_id, lesson_title, chunk_index)
- Each chunk gets embedded and stored with its metadata
- Queries trigger semantic search across chunks
- Claude receives search results as tool responses and synthesizes answers
- Session state maintains conversation context

### API Structure
**FastAPI app** (`app.py`) exposes:
- `POST /api/query` - Main query endpoint with session support
- `GET /api/courses` - Course analytics and statistics  
- Static file serving for frontend at root `/`
- Auto-loads documents from `/docs/` on startup

### Frontend Integration
Simple HTML/JS interface communicates with FastAPI backend. No complex framework dependencies - uses vanilla JS with fetch API for backend communication.
- always use uv to run server do not use pip directly
- make sure to use uv to manage all dependencies