# RLM MCP Server

An intelligent document analysis platform powered by AI agents and the Model Context Protocol (MCP). This application enables LLMs to efficiently analyze, extract information from, and summarize large documents through an intelligent tool-based interface.

## Overview

RLM MCP Server is a sophisticated client-server application that combines the power of large language models (LLMs) with specialized tools for document analysis. It provides a scalable, session-aware architecture that maintains isolated Python environments for each user conversation, enabling complex multi-step document processing workflows.

### Key Features

- **Intelligent Document Analysis**: Leverage AI agents to understand and extract insights from large documents
- **Multi-LLM Support**: Integrate multiple LLM models via OpenRouter API (root model for planning, sub-models for specialized tasks)
- **Efficient Context Processing**: Smart tools prevent loading entire documents into memory, enabling analysis of very large files
- **Session Management**: Per-user-per-chat isolated Python REPL environments that maintain state across conversations
- **Extensible Tool System**: Both built-in tools and custom Python code execution for flexible document analysis
- **Real-time Code Execution**: Execute arbitrary Python code within isolated environments for advanced analysis

## Architecture

![RLM MCP Architecture](./RLM%20MCP%20Architecture.png)

### System Components

#### Client Layer (`rlm_mcp_client`)
- **FastAPI REST API**: Entry point for chat completion requests
- **Agent Factory**: Builds and configures AI agents with MCP toolsets
- **MCP HTTP Client**: Manages HTTP communication with the MCP server

#### Server Layer (`rlm_mcp_server`)
- **MCP FastMCP Server**: Central server exposing tools and handling requests
- **Session Manager**: Maintains isolated REPL environments per user-chat session
- **Tool Registry**: Provides specialized tools for document analysis

#### Tool Categories

**Execution Tools**:
- `run_code`: Execute arbitrary Python code in the user's isolated REPL environment
- `sub_llm`: Call a secondary LLM for text processing tasks like summarization

**Probing Tools** (for lightweight document exploration):
- `find_content_using_regex`: Search documents using regex patterns
- `get_context_metadata`: Retrieve metadata about the document (type, length)
- `get_all_variable_names`: List all available variables in the REPL environment
- `peek_context`: Preview portions of the document (head, middle, tail) using different strategies (chars, words, lines)

## Installation

### Prerequisites
- Python 3.12+
- OpenRouter API key (for LLM access)

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://gitlab.aruba.it/devspa/ai/lab-prototypes/mcp_servers/rlm-mcp-server.git
   cd rlm-mcp-server
   ```

2. **Install Dependencies**
   ```bash
   pip install poetry
   poetry install
   ```

3. **Configure Environment Variables**
   Create a `.env` file in the project root:
   ```env
   OPENROUTER_API_KEY=your_api_key_here
   MCP_SERVER_BASE_URL=http://localhost:8000
   ```

4. **Run the Application**
   
   Start the MCP server (in one terminal):
   ```bash
   python -m rlm_mcp.rlm_mcp_server.mcp_main
   ```
   
   Start the client API (in another terminal):
   ```bash
   uvicorn rlm_mcp.rlm_mcp_client.api_main:app --reload --port 8001
   ```

## Usage

### Making a Chat Completion Request

Send a POST request to `/v1/chat/completions`:

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user123",
    "chat_id": "chat_001",
    "root_model": "openrouter/free",
    "sub_model": "openrouter/free",
    "openrouter_api_key": "sk-or-v1-...",
    "messages": [
      {"role": "user", "content": "Summarize the document"}
    ],
    "document": "Your large document text here..."
  }'
```

### Use Cases

#### 1. Document Summarization
Ask the agent to summarize large documents by leveraging the sub-LLM tool to process chunks.

#### 2. Information Extraction
Use regex tools to find specific patterns in documents, then extract surrounding context.

#### 3. Complex Analysis
Write and execute Python code in the REPL environment to perform custom analysis on document data.

#### 4. Question Answering
Ask the AI agent questions about document content while it intelligently manages context through available tools.

## API Endpoints

### Health Check
- `GET /ping` - Returns `{"healthy": "true"}`

### Chat Completions
- `POST /v1/chat/completions` - Process document analysis requests
  - **Request Body**:
    - `username` (string): User identifier
    - `chat_id` (string): Conversation identifier
    - `root_model` (string): Primary LLM model (via OpenRouter)
    - `sub_model` (string): Secondary LLM model for specialized tasks
    - `openrouter_api_key` (string): OpenRouter API key
    - `messages` (list): Conversation history
    - `document` (string): Document to analyze

### Internal Endpoints
- `POST /internal/repl/context` - Preload context into REPL environment (requires X-User-Id and X-Chat-Id headers)

## MCP Tools Reference

### Execution Tools
- **`run_code`** - Execute Python code in user's isolated REPL environment
- **`sub_llm`** - Query secondary LLM for text processing tasks

### Probing Tools
- **`find_content_using_regex`** - Find content using regex patterns with surrounding context window
- **`get_context_metadata`** - Get document type and length information
- **`get_all_variable_names`** - List available variables in REPL environment
- **`peek_context`** - Preview document sections (head/middle/tail) by characters/words/lines

## Testing

Run the test suite:

```bash
pytest tests/
```

The test suite covers:
- Main API endpoints
- REPL environment management
- Probing tools functionality

## Project Structure

```
rlm-mcp-server/
├── src/rlm_mcp/
│   ├── domain/                    # Shared domain models
│   ├── rlm_mcp_client/           # Client-side components
│   │   ├── agents/               # Agent factory and configuration
│   │   ├── prompts/              # System prompts for agents
│   │   └── api_main.py           # FastAPI application
│   └── rlm_mcp_server/           # Server-side components
│       ├── tools/                # Tool implementations
│       ├── repl_environment/     # Python REPL management
│       ├── stateful_mcp/         # Session management
│       └── mcp_main.py           # MCP server entry point
├── tests/                        # Test suite
├── pyproject.toml               # Project dependencies
└── README.md                    # This file
```

## Technologies

- **FastAPI** - Modern web framework for building APIs
- **Pydantic-AI** - AI framework for building LLM-powered applications
- **Model Context Protocol (MCP)** - Standardized protocol for AI tool integration
- **OpenRouter** - Multi-model LLM API for accessing various language models
- **Python REPL** - Isolated Python environments per session

## Development

### Code Style & Quality
- Tests located in `tests/` directory
- Run tests with: `pytest tests/`

### Contributing
When contributing, ensure:
1. All tests pass
2. New features include appropriate tests
3. Code follows project conventions

## Author

**Andrea Cacioli** - Lead Developer

## License

This project is proprietary and developed for Aruba.

## Support

For issues, questions, or contributions, please contact the development team.
