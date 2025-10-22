# B2B LLM Chatbot Project (Sahabat Chatbot)

A Django-based intelligent chatbot application with LLM integration and RAG (Retrieval-Augmented Generation) capabilities for B2B use cases, featuring document processing and tech support functionality.

## Features

### Core Chatbot Features
- Real-time chat interface with streaming responses
- User authentication and session management
- Chat history and memory management
- Multiple chat conversations support
- Profile and settings management
- Collapsible sidebar with navigation

### LLM Agent Features
- Tech Support Agent with RAG capabilities
- Custom prompt configuration
- RAG mode toggle for enhanced responses
- Configurable LLM settings

### Document Management (RAG)
- Multi-format document support (PDF, TXT, DOCX, XLSX, MD)
- Folder scanning and batch processing
- Document upload and management
- Vector embedding with sentence-transformers
- Semantic search capabilities
- Document analytics

### UI/UX Features
- Responsive Bootstrap-based interface
- Indosat branding with logo
- Dark/Light theme support
- Real-time streaming responses
- Sidebar with collapsible navigation

## Technology Stack

- **Backend Framework**: Django 5.2
- **Database**: SQLite with sqlite-vec extension
- **LLM Framework**: LangChain, LangGraph
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Document Processing**: PyPDF2, python-docx2txt, openpyxl
- **Frontend**: Bootstrap 5, Bootstrap Icons
- **Environment Management**: python-dotenv

## Project Structure

```
B2B-LLM-Project/
├── chatbot/
│   ├── app/                    # Main chatbot application
│   │   ├── models.py          # Chat, Message models
│   │   ├── views.py           # Chat views and API endpoints
│   │   ├── templates/         # HTML templates
│   │   └── static/            # CSS, JS, images
│   ├── chat_rag/              # RAG functionality
│   │   ├── models.py          # Document models
│   │   ├── views.py           # Document and RAG API views
│   │   └── templates/         # RAG templates
│   ├── chatbot/               # Django project settings
│   │   ├── settings.py        # Project configuration
│   │   └── urls.py            # URL routing
│   ├── documents/             # Document storage (gitignored)
│   ├── static/                # Global static files
│   │   └── image/             # Images (Indosat logo)
│   ├── manage.py              # Django management script
│   └── db.sqlite3             # SQLite database
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone http://10.45.127.199:8082/b2b-services-delivery/sahabat_chatbot.git
   cd B2B-LLM-Project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file in the `chatbot` directory with the following variables:
   ```env
   # LLM Configuration
   LLM_API_KEY=your_api_key_here
   LLM_API_URL=your_llm_endpoint

   # Django Settings (optional)
   SECRET_KEY=your_secret_key
   DEBUG=True
   ```

5. **Run migrations**
   ```bash
   cd chatbot
   python manage.py migrate
   ```

6. **Create a superuser (optional)**
   ```bash
   python manage.py createsuperuser
   ```

7. **Collect static files**
   ```bash
   python manage.py collectstatic --noinput
   ```

8. **Run the development server**
   ```bash
   python manage.py runserver
   ```

9. **Access the application**

   Open your browser and navigate to:
   - Main application: `http://127.0.0.1:8000/`
   - Admin panel: `http://127.0.0.1:8000/admin/`

## Configuration

### RAG Configuration

The RAG system can be configured in `chatbot/settings.py`:

```python
RAG_CONFIG = {
    'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',  # Sentence transformer model
    'CHUNK_SIZE': 1000,                      # Characters per chunk
    'CHUNK_OVERLAP': 200,                    # Overlap between chunks
    'TOP_K_RESULTS': 5,                      # Number of results to return
    'DOCUMENT_FOLDER': BASE_DIR / 'documents',  # Folder to process documents
    'SUPPORTED_FORMATS': ['.pdf', '.txt', '.docx', '.xlsx', '.md'],
}
```

### Network Access

To allow network access from other devices, update `ALLOWED_HOSTS` in `settings.py`:

```python
ALLOWED_HOSTS = ['localhost', '127.0.0.1', 'your-server-ip']
```

Then run the server with:
```bash
python manage.py runserver 0.0.0.0:8000
```

## Usage

### Basic Chat

1. Register a new account or login
2. Click "New Chat" to start a conversation
3. Type your message and press Enter or click Send
4. View chat history in the sidebar

### Tech Support with RAG

1. Navigate to "LLM Agent" > "Tech Support" in the sidebar
2. Upload documents via the document management interface
3. Process documents to create embeddings
4. Ask questions related to your uploaded documents
5. The system will provide context-aware answers

### Document Management

1. Go to the RAG documents page (`/rag/documents/`)
2. Upload documents (PDF, DOCX, TXT, XLSX, MD)
3. Click "Process" to generate embeddings
4. Documents are now searchable via semantic search

## API Endpoints

### Chat API

- `POST /api/chat/` - Send a chat message (JSON response)
- `POST /api/chat/stream/` - Send a chat message (streaming response)

### RAG API

- `POST /rag/chat/api/` - RAG-enhanced chat (JSON response)
- `POST /rag/chat/stream/` - RAG-enhanced chat (streaming response)
- `POST /rag/documents/` - Upload and manage documents
- `GET /rag/search/` - Search documents semantically

## Development

### Running Tests

```bash
python manage.py test
```

### Database Management

**Reset database:**
```bash
rm db.sqlite3
python manage.py migrate
```

**Create migrations:**
```bash
python manage.py makemigrations
python manage.py migrate
```

## Troubleshooting

### Static Files Not Loading

```bash
python manage.py collectstatic --clear --noinput
```

### Database Issues

```bash
python manage.py migrate --run-syncdb
```

### Port Already in Use

```bash
python manage.py runserver 8001  # Use a different port
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a merge request

## Support

For support and questions, please contact the B2B Services Delivery team.

## Project Status

Active development - Maintained by B2B Services Delivery Team

## Acknowledgments

- Django framework
- LangChain for LLM integration
- Sentence Transformers for embeddings
- Bootstrap for UI components
- Indosat for branding and support
