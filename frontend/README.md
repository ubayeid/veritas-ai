# Compliance RAG Chatbot - Web Interface

Web-based interface for the Compliance RAG Chatbot, allowing you to query company policies, AIID incidents, and regulatory standards through a modern browser interface.

## Features

- 🎨 Modern, responsive UI
- 💬 Real-time chat interface
- ⚙️ Configurable search settings
- 📊 Display of search results with similarity scores
- 🤖 AI-generated contextualized answers
- 📱 Mobile-friendly design

## Setup

### Prerequisites

1. Backend API server must be running (see `backend/searching/README.md`)
2. Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

No installation needed! Just open `index.html` in your browser, or serve it through a web server.

### Running Locally

#### Option 1: Using the Provided Script (Recommended)

**Windows:**
```bash
# From project root
start_web_app.bat
```

**Linux/Mac:**
```bash
# From project root
chmod +x start_web_app.sh
./start_web_app.sh
```

This will start both the backend API server and frontend server automatically.

#### Option 2: Manual Start - Python HTTP Server

**Start Backend:**
```bash
# Terminal 1 - Backend API Server
python backend/searching/api_server.py
```

**Start Frontend:**
```bash
# Terminal 2 - Frontend Server
cd frontend
python start_server.py
# Or use: python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

#### Option 3: Direct File Access (Not Recommended)

Simply open `frontend/index.html` in your web browser.

**Note:** Some browsers may block CORS requests when opening files directly. Use Option 1 or 2 instead.

## Configuration

### API Endpoint

By default, the frontend connects to `http://localhost:5000/api`. 

To change this, edit `frontend/app.js` and modify the `API_BASE_URL` constant:

```javascript
const API_BASE_URL = 'http://your-api-url:port/api';
```

### Settings

The web interface allows you to configure:

- **Databases**: Select which databases to search (Company, AIID, Standards)
- **Top K Results**: Number of results to retrieve (1-50)
- **Rerank Results**: Enable/disable LLM-based reranking
- **Generate Answer**: Enable/disable contextualized answer generation
- **Similarity Threshold**: Minimum similarity score (0.0-1.0)

## Usage

1. **Start the Backend API Server**:
   ```bash
   python backend/searching/api_server.py
   ```

2. **Open the Frontend**:
   - Option 1: Open `frontend/index.html` directly in your browser
   - Option 2: Serve it using a web server (see above)

3. **Start Chatting**:
   - Type your question in the input box
   - Click "Send" or press Enter
   - View the AI-generated answer and search results

## API Endpoints

The frontend uses the following API endpoints:

- `GET /api/health` - Health check
- `GET /api/databases` - List available databases
- `POST /api/query` - Submit a query and get answer + results
- `POST /api/search` - Search without contextualization

See `backend/searching/api_server.py` for API documentation.

## Troubleshooting

### CORS Errors

If you see CORS errors in the browser console:
- Make sure you're serving the frontend through a web server (not opening the file directly)
- Or configure CORS in the backend API server

### Cannot Connect to API

- Ensure the backend API server is running on port 5000
- Check that the API URL in `app.js` matches your backend server URL
- Check browser console for detailed error messages

### No Results Found

- Verify that FAISS databases are built
- Check backend server logs for errors
- Try adjusting the similarity threshold in settings

## Browser Compatibility

- Chrome/Edge: ✅ Fully supported
- Firefox: ✅ Fully supported
- Safari: ✅ Fully supported
- Internet Explorer: ❌ Not supported

## Development

To modify the frontend:

1. Edit `index.html` for structure
2. Edit `styles.css` for styling
3. Edit `app.js` for functionality

The frontend uses vanilla JavaScript (no frameworks required) for simplicity and ease of customization.

