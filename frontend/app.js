// ChatGPT-style Chatbot Application

// Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// State
let conversationHistory = [];

// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');
const themeToggle = document.getElementById('theme-toggle');
const apiStatus = document.getElementById('api-status');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkApiHealth();
    setupEventListeners();
    setupExampleQueries();
    autoResizeTextarea();
});

// Check API health
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const data = await response.json();
        if (data.query_engine_loaded) {
            apiStatus.textContent = '● Backend Connected';
            apiStatus.className = 'api-status connected';
        } else {
            apiStatus.textContent = '● Backend Not Ready';
            apiStatus.className = 'api-status disconnected';
        }
    } catch (error) {
        apiStatus.textContent = '● Backend Disconnected';
        apiStatus.className = 'api-status disconnected';
        console.error('Health check failed:', error);
        console.error('Make sure the backend server is running on http://localhost:5000');
        console.error('Try: python backend/searching/api_server.py');
    }
}

// Setup event listeners
function setupEventListeners() {
    sendBtn.addEventListener('click', handleSend);
    newChatBtn.addEventListener('click', startNewChat);
    sidebarToggle.addEventListener('click', toggleSidebar);
    themeToggle.addEventListener('click', toggleTheme);
    
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
    
    queryInput.addEventListener('input', () => {
        sendBtn.disabled = !queryInput.value.trim();
    });
}

// Setup example query buttons
function setupExampleQueries() {
    const exampleBtns = document.querySelectorAll('.example-btn');
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.getAttribute('data-query');
            queryInput.value = query;
            sendBtn.disabled = false;
            handleSend();
        });
    });
}

// Auto-resize textarea
function autoResizeTextarea() {
    queryInput.addEventListener('input', () => {
        queryInput.style.height = 'auto';
        queryInput.style.height = Math.min(queryInput.scrollHeight, 200) + 'px';
    });
}

// Handle send
async function handleSend() {
    const query = queryInput.value.trim();
    if (!query) return;

    // Remove welcome screen
    const welcomeScreen = document.querySelector('.welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.remove();
    }

    // Add user message
    addMessage('user', query);
    queryInput.value = '';
    queryInput.style.height = 'auto';
    sendBtn.disabled = true;
    
    // Show typing indicator
    showTypingIndicator();

    try {
        const settings = getSettings();
        
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                db_names: settings.db_names,
                top_k: settings.top_k,
                rerank: settings.rerank,
                contextualize: settings.contextualize,
                similarity_threshold: settings.similarity_threshold
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Request failed');
        }

        if (data.success) {
            hideTypingIndicator();
            
            // Show answer or helpful message
            if (data.answer) {
                addMessage('bot', data.answer);
            } else if (data.num_results === 0) {
                let errorMsg = "I couldn't find any relevant information to answer your query.\n\n";
                if (data.debug) {
                    errorMsg += `Debug Info:\n`;
                    errorMsg += `- Databases searched: ${data.debug.databases_searched.join(', ')}\n`;
                    errorMsg += `- Databases loaded: ${data.debug.databases_loaded.length > 0 ? data.debug.databases_loaded.join(', ') : 'None'}\n`;
                    errorMsg += `- Similarity threshold: ${data.debug.similarity_threshold}\n\n`;
                }
                errorMsg += `Suggestions:\n`;
                errorMsg += `- Try rephrasing your question\n`;
                const currentThreshold = parseFloat(document.getElementById('threshold').value) || 0.0;
                if (currentThreshold > -0.5) {
                    errorMsg += `- Lower the similarity threshold (currently ${currentThreshold}). Try -0.5 or -1.0\n`;
                } else {
                    errorMsg += `- Check if the databases are properly loaded\n`;
                }
                addMessage('bot', errorMsg);
            } else {
                addMessage('bot', 'No answer generated. Please enable "Generate Answer" in settings.');
            }
            
            conversationHistory.push({
                query: query,
                answer: data.answer,
                timestamp: new Date().toISOString()
            });
        } else {
            throw new Error(data.error || 'Query failed');
        }

    } catch (error) {
        hideTypingIndicator();
        addErrorMessage(error.message);
        console.error('Error:', error);
    }
}

// Get settings
function getSettings() {
    const dbCheckboxes = document.querySelectorAll('.checkbox-group input[type="checkbox"]:checked');
    const dbNames = Array.from(dbCheckboxes).map(cb => cb.value);
    
    return {
        db_names: dbNames.length > 0 ? dbNames : null,
        top_k: parseInt(document.getElementById('top-k').value) || 10,
        rerank: document.getElementById('rerank').checked,
        contextualize: document.getElementById('contextualize').checked,
        similarity_threshold: parseFloat(document.getElementById('threshold').value) || 0.0
    };
}

// Add message
function addMessage(type, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${type}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = type === 'user' ? 'U' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Show typing indicator
function showTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message message-bot';
    messageDiv.id = 'typing-indicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    
    const indicator = document.createElement('div');
    indicator.className = 'message-content';
    indicator.innerHTML = '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(indicator);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Hide typing indicator
function hideTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

// Add error message
function addErrorMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message message-bot';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = `Error: ${message}`;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(errorDiv);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Start new chat
function startNewChat() {
    conversationHistory = [];
    chatMessages.innerHTML = `
        <div class="welcome-screen">
            <div class="welcome-icon">🤖</div>
            <h2>Welcome to Compliance RAG Chatbot</h2>
            <p>Ask me anything about company policies, AIID incidents, or regulatory standards.</p>
            <div class="example-queries">
                <button class="example-btn" data-query="What are the privacy policies?">
                    What are the privacy policies?
                </button>
                <button class="example-btn" data-query="Find incidents related to data breaches">
                    Find incidents related to data breaches
                </button>
                <button class="example-btn" data-query="What GDPR requirements apply?">
                    What GDPR requirements apply?
                </button>
            </div>
        </div>
    `;
    setupExampleQueries();
}

// Toggle sidebar
function toggleSidebar() {
    sidebar.classList.toggle('open');
}

// Toggle theme
function toggleTheme() {
    const body = document.body;
    const isDark = body.classList.contains('dark-theme');
    
    if (isDark) {
        body.classList.remove('dark-theme');
        body.classList.add('light-theme');
        themeToggle.innerHTML = '<span>🌙</span>';
    } else {
        body.classList.remove('light-theme');
        body.classList.add('dark-theme');
        themeToggle.innerHTML = '<span>☀️</span>';
    }
    
    // Save preference
    localStorage.setItem('theme', isDark ? 'light' : 'dark');
}

// Load saved theme
function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    const body = document.body;
    
    if (savedTheme === 'dark') {
        body.classList.remove('light-theme');
        body.classList.add('dark-theme');
        themeToggle.innerHTML = '<span>☀️</span>';
    } else {
        body.classList.remove('dark-theme');
        body.classList.add('light-theme');
        themeToggle.innerHTML = '<span>🌙</span>';
    }
}

// Scroll to bottom
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Load theme on init
loadTheme();
