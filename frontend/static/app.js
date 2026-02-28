// API Configuration
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:5000/api' 
    : '/api';  // Use relative path if same origin

let conversationHistory = [];

// DOM elements
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const searchMode = document.getElementById('search-mode');
const topK = document.getElementById('top-k');
const rerank = document.getElementById('rerank');
const generateAnswer = document.getElementById('generate-answer');

// Event listeners
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function getModeBadge(mode) {
    const badges = {
        'vector': 'VECTOR',
        'graph': 'GRAPH',
        'hybrid': 'HYBRID'
    };
    return badges[mode] || mode.toUpperCase();
}

function addMessage(content, isUser, mode = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    
    let html = '';
    if (mode) {
        html += `<div class="mode-badge">${getModeBadge(mode)}</div>`;
    }
    
    if (isUser) {
        html += `<div>${escapeHtml(content)}</div>`;
    } else {
        if (content.answer) {
            html += `<div class="answer">${escapeHtml(content.answer)}</div>`;
        }
        if (content.results && content.results.length > 0) {
            html += `<div class="results">Found ${content.num_results} result(s)</div>`;
        }
    }
    
    messageDiv.innerHTML = html;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function sendMessage() {
    const query = userInput.value.trim();
    if (!query) return;
    
    const mode = searchMode.value;
    
    // Add user message
    addMessage(query, true, mode);
    userInput.value = '';
    sendBtn.disabled = true;
    
    // Show loading
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.textContent = 'Processing...';
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                mode: mode,
                top_k: parseInt(topK.value),
                rerank: rerank.checked,
                generate_answer: generateAnswer.checked
            })
        });
        
        const data = await response.json();
        
        // Remove loading
        chatMessages.removeChild(loadingDiv);
        
        if (data.success) {
            addMessage({
                answer: data.answer || 'No answer generated.',
                results: data.results,
                num_results: data.num_results
            }, false, mode);
            
            conversationHistory.push({
                query: query,
                mode: mode,
                response: data
            });
        } else {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = `Error: ${data.error || 'Unknown error'}`;
            chatMessages.appendChild(errorDiv);
        }
    } catch (error) {
        chatMessages.removeChild(loadingDiv);
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error';
        errorDiv.textContent = `Error: ${error.message}`;
        chatMessages.appendChild(errorDiv);
    } finally {
        sendBtn.disabled = false;
    }
}

// Initialize
addMessage('Welcome! Select a search mode and ask your question.', false);
