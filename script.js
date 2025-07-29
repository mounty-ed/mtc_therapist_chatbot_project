const API_URL = 'http://localhost:5000';
let sessionId = null;

async function init() {
    try {
        const response = await fetch(`${API_URL}/new_session`, { method: 'POST' });
        const data = await response.json();
        sessionId = data.session_id;
        console.log('Session created:', sessionId);
    } catch (error) {
        addMessage('error', 'Failed to connect to therapist service');
    }
}

async function sendMessage() {
    const input = document.getElementById('input');
    const message = input.value.trim();
    if (!message) return;

    input.value = '';
    addMessage('user', message);
    addMessage('typing', 'Therapist is typing...');

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_id: sessionId })
        });
        
        const data = await response.json();
        removeTyping();
        addMessage('therapist', data.response);
    } catch (error) {
        removeTyping();
        addMessage('error', 'Error sending message. Please try again.');
    }
}

function addMessage(type, content) {
    const messages = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = `message ${type}`;
    div.textContent = content;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function removeTyping() {
    const typing = document.querySelector('.typing');
    if (typing) typing.remove();
}

document.getElementById('send').onclick = sendMessage;
document.getElementById('input').onkeypress = (e) => {
    if (e.key === 'Enter') sendMessage();
};

init();