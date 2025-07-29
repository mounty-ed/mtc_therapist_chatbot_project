# MTC Therapist Chatbot

A web-based **therapist chatbot** designed to bring warm, empathetic support—24/7—to community workers and the most vulnerable among us. Imagine someone sleeping on a park bench, struggling with anxiety, loneliness, or past trauma, with no one to turn to. Our bot listens without judgment, offers reflective questions, and connects people to resources when it matters most.

---

## 💔 Why This Matters

Every night, thousands of people in our city face darkness alone. Homeless shelters overflow, social services are stretched thin, and a moment of crisis can slip by unheard. This chatbot acts as a first line of compassionate care: 

- **Immediate companionship** for someone feeling isolated on the streets.
- **Gentle guidance** toward breathing exercises or grounding techniques.
- **Crisis detection** that triggers emergency hotlines, sending a life-saving lifeline when triggered.

Whether you’re an outreach worker, a volunteer at a drop-in center, or a neighbor checking in on someone sleeping rough, MTC fills a gap—anywhere there’s internet or a shared device.

---

## 📁 Project Structure

```

.
├── README.md                    ← This documentation
├── back\_end
│   ├── app.py                   ← Flask application (API endpoints)
│   └── therapist\_agent.py       ← Core “TherapistAgent” state-graph logic
├── index.html                   ← Frontend chat UI
├── script.js                    ← Frontend message handling & AJAX
├── style.css                    ← Frontend styling
└── therapy\_workflow\.png         ← Optional: visual of the agent’s workflow

````

---

## 🚀 Getting Started

### 1. Clone & Prepare

Download the files and enter mtc_therapist_chatbot_project in the terminal.

### 2. Python Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
.\.venv\Scripts\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:**
>
> * You’ll also need an Openrouter or Openai API key via `.env` (you will have to edit ChatOpenAI parameters in back_end/therapist_agent.py to support Openai).
> * If you encounter Graphviz errors when generating the workflow image, install the system package: `sudo apt install graphviz` (or your OS’s equivalent).

### 4. Environment Configuration

Create a `.env` file in the project root:

```ini
OPENROUTER_API_KEY=your_openrouter_key_here
```

### 5. Launch Backend

```bash
cd back_end
python app.py
```

* **Health check**: `GET http://localhost:5000/health`
* **Start a new session**: `POST http://localhost:5000/new_session`
* **Send a message**: `POST http://localhost:5000/chat`

  ```json
  { "session_id": "…", "message": "Hello, I’m feeling anxious." }
  ```

### 6. Serve Frontend

You can open `index.html` directly in your browser or serve it via a simple HTTP server:

```bash
# From project root:
python3 -m http.server 8000
# Then visit http://localhost:8000/index.html
```

The frontend (in `script.js`) will:

1. Capture user input
2. Send it to `/chat`
3. Render messages and timestamps
4. Display a disclaimer with crisis-support hotlines

---

## 🔧 How It Works

1. **StateGraph Workflow**
   `TherapistAgent` (in `therapist_agent.py`) uses LangGraph’s **StateGraph** to:

   * **Detect crisis** keywords (e.g. self-harm, violence).
   * **Route** to a special crisis branch if needed.
   * **Manage memory**: summarize long histories.
   * **Generate or regenerate** therapeutic responses.
   * **Reflect** on response quality, optionally re-invoke LLMs for improvements.

2. **LLM Models**

   * gpt-4o-mini for core chat
   * mistralai/ministral-8b for summaries and reflection
   * Specialized “crisis” and “reflection” configurations to ensure safety, empathy, and professionalism.

3. **Frontend**

   * Minimal HTML/CSS/JS
   * Clean chat window with accessible input, send button, and footnote disclaimer.

---

## 🤝 For Social-Good Practitioners

* **Not a replacement** for real counseling—always refer crisis cases to professional services.
* **Customizable**: tweak prompts, add local hotlines, or integrate with existing community-support platforms.
* **Open-source** and extensible: invite contributions from fellow social-good devs!

---

## 🛠️ Future Improvements

* Persist conversation history in a database
* Add user authentication & role-based access
* Enhance UI/UX: mobile responsiveness, theming
* Analytics dashboard to track usage & sentiment
* Trigger emergency guidance and hotline info during crisis response

---

Disclaimer: This chatbot is not a substitute for professional mental health care. In case of crisis, always contact real emergency services or hotlines. Use this chatbot as supplemental support