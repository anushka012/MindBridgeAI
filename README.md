# 🧠 MindBridgeAI - Mental Health Support Chatbot

MindBridgeAI is an intelligent mental health assistant powered by the Gemini API and enhanced with contextual retrieval, sentiment analysis, and personalized mental health insights. It aims to provide brief, empathetic, and relevant responses based on user queries, combining natural language processing, FAISS vector search, and demographic-based insights.

---

## 📚 Table of Contents

- [Project Structure](#-project-structure)
- [Features](#-features)
- [Installation](#-installation)
- [Chat Interface](#-chat-interface)
- [How It Works](#-how-it-works)
- [Technologies Used](#technologies-used)

---

## 📁 Project Structure

```bash
mindbridgeai/
│
├── data/                            # Datasets and embeddings
│   ├── Combined Data.csv
│   ├── mental_health_data final data.csv
│   ├── data.csv
│   ├── KB.json
│   └── faiss_index.bin
│
├── models/
│   ├── context_retreival.py         # Contextual search using FAISS
│   └── sentiment_analysis.py        # NLTK VADER sentiment analysis
│
├── templates/
│   └── index.html                   # Frontend for chat
│
├── therapist_bot_env/              # Virtual environment folder
├── .env                            # API keys and configuration
├── .gitignore
├── app.py                          # Flask app
├── chatbot.py                      # Gemini response logic
├── config.py                       # Loads environment config
├── faiss_index.py                  # Index builder
├── intent_matcher.py               # Pattern matcher for known intents
├── preload.py                      # Pre-downloads model
├── requirements.txt
└── README.md
```

---

## ✨ Features

- 🌐 Web-based mental health chatbot (Flask UI)
- 🔍 Semantic search via FAISS and Sentence Transformers
- 🧠 Gemini integration for generative responses
- ❤️ Sentiment analysis with NLTK VADER
- 📊 Dynamic advice based on user demographics and stress levels
- 🧾 Intent recognition through pattern matching (regex-based)
- 💾 Supports multiple data sources (CSV-based)

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/mindbridgeai.git
cd mindbridgeai
```

### 2. Create a virtual environment
```bash
python -m venv therapist_bot_env
source therapist_bot_env/bin/activate  # or therapist_bot_env\Scripts\activate on Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create a .env file in the root directory
```bash
GEMINI_API_KEY=your-google-api-key
GEMINI_MODEL=models/gemini-1.0-pro
```

### 5. Build the FAISS index (first time only)
```bash
python faiss_index.py
```

### 6. (Optional) Preload the embedding model
```bash
python preload.py
```

### 7. Run the Flask app
```bash
python app.py
```

---

## 🌐 Chat Interface

- Visit http://localhost:5000 in your browser

- Chat messages are POSTed to /chat and responded with empathetic, intelligent replies

---

## 🧠 How It Works

1. Intent Matching: Checks if user input matches predefined patterns in KB.json.

2. If no match:

  - Retrieves relevant context via FAISS semantic search

  - Analyzes sentiment with VADER

  - Gathers stress/resource advice from data

  - Builds a prompt and sends it to Google Gemini for final response

3. Responds with an empathetic, concise reply

---

## Technologies Used

- Flask

- FAISS

- Sentence-Transformers

- Google Generative AI (Gemini)

- NLTK VADER

- Pandas / NumPy
