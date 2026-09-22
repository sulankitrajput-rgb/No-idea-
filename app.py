from flask import (
    Flask,
    request,
    jsonify,
    session,
    redirect,
    url_for
)
from flask_cors import CORS
import requests
import os
import psycopg
import psycopg.rows
import base64
from html import escape
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)

app.secret_key = os.getenv("SECRET_KEY", "change-this-secret-key")

# Render's Postgres add-on provides this env var automatically once the
# database is created and linked to the web service. Locally, set it
# yourself, e.g.:
#   export DATABASE_URL=postgresql://user:password@localhost:5432/atlas
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Add a PostgreSQL database on Render "
        "(or point this at your own Postgres instance) and set "
        "DATABASE_URL before starting the app."
    )

# Some providers (Render included, historically) hand out URLs starting
# with "postgres://", but psycopg/SQLAlchemy-style URLs expect
# "postgresql://". Normalize it so either form works.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


def get_db_connection():
    connection = psycopg.connect(
        DATABASE_URL,
        row_factory=psycopg.rows.dict_row
    )
    return connection


def initialize_database():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    connection.commit()
    cursor.close()
    connection.close()


def login_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return function(*args, **kwargs)
    return wrapper


initialize_database()


# ============================================================
# PROJECT ATLAS - API KEYS
# ============================================================

GROQ_KEY = os.getenv("GROQ_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")
OPENAI_KEY = os.getenv("OPENAI_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")


# ============================================================
# CONVERSATION MEMORY (in-memory, per logged-in user)
# ============================================================
# Note: this resets whenever the server restarts, and won't be
# shared across multiple worker processes. Good enough for a
# per-session "remember our chat" feature; swap for a database
# table keyed by user_id if you need it to persist long-term.

CONVERSATION_MEMORY = {}
MAX_MEMORY_TURNS = 6
MAX_STORED_ANSWER_CHARS = 600


def get_memory(user_id):
    return CONVERSATION_MEMORY.get(user_id, [])


def add_memory_turn(user_id, question, answers):
    history = CONVERSATION_MEMORY.setdefault(user_id, [])
    history.append({
        "question": question,
        "chatgpt": answers.get("chatgpt", "")[:MAX_STORED_ANSWER_CHARS],
        "gemini": answers.get("gemini", "")[:MAX_STORED_ANSWER_CHARS],
        "groq": answers.get("groq", "")[:MAX_STORED_ANSWER_CHARS],
        "claude": answers.get("claude", "")[:MAX_STORED_ANSWER_CHARS],
    })
    CONVERSATION_MEMORY[user_id] = history[-MAX_MEMORY_TURNS:]


def build_prompt_with_memory(user_id, model_key, question, memory_enabled):
    if not memory_enabled:
        return question

    history = get_memory(user_id)

    if not history:
        return question

    lines = ["Here is our conversation so far. Use it as context for the new question.\n"]

    for turn in history:
        lines.append(f"Previous question: {turn['question']}")
        lines.append(f"Previous answer: {turn.get(model_key, '')}")
        lines.append("")

    lines.append(f"New question: {question}")

    return "\n".join(lines)


# ============================================================
# HELPER
# ============================================================

def get_text(response):
    if response is None:
        return "No response."

    if not isinstance(response, dict):
        return str(response)

    if "answer" in response:
        return str(response["answer"])

    if "error" in response:
        error = response["error"]

        if isinstance(error, dict):
            error = error.get("message", str(error))

        return "❌ " + str(error)

    return str(response)


# ============================================================
# GROQ
# ============================================================

def ask_groq(question):

    if not GROQ_KEY:
        return {
            "error": "GROQ_KEY is not configured."
        }

    try:

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",

            headers={
                "Authorization": f"Bearer {GROQ_KEY}",
                "Content-Type": "application/json"
            },

            json={
                "model": "openai/gpt-oss-20b",

                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ],

                "temperature": 0.7,

                "max_tokens": 2048
            },

            timeout=60
        )

        if response.status_code == 429:
            return {
                "error": "Groq is temporarily busy or rate-limited. Please try again in a few seconds."
            }

        if response.status_code != 200:
            try:
                error_data = response.json()
                error_message = (
                    error_data
                    .get("error", {})
                    .get("message", response.text)
                )
            except Exception:
                error_message = response.text

            return {
                "error": f"Groq API error: {error_message}"
            }

        data = response.json()
        choices = data.get("choices", [])

        if not choices:
            return {
                "error": "Groq returned no answer."
            }

        message = choices[0].get("message", {})
        answer = message.get("content", "")

        if not answer:
            return {
                "error": "Groq returned an empty answer."
            }

        return {
            "answer": answer
        }

    except requests.exceptions.Timeout:
        return {
            "error": "Groq took too long to respond. Please try again."
        }

    except requests.exceptions.RequestException as e:
        return {
            "error": f"Groq connection error: {str(e)}"
        }

    except Exception as e:
        return {
            "error": f"Groq error: {str(e)}"
        }


# ============================================================
# GEMINI
# ============================================================

def ask_gemini(question, image=None):

    if not GEMINI_KEY:
        return {"error": "GEMINI_KEY is not configured."}

    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/gemini-3.5-flash:generateContent"
    )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_KEY
    }

    try:
        if not image:
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": question
                            }
                        ]
                    }
                ]
            }
        else:
            if isinstance(image, bytes):
                image_data = base64.b64encode(image).decode("utf-8")
            else:
                image_data = image

                if "," in image_data and image_data.startswith("data:"):
                    image_data = image_data.split(",", 1)[1]

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": question
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_data
                                }
                            }
                        ]
                    }
                ]
            }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=(5, 15)
        )

        if response.status_code != 200:
            return {
                "error": "Gemini API error: " + response.text
            }

        data = response.json()
        candidates = data.get("candidates", [])

        if not candidates:
            return {
                "error": "Gemini returned no answer."
            }

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        if not parts:
            return {
                "error": "Gemini returned no text."
            }

        text = parts[0].get("text", "")

        if not text:
            return {
                "error": "Gemini returned an empty answer."
            }

        return {
            "answer": text
        }

    except requests.exceptions.Timeout:
        return {
            "error": "Gemini request timed out."
        }

    except requests.exceptions.RequestException as e:
        return {
            "error": "Gemini connection error: " + str(e)
        }

    except Exception as e:
        return {
            "error": "Gemini error: " + str(e)
        }


# ============================================================
# CHATGPT / OPENAI
# ============================================================

def ask_openai(question):

    if not OPENAI_KEY:
        return {"error": "OPENAI_KEY is not configured."}

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4.1-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            },
            timeout=60
        )

        if response.status_code != 200:
            return {"error": response.text}

        data = response.json()
        choices = data.get("choices", [])

        if not choices:
            return {"error": "ChatGPT returned no answer."}

        return {
            "answer": choices[0]["message"]["content"]
        }

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# CLAUDE
# ============================================================

def ask_claude(question):

    if not ANTHROPIC_KEY:
        return {"error": "ANTHROPIC_KEY is not configured."}

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            },
            timeout=60
        )

        if response.status_code != 200:
            return {"error": response.text}

        data = response.json()
        content = data.get("content", [])

        if not content:
            return {"error": "Claude returned no answer."}

        return {
            "answer": content[0].get("text", "Claude returned an empty answer.")
        }

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# AUTHENTICATION ROUTES
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username = %s",
            (username,)
        )
        user = cursor.fetchone()
        cursor.close()
        connection.close()

        if user and check_password_hash(user["password"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("home"))

        return """
        <h2>Invalid username or password</h2>
        <a href="/login">Try again</a>
        """

    return """
<!DOCTYPE html>
<html>
<head>
    <title>Sign In - Project Atlas</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            margin: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: Arial, sans-serif;
            background: #f4f7fb;
        }
        .login-card {
            width: 90%;
            max-width: 400px;
            padding: 30px;
            background: white;
            border-radius: 14px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.12);
        }
        h1 {
            text-align: center;
            color: #1769ff;
        }
        input {
            width: 100%;
            padding: 13px;
            margin: 8px 0;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 8px;
            font-size: 16px;
        }
        button {
            width: 100%;
            padding: 14px;
            margin-top: 12px;
            background: #1769ff;
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover {
            background: #0d55d9;
        }
        .signup-link {
            margin-top: 18px;
            text-align: center;
        }
        a {
            color: #1769ff;
        }
    </style>
</head>
<body>
    <div class="login-card">
        <h1>Project Atlas</h1>
        <form method="POST" action="/login">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Sign In</button>
        </form>
        <div class="signup-link">
            Do not have an account?
            <a href="/signup">Create one</a>
        </div>
    </div>
</body>
</html>
"""


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            return """
            <h2>Username and password are required.</h2>
            <a href="/signup">Try again</a>
            """

        if len(password) < 6:
            return """
            <h2>Password must contain at least 6 characters.</h2>
            <a href="/signup">Try again</a>
            """

        hashed_password = generate_password_hash(password)
        connection = get_db_connection()
        cursor = connection.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO users (username, password)
                VALUES (%s, %s)
                """,
                (username, hashed_password)
            )
            connection.commit()
            cursor.close()
            connection.close()
            return redirect(url_for("login"))

        except psycopg.errors.UniqueViolation:
            connection.rollback()
            cursor.close()
            connection.close()
            return """
            <h2>That username already exists.</h2>
            <a href="/signup">Try another username</a>
            """

    return """
<!DOCTYPE html>
<html>
<head>
    <title>Create Account - Project Atlas</title>
    <style>
        body {
            margin: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: Arial, sans-serif;
            background: #f4f7fb;
        }
        .signup-card {
            width: 90%;
            max-width: 400px;
            padding: 30px;
            background: white;
            border-radius: 14px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.12);
        }
        h1 {
            text-align: center;
            color: #1769ff;
        }
        input {
            width: 100%;
            padding: 13px;
            margin: 8px 0;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 8px;
            font-size: 16px;
        }
        button {
            width: 100%;
            padding: 14px;
            margin-top: 12px;
            background: #1769ff;
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }
        .login-link {
            margin-top: 18px;
            text-align: center;
        }
        a {
            color: #1769ff;
        }
    </style>
</head>
<body>
    <div class="signup-card">
        <h1>Create Account</h1>
        <form method="POST" action="/signup">
            <input type="text" name="username" placeholder="Choose a username" required>
            <input type="password" name="password" placeholder="Choose a password" minlength="6" required>
            <button type="submit">Create Account</button>
        </form>
        <div class="login-link">
            Already have an account?
            <a href="/login">Sign in</a>
        </div>
    </div>
</body>
</html>
"""


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ============================================================
# MEMORY CONTROLS
# ============================================================

@app.route("/memory/toggle", methods=["POST"])
@login_required
def toggle_memory():
    current = session.get("memory_enabled", False)
    session["memory_enabled"] = not current
    return jsonify({
        "memory_enabled": session["memory_enabled"]
    })


@app.route("/memory/clear", methods=["POST"])
@login_required
def clear_memory():
    user_id = session["user_id"]
    CONVERSATION_MEMORY[user_id] = []
    return jsonify({
        "cleared": True
    })


# ============================================================
# SINGLE MODEL
# ============================================================

@app.route("/ask", methods=["POST"])
@login_required
def ask():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "No valid JSON received."
        }), 400

    question = data.get("question", "").strip()
    model = data.get("model", "groq").lower()

    if not question:
        return jsonify({
            "error": "Question is empty."
        }), 400

    if model == "groq":
        result = ask_groq(question)
    elif model == "gemini":
        result = ask_gemini(question)
    elif model == "chatgpt":
        result = ask_openai(question)
    elif model == "claude":
        result = ask_claude(question)
    elif model == "deepseek":
        result = ask_deepseek(question)
    else:
        return jsonify({
            "error": f"Unknown model: {model}"
        }), 400

    return jsonify(result)


# ============================================================
# COMPARE ALL AI MODELS
# ============================================================

@app.route("/compare", methods=["POST"])
@login_required
def compare():
    print("===== PROJECT ATLAS COMPARE =====")

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "No valid JSON received."
        }), 400

    question = data.get("question", "").strip()
    image = data.get("image", "")

    if not question:
        return jsonify({
            "error": "Question is empty."
        }), 400

    user_id = session["user_id"]
    memory_enabled = session.get("memory_enabled", False)

    prompt_for_chatgpt = build_prompt_with_memory(user_id, "chatgpt", question, memory_enabled)
    prompt_for_gemini = build_prompt_with_memory(user_id, "gemini", question, memory_enabled)
    prompt_for_groq = build_prompt_with_memory(user_id, "groq", question, memory_enabled)
    prompt_for_claude = build_prompt_with_memory(user_id, "claude", question, memory_enabled)

    try:
        chatgpt = ask_openai(prompt_for_chatgpt)
    except Exception as e:
        chatgpt = {"error": str(e)}

    try:
        gemini = ask_gemini(prompt_for_gemini, image)
    except Exception as e:
        gemini = {"error": str(e)}

    try:
        groq = ask_groq(prompt_for_groq)
    except Exception as e:
        groq = {"error": str(e)}

    try:
        claude = ask_claude(prompt_for_claude)
    except Exception as e:
        claude = {"error": str(e)}

    chatgpt_text = get_text(chatgpt)
    gemini_text = get_text(gemini)
    groq_text = get_text(groq)
    claude_text = get_text(claude)

    if memory_enabled:
        add_memory_turn(user_id, question, {
            "chatgpt": chatgpt_text,
            "gemini": gemini_text,
            "groq": groq_text,
            "claude": claude_text,
        })

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{
    margin: 0;
    padding: 20px;
    background: #f3f6fb;
    font-family: Arial, sans-serif;
}}
.question-box {{
    background: white;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 25px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
}}
.question-box b {{
    color: #1769ff;
}}
.compare-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 20px;
}}
.ai-card {{
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.09);
    border-top: 5px solid #1769ff;
    min-width: 0;
}}
.ai-card h2 {{
    margin-top: 0;
    color: #172554;
}}
.ai-card pre {{
    margin: 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    word-break: break-word;
    font-family: Arial, sans-serif;
    font-size: 15px;
    line-height: 1.6;
    color: #263244;
}}
@media (max-width: 700px) {{
    .compare-grid {{
        grid-template-columns: 1fr;
    }}
}}
</style>
</head>
<body>
<div class="question-box">
<b>Question:</b><br>
{escape(question)}
</div>
<div class="compare-grid">
<div class="ai-card">
<h2>ChatGPT</h2>
<pre>{escape(chatgpt_text)}</pre>
</div>
<div class="ai-card">
<h2>Gemini</h2>
<pre>{escape(gemini_text)}</pre>
</div>
<div class="ai-card">
<h2>Groq</h2>
<pre>{escape(groq_text)}</pre>
</div>
<div class="ai-card">
<h2>Claude</h2>
<pre>{escape(claude_text)}</pre>
</div>
</div>
</body>
</html>
"""


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
@login_required
def home():
    username = session.get("username", "")
    memory_enabled = session.get("memory_enabled", False)
    memory_label = "🧠 Memory: ON" if memory_enabled else "🧠 Memory: OFF"
    html = """
<!DOCTYPE html>
<html>
<head>
<title>Project Atlas - AI Comparison</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f7fb;
}
.header {
    background: #1769ff;
    color: white;
    padding: 30px;
    text-align: center;
}
.header h1 {
    margin: 0;
    font-size: 34px;
}
.header p {
    margin: 10px 0 0;
    font-size: 18px;
}
.container {
    max-width: 900px;
    margin: 30px auto;
    padding: 20px;
}
textarea {
    width: 100%;
    height: 120px;
    padding: 15px;
    font-size: 17px;
    border: 1px solid #ccc;
    border-radius: 12px;
    box-sizing: border-box;
    resize: vertical;
}
.style-label {
    display: block;
    margin-top: 20px;
    margin-bottom: 8px;
    font-size: 17px;
    font-weight: bold;
}
select {
    width: 100%;
    padding: 14px;
    font-size: 16px;
    border: 1px solid #ccc;
    border-radius: 10px;
    background: white;
    box-sizing: border-box;
}
button {
    width: 100%;
    margin-top: 18px;
    padding: 15px;
    font-size: 18px;
    font-weight: bold;
    color: white;
    background: #1769ff;
    border: none;
    border-radius: 10px;
    cursor: pointer;
}
button:hover {
    background: #0d55d9;
}
#loading {
    display: none;
    text-align: center;
    margin: 25px;
    font-size: 18px;
    font-weight: bold;
}
#result {
    margin-top: 30px;
}
.compare-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 20px;
}
.ai-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    border-left: 5px solid #1769ff;
}
.ai-card h2 {
    margin-top: 0;
    margin-bottom: 15px;
    font-size: 22px;
}
.ai-answer {
    overflow-x: auto;
}
pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: Arial, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    margin: 0;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}
.footer {
    text-align: center;
    color: #777;
    margin: 30px;
}
.camera-row {
    display: flex;
    gap: 10px;
    margin-top: 14px;
    flex-wrap: wrap;
}
.camera-row button {
    margin-top: 0;
    width: auto;
    flex: 1;
    min-width: 140px;
    background: #34a853;
}
.camera-row button:hover {
    background: #2a8a44;
}
.camera-row .secondary {
    background: #6b7280;
}
.camera-row .secondary:hover {
    background: #52585f;
}
.camera-row .memory-off {
    background: #8b5cf6;
}
.camera-row .memory-off:hover {
    background: #7443e0;
}
.camera-row .memory-on {
    background: #059669;
}
.camera-row .memory-on:hover {
    background: #047857;
}
.memory-clear-link {
    display: none;
    text-align: center;
    margin-top: 8px;
    font-size: 14px;
}
.memory-clear-link a {
    color: #d93025;
    cursor: pointer;
    text-decoration: underline;
}
.memory-clear-link.visible {
    display: block;
}
#cameraModal {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.75);
    z-index: 999;
    align-items: center;
    justify-content: center;
}
#cameraModal.open {
    display: flex;
}
.camera-box {
    background: white;
    padding: 16px;
    border-radius: 14px;
    width: 92%;
    max-width: 480px;
    text-align: center;
}
#cameraVideo {
    width: 100%;
    border-radius: 10px;
    background: #000;
}
.camera-controls {
    display: flex;
    gap: 10px;
    margin-top: 12px;
}
.camera-controls button {
    margin-top: 0;
}
#photoPreview {
    margin-top: 14px;
    display: none;
}
#photoPreview img {
    max-width: 100%;
    border-radius: 10px;
    border: 1px solid #ddd;
}
#photoPreview .remove-photo {
    display: inline-block;
    margin-top: 8px;
    padding: 8px 14px;
    background: #d93025;
    color: white;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
}
@media (max-width: 600px) {
    .container {
        width: 94%;
        margin: 20px auto;
        padding: 10px;
    }
    .header h1 {
        font-size: 28px;
    }
    textarea {
        height: 120px;
    }
    button {
        font-size: 16px;
    }
}
</style>
</head>
<body>
<div class="header">
    <h1>PROJECT ATLAS</h1>
    <p>AI COMPARISON</p>
    <p>
        Signed in as __USERNAME__ |
        <a href="/logout" style="color: white;">Logout</a>
    </p>
</div>
<div class="container">
    <textarea id="question" placeholder="Ask Project Atlas anything..."></textarea>
    <label class="style-label" for="style">Answer Style:</label>
    <select id="style">
        <option value="balanced">Balanced</option>
        <option value="simple">Simple</option>
        <option value="detailed">Detailed</option>
        <option value="creative">Creative</option>
    </select>
    <div class="camera-row">
        <button onclick="openCamera()">📷 Take Photo</button>
        <button id="memoryButton" class="__MEMORY_BUTTON_CLASS__" onclick="toggleMemory()">__MEMORY_LABEL__</button>
    </div>
    <div id="memoryClearLink" class="memory-clear-link __MEMORY_CLEAR_VISIBLE__">
        <a onclick="clearMemory()">Clear conversation memory</a>
    </div>
    <div id="photoPreview">
        <img id="photoPreviewImg" src="" alt="Captured photo">
        <div class="remove-photo" onclick="removePhoto()">Remove Photo</div>
    </div>
    <button onclick="askAtlas()">COMPARE AI MODELS</button>
    <div id="loading">Comparing AI models...</div>
    <div id="result"></div>
</div>
<div class="footer">Project Atlas • AI Comparison</div>

<div id="cameraModal">
    <div class="camera-box">
        <video id="cameraVideo" autoplay playsinline></video>
        <canvas id="cameraCanvas" style="display:none;"></canvas>
        <div class="camera-controls">
            <button onclick="capturePhoto()">📸 Capture</button>
            <button class="secondary" onclick="closeCamera()">Cancel</button>
        </div>
    </div>
</div>

<script>
let capturedImage = "";
let cameraStream = null;

async function openCamera() {
    const modal = document.getElementById("cameraModal");
    const video = document.getElementById("cameraVideo");

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" },
            audio: false
        });
        video.srcObject = cameraStream;
        modal.classList.add("open");
    } catch (error) {
        console.error("Camera error:", error);
        alert("Could not access the camera. Please check permissions and try again.");
    }
}

function closeCamera() {
    const modal = document.getElementById("cameraModal");
    const video = document.getElementById("cameraVideo");

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    video.srcObject = null;
    modal.classList.remove("open");
}

function capturePhoto() {
    const video = document.getElementById("cameraVideo");
    const canvas = document.getElementById("cameraCanvas");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    capturedImage = canvas.toDataURL("image/jpeg", 0.85);

    const preview = document.getElementById("photoPreview");
    const previewImg = document.getElementById("photoPreviewImg");
    previewImg.src = capturedImage;
    preview.style.display = "block";

    closeCamera();
}

function removePhoto() {
    capturedImage = "";
    const preview = document.getElementById("photoPreview");
    const previewImg = document.getElementById("photoPreviewImg");
    previewImg.src = "";
    preview.style.display = "none";
}

async function toggleMemory() {
    const memoryButton = document.getElementById("memoryButton");
    const clearLink = document.getElementById("memoryClearLink");

    try {
        const response = await fetch("/memory/toggle", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });

        const data = await response.json();

        if (data.memory_enabled) {
            memoryButton.innerText = "🧠 Memory: ON";
            memoryButton.classList.remove("memory-off");
            memoryButton.classList.add("memory-on");
            clearLink.classList.add("visible");
        } else {
            memoryButton.innerText = "🧠 Memory: OFF";
            memoryButton.classList.remove("memory-on");
            memoryButton.classList.add("memory-off");
            clearLink.classList.remove("visible");
        }
    } catch (error) {
        console.error("Memory toggle error:", error);
        alert("Could not update memory setting. Please try again.");
    }
}

async function clearMemory() {
    try {
        await fetch("/memory/clear", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });
        alert("Conversation memory cleared.");
    } catch (error) {
        console.error("Memory clear error:", error);
        alert("Could not clear memory. Please try again.");
    }
}

async function askAtlas() {
    const questionElement = document.getElementById("question");
    const styleElement = document.getElementById("style");
    const result = document.getElementById("result");
    const loading = document.getElementById("loading");
    const button = document.querySelector(".container > button");

    const question = questionElement ? questionElement.value.trim() : "";
    const style = styleElement ? styleElement.value : "balanced";

    if (!question) {
        alert("Please enter a question.");
        return;
    }

    if (button) {
        button.disabled = true;
        button.innerText = "⏳ Thinking...";
    }

    loading.style.display = "block";
    loading.innerText = "🤖 Comparing AI models...";
    result.innerHTML = "";

    try {
        const response = await fetch("/compare", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "text/html"
            },
            body: JSON.stringify({
                question: question,
                image: capturedImage,
                style: style
            })
        });

        const data = await response.text();

        if (!response.ok) {
            throw new Error(data || "Server returned an error.");
        }

        result.innerHTML = data;

    } catch (error) {
        console.error("Project Atlas error:", error);
        result.innerHTML = `
            <div class="card">
                <h3>⚠️ Something went wrong</h3>
                <p>${error.message}</p>
            </div>
        `;
    } finally {
        loading.style.display = "none";
        if (button) {
            button.disabled = false;
            button.innerText = "COMPARE AI MODELS";
        }
    }
}
</script>
</body>
</html>
"""
    memory_button_class = "memory-on" if memory_enabled else "memory-off"
    memory_clear_visible = "visible" if memory_enabled else ""

    html = html.replace("__USERNAME__", escape(username))
    html = html.replace("__MEMORY_LABEL__", memory_label)
    html = html.replace("__MEMORY_BUTTON_CLASS__", memory_button_class)
    html = html.replace("__MEMORY_CLEAR_VISIBLE__", memory_clear_visible)

    return html


# ============================================================
# TEST
# ============================================================

@app.route("/test", methods=["POST"])
def test():
    return request.get_data(as_text=True)


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000))
    )
