import os
import json

import gradio as gr
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from ddgs import DDGS


# ==========================================
# KI
# ==========================================

HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)

MODEL = "openai/gpt-oss-120b"


# ==========================================
# MEMORY
# ==========================================

memory = []


def merke(text):
    memory.append(text)
    return "Erinnerung gespeichert."


def zeige_memory():
    if not memory:
        return "Noch keine Erinnerungen gespeichert."

    return "\n".join(memory)


# ==========================================
# RECHNER
# ==========================================

def rechner(a, b, operation):

    if operation == "plus":
        return a + b

    if operation == "minus":
        return a - b

    if operation == "mal":
        return a * b

    if operation == "geteilt":
        if b == 0:
            return "Division durch 0 ist nicht möglich."

        return a / b

    return "Unbekannte Operation."


# ==========================================
# INTERNET
# ==========================================

def internet_suche(suchbegriff):

    ergebnisse = DDGS().text(
        suchbegriff,
        max_results=5
    )

    text = ""

    for ergebnis in ergebnisse:

        text += (
            f"Titel: {ergebnis.get('title')}\n"
            f"Info: {ergebnis.get('body')}\n"
            f"Link: {ergebnis.get('href')}\n\n"
        )

    return text


# ==========================================
# TOOLS
# ==========================================

tools = [

    {
        "type": "function",
        "function": {
            "name": "rechner",
            "description": "Berechnet zwei Zahlen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "operation": {
                        "type": "string",
                        "enum": [
                            "plus",
                            "minus",
                            "mal",
                            "geteilt"
                        ]
                    }
                },
                "required": [
                    "a",
                    "b",
                    "operation"
                ]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "internet_suche",
            "description": "Sucht aktuelle Informationen im Internet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "suchbegriff": {
                        "type": "string"
                    }
                },
                "required": ["suchbegriff"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "merke",
            "description": "Speichert eine Information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "zeige_memory",
            "description": "Liest gespeicherte Erinnerungen.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


# ==========================================
# AGENT
# ==========================================

def agent(frage):

    messages = [
        {
            "role": "system",
            "content": """
Du bist Nova, ein persönlicher AI-Agent.

Antworte auf Deutsch.

Du kannst:
- Fragen beantworten
- rechnen
- aktuelle Informationen im Internet suchen
- Informationen speichern
- gespeicherte Informationen abrufen

Nutze deine Werkzeuge selbstständig.
Für aktuelle Informationen sollst du die Internetsuche benutzen.
"""
        },

        {
            "role": "user",
            "content": frage
        }
    ]


    for schritt in range(5):

        antwort = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = antwort.choices[0].message


        if not message.tool_calls:
            return message.content


        tool_call = message.tool_calls[0]

        tool_name = tool_call.function.name

        args = json.loads(
            tool_call.function.arguments
        )


        if tool_name == "rechner":

            ergebnis = rechner(
                args["a"],
                args["b"],
                args["operation"]
            )


        elif tool_name == "internet_suche":

            ergebnis = internet_suche(
                args["suchbegriff"]
            )


        elif tool_name == "merke":

            ergebnis = merke(
                args["text"]
            )


        elif tool_name == "zeige_memory":

            ergebnis = zeige_memory()


        else:

            ergebnis = "Unbekanntes Werkzeug."


        messages.append({
            "role": "assistant",
            "content":
                f"Ergebnis von {tool_name}:\n{ergebnis}"
        })


    return "Ich konnte die Aufgabe nicht vollständig abschließen."


# ==========================================
# API FÜR DIE ANDROID-APP
# ==========================================

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
def chat_api(request: ChatRequest):

    try:

        antwort = agent(request.message)

        return {
            "reply": antwort
        }

    except Exception as fehler:

        return {
            "reply": f"Fehler: {str(fehler)}"
        }


# ==========================================
# WEB-OBERFLÄCHE
# ==========================================

def web_chat(message, history):

    try:
        return agent(message)

    except Exception as fehler:
        return f"Fehler: {fehler}"


demo = gr.ChatInterface(
    fn=web_chat,
    title="🤖 Nova",
    description="Mein persönlicher AI-Agent"
)


app = gr.mount_gradio_app(
    app,
    demo,
    path="/"
)


# ==========================================
# START
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 7860)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
