---
title: Insight Bot
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 6.0.2
app_file: app.py
pinned: false
license: apache-2.0
short_description: 'Advanced Llama-based chatbot'
---




🤖 Advanced LLM - Insight Bo


Overview

This is an Advanced AI Chatbot deployed on Hugging Face Spaces. It is powered by Meta LLaMA-3.2-1B Instruct and includes:

RAG (Retrieval-Augmented Generation) pipeline using Wikipedia for real-time knowledge.

Enhanced tools: Calculator and Wikipedia search.

Conversation memory with trimming for long chats.

Multiple AI personalities for different interaction styles.

Interactive Gradio UI for seamless web access.

Open the app on HF Spaces

Features
🔹 RAG Pipeline

Retrieves information from Wikipedia to provide factual answers.

Maintains conversation history and summarizes older messages to stay efficient.

🔹 Enhanced Functionalities

calc: <expression> → Calculator tool

wiki: <topic> → Wikipedia search tool

Safety guard to block harmful or unsafe requests

🔹 AI Personalities

Default, Teacher, Friendly, Programmer, Strict, Child-Friendly Tutor

Adjusts response style and tone dynamically

🔹 Gradio Interface

Choose personality and custom instructions

Adjustable response temperature

Maintains conversation history per session

How to Use

Open the app in Hugging Face Spaces.

Select an AI Personality.

Optionally provide Custom System Instructions.

Type messages in the chat box or use the tools:

calc: 12*7 → 84

wiki: Quantum Mechanics → Short Wikipedia summary

Technical Details

Model: meta-llama/Llama-3.2-1B-Instruct

Quantization: 4-bit using BitsAndBytes for efficiency

Tokenizer: AutoTokenizer from Hugging Face

Tools: Wikipedia + Python calculator

Dependencies
gradio==5.50.0
torch
transformers
python-dotenv---
title: Insight Bot
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 6.0.2
app_file: app.py
pinned: false
license: apache-2.0
short_description: 'Advanced Llama-based chatbot'
---

# 🤖 Advanced LLM - Insight Bot

## Overview

This is an **Advanced AI Chatbot** deployed on **Hugging Face Spaces**. It is powered by **Meta LLaMA-3.2-1B Instruct** and includes:

- **RAG (Retrieval-Augmented Generation) pipeline** using Wikipedia for real-time knowledge.  
- **Enhanced tools**: Calculator and Wikipedia search.  
- **Conversation memory** with trimming for long chats.  
- **Multiple AI personalities** for different interaction styles.  
- **Interactive Gradio UI** for seamless web access.  

[Open the app on HF Spaces](https://huggingface.co/spaces/<your-username>/<space-name>)

---

## Features

### 🔹 RAG Pipeline
- Retrieves information from Wikipedia to provide factual answers.  
- Maintains conversation history and summarizes older messages to stay efficient.  

### 🔹 Enhanced Functionalities
- `calc: <expression>` → Calculator tool  
- `wiki: <topic>` → Wikipedia search tool  
- **Safety guard** to block harmful or unsafe requests  

### 🔹 AI Personalities
- Default, Teacher, Friendly, Programmer, Strict, Child-Friendly Tutor  
- Adjusts response style and tone dynamically  

### 🔹 Gradio Interface
- Choose personality and custom instructions  
- Adjustable response temperature  
- Maintains conversation history per session  

---

## How to Use

1. Open the app in **Hugging Face Spaces**.  
2. Select an **AI Personality**.  
3. Optionally provide **Custom System Instructions**.  
4. Type messages in the chat box or use the tools:  
   - `calc: 12*7` → `84`  
   - `wiki: Quantum Mechanics` → Short Wikipedia summary  

---

## Technical Details
- **Model**: `meta-llama/Llama-3.2-1B-Instruct`  
- **Quantization**: 4-bit using **BitsAndBytes** for efficiency  
- **Tokenizer**: AutoTokenizer from Hugging Face  
- **Tools**: Wikipedia + Python calculator  

---

## Dependencies
```text
gradio==5.50.0
torch
transformers
python-dotenv
wikipedia

wikipedia

Deployment

Hosted on Hugging Face Spaces

Environment variables such as HF_TOKEN are stored as Secrets for authentication

Gradio handles the web interface

License

Apache-2.0
