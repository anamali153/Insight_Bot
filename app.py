# -*- coding: utf-8 -*-
"""AI_chatbot.ipynb"""

import os
import torch
import transformers
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

import wikipedia
import math
import gradio as gr

# -----------------------------
# Load Hugging Face token
# -----------------------------
load_dotenv()  # if you store token in .env
login(token=os.getenv("HF_TOKEN"))

# -----------------------------
# Conversation trimming
# -----------------------------
def trim_history(messages, max_length=3000):
    text = "".join([m["content"] for m in messages])
    if len(text) < max_length:
        return messages
    # Optional: replace with your summary API if needed
    return messages[-5:]

# -----------------------------
# Personalities
# -----------------------------
PERSONALITIES = {
    "default": "You are a normal helpful assistant.",
    "Teacher": "You explain concepts with simple examples.",
    "Friendly": "You speak like a warm, supportive friend.",
    "Programmer": "You answer like a senior software engineer.",
    "Strict": "Give short, factual answers only.",
    "Child-Friendly Tutor": "Explain everything like you are teaching a 7-year-old child."
}

# -----------------------------
# Safety guard
# -----------------------------
def safety_guard(user_text):
    blocked_words = ["kill", "harm", "suicide", "bomb"]
    for w in blocked_words:
        if w in user_text.lower():
            return "❗ I cannot help with harmful or dangerous requests."
    return None

# -----------------------------
# Tools
# -----------------------------
def tool_calculator(expr):
    try:
        return str(eval(expr))
    except:
        return "Invalid mathematical expression."

def tool_wiki(query):
    try:
        return wikipedia.summary(query, sentences=3)
    except:
        return "No Wikipedia results found."

# -----------------------------
# Device and model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
LLAMA = "meta-llama/Llama-3.2-1B-Instruct"


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
    llm_int8_enable_fp32_cpu_offload=True
)


tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token

def extract_text(decoded: str) -> str:
    if "<|start_header_id|>assistant<|end_header_id|>" in decoded:
        assistant_part = decoded.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        return assistant_part.split("<|eot_id|>")[0].strip()
    return decoded.strip()

# -----------------------------
# Main response function
# -----------------------------
def generate_response(system_prompt, personality, user_prompt, temperature, history):
    # Safety check
    safe = safety_guard(user_prompt)
    if safe:
        return safe, history

    # Tools
    if user_prompt.lower().startswith("calc:"):
        result = tool_calculator(user_prompt[5:])
        history.append((user_prompt, result))
        return result, history

    if user_prompt.lower().startswith("wiki:"):
        result = tool_wiki(user_prompt[5:])
        history.append((user_prompt, result))
        return result, history

    # Build messages
    messages = []
    full_system_prompt = PERSONALITIES.get(personality, PERSONALITIES["default"]) + "\n" + system_prompt
    messages.append({"role": "system", "content": full_system_prompt})

    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": user_prompt})
    messages = trim_history(messages)

    # Tokenize
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Generate
    output_ids = model.generate(
        inputs,
        max_new_tokens=300,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output_ids[0])
    reply = extract_text(decoded)
    history.append((user_prompt, reply))
    return reply, history

# -----------------------------
# COMMENT OUT TERMINAL LOOP
# -----------------------------
"""
system_prompt = "You are a helpful assistant."
personality = "default"
temperature = 0.7
history = []
messages = []
prompt = ""

while prompt != "quit":
    prompt = input("User: ")
    if prompt == "quit":
        break
    generated_text, history = generate_response(
        system_prompt=system_prompt,
        personality=personality,
        user_prompt=prompt,
        temperature=temperature,
        history=history
    )
    print(f"\nAssistant: {generated_text}\n")
"""

# -----------------------------
# Gradio App
# -----------------------------
with gr.Blocks(title="Advanced LLM Chatbot") as demo:
    gr.Markdown("# 🤖 Advanced AI Assistant")

    personality = gr.Dropdown(
        list(PERSONALITIES.keys()),
        value="Friendly",
        label="Choose AI Personality"
    )

    system_prompt = gr.Textbox(
        label="Custom System Instructions (optional)",
        placeholder="e.g., 'Explain everything with examples'"
    )

    user_input = gr.Textbox(label="Your Message")
    temperature = gr.Slider(0.0, 1.0, value=0.4, label="Temperature")
    output = gr.Textbox(label="Response")
    history = gr.State([])

    btn = gr.Button("Send")

    btn.click(
        generate_response,
        inputs=[system_prompt, personality, user_input, temperature, history],
        outputs=[output, history]
    )

demo.launch()
