# -*- coding: utf-8 -*-
"""
Created on Sat May 24 03:56:00 2025

@author: emrek
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

# Model tanımlama
model_id = "Qwen/Qwen1.5-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print("💬 Qwen Chatbot (Çıkmak için 'çık' yazın)\n")

history = []

while True:
    user_input = input("👤 Sen: ")
    if user_input.lower() in ["çık", "exit", "quit"]:
        print("🤖 Qwen: Görüşmek üzere!")
        break

    # Geçmişle birlikte sohbeti oluştur
    chat = history + [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # Tokenlara dönüştür ve modele ver
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Cevap üret
    output = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05
    )

    # Cevabı ayıkla
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded.split("Assistant:")[-1].strip()

    # Terminale yaz
    print(f"🤖 Qwen: {response}")

    # Geçmişe ekle
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
