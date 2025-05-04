import os
import json
import time
import traceback
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import pipeline
from huggingface_hub import login  # トークン認証用

import torch
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from dotenv import load_dotenv  # ← 追加

# --- .envファイルの読み込み ---
load_dotenv()

# --- トークン認証（Hugging Face） ---
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(hf_token)
else:
    print("⚠️ HUGGINGFACE_TOKEN が .env に設定されていません")

# --- 設定 ---
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
print(f"使用モデル: {MODEL_NAME}")
model = None

# --- FastAPIアプリ作成 ---
app = FastAPI(title="FastAPI LLM API", description="Hugging Face TransformersベースのチャットAPI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- データ構造定義 ---
class Message(BaseModel):
    role: str
    content: str

class RequestBody(BaseModel):
    message: str
    conversationHistory: Optional[List[Message]] = []

class ResponseBody(BaseModel):
    success: bool
    response: str
    conversationHistory: List[Message]

# --- モデル読み込み ---
def load_model():
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")
    model = pipeline(
        "text-generation",
        model=MODEL_NAME,
        model_kwargs={"torch_dtype": torch.bfloat16} if device == "cuda" else {},
        device=device
    )
    print("モデル読み込み完了")

# --- アシスタント応答抽出関数 ---
def extract_response(outputs, prompt):
    if outputs and isinstance(outputs, list):
        text = outputs[0]["generated_text"]
        if prompt in text:
            return text.split(prompt)[-1].strip()
        else:
            return text.strip()
    return "応答を生成できませんでした。"

# --- APIエンドポイント ---
@app.get("/")
async def root():
    return {"message": "FastAPI LLM API is running!"}
@app.post("/chat", response_model=ResponseBody)
async def chat(request: RequestBody):
    global model
    try:
        if model is None:
            load_model()

        messages = request.conversationHistory or []
        messages.append({"role": "user", "content": request.message})

        prompt = ""
        for msg in messages:
            prompt += f"{msg['role']}: {msg['content']}\n"
        prompt += "assistant: "

        print("生成プロンプト:", prompt)

        outputs = model(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)
        assistant_reply = extract_response(outputs, prompt)

        messages.append({"role": "assistant", "content": assistant_reply})

        return {
            "success": True,
            "response": assistant_reply,
            "conversationHistory": messages
        }

    except Exception as e:
        print("エラー:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="モデルによる応答生成に失敗しました")

# --- 起動時にモデルをロード ---
@app.on_event("startup")
def on_startup():
    load_model()

# --- ngrokトンネル起動 ---
def run_with_ngrok(port=8000):
    nest_asyncio.apply()
    ngrok_token = os.getenv("NGROK_TOKEN")
    if not ngrok_token:
        print("⚠️ NGROK_TOKEN が .env に設定されていません")
        return
    ngrok.set_auth_token(ngrok_token)
    public_url = ngrok.connect(port).public_url
    print("🚀 公開URL:", public_url)
    print("📘 ドキュメント:", public_url + "/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)

# --- 実行ブロック ---
if __name__ == "__main__":
    run_with_ngrok(port=8000)
