# lambda/index.py
import json
import os
import urllib.request
import urllib.error
import re

# ──────────────────────────────────────────────
# 1. 定数とユーティリティ
# ──────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "")      # CDK で渡した URL
TIMEOUT  = 10                                # 秒

def extract_region_from_arn(arn: str) -> str:
    """今回リージョンは使わないが、既存実装を残しておく"""
    m = re.search(r'arn:aws:lambda:([^:]+):', arn)
    return m.group(1) if m else "us-east-1"

def call_external_model(prompt: str) -> str:
    """Colab FastAPI へ JSON POST して応答文字列を返す"""
    if not API_URL:
        raise RuntimeError("API_URL environment variable is not set")

    payload = json.dumps({"text": prompt}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            # FastAPI が {"answer": "..."} を返す想定
            return data.get("answer") or str(data)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"External API HTTP {e.code}: {e.read().decode()}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"External API URL Error: {e.reason}")

# ──────────────────────────────────────────────
# 2. Lambda メインハンドラ
# ──────────────────────────────────────────────
def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))

        # --- リクエストを取り出す ---
        body = json.loads(event["body"])
        message = body["message"]
        conversation_history = body.get("conversationHistory", [])

        # --- Colab FastAPI へ投げる ---
        print("Sending to external model:", message)
        assistant_response = call_external_model(message)
        print("Assistant response:", assistant_response)

        # --- 会話履歴を更新 ---
        messages = conversation_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": assistant_response}
        ]

        # --- 正常レスポンス ---
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": messages
            })
        }

    except Exception as err:
        print("Error:", str(err))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(err)
            })
        }
