import json
import os
import re
import urllib.request
import boto3

# ---------- 共通ヘルパ ---------- #
def extract_region_from_arn(arn: str) -> str:
    """Lambda ARN からリージョン部分だけ抜き出す"""
    m = re.search(r"arn:aws:lambda:([^:]+):", arn)
    return m.group(1) if m else "us-east-1"

def build_resp(code: int, body: dict) -> dict:
    return {
        "statusCode": code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods": "OPTIONS,POST"
        },
        "body": json.dumps(body)
    }

# ---------- Bedrock 呼び出し ---------- #
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")
_bedrock = None

def call_bedrock(prompt: str, history, context) -> str:
    global _bedrock
    if _bedrock is None:
        region = extract_region_from_arn(context.invoked_function_arn)   # ★ ここを修正
        _bedrock = boto3.client("bedrock-runtime", region_name=region)

    msgs = history + [{"role": "user", "content": prompt}]
    bedrock_msgs = [
        {"role": m["role"], "content": [{"text": m["content"]}]}
        for m in msgs
    ]
    payload = {
        "messages": bedrock_msgs,
        "inferenceConfig": {
            "maxTokens": 512,
            "temperature": 0.7,
            "topP": 0.9,
            "stopSequences": []
        }
    }
    res = _bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json"
    )
    body = json.loads(res["body"].read())
    return body["output"]["message"]["content"][0]["text"]

# ---------- 外部モデル API 呼び出し（任意） ---------- #
PREDICT_URL = os.environ.get("https://8753-34-106-21-190.ngrok-free.app/predict")

def call_external_api(prompt: str) -> str:
    if not PREDICT_URL:
        raise RuntimeError("PREDICT_URL not set")
    data = json.dumps({"text": prompt}).encode()
    req = urllib.request.Request(
        PREDICT_URL, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["answer"]

# ---------- Lambda ハンドラ ---------- #
def lambda_handler(event, context):
    try:
        body    = json.loads(event.get("body", "{}"))
        prompt  = body["message"]
        history = body.get("conversationHistory", [])

        if PREDICT_URL:                 # 外部 API 優先
            reply = call_external_api(prompt)
        else:                           # Bedrock を使用
            reply = call_bedrock(prompt, history, context)

        history.append({"role": "assistant", "content": reply})
        return build_resp(200, {"success": True,
                                "response": reply,
                                "conversationHistory": history})
    except Exception as e:
        print("❌ Error:", repr(e))
        return build_resp(500, {"success": False, "error": str(e)})
