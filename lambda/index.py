import json
import os
import re
import urllib.request
import urllib.error
import boto3
from botocore.exceptions import ClientError

# ========================== ① 共通ユーティリティ ========================== #
def extract_region_from_arn(arn: str) -> str:
    """Lambda ARN からリージョンを取得（例: arn:aws:lambda:us-east-1:...）"""
    m = re.search(r"arn:aws:lambda:([^:]+):", arn)
    return m.group(1) if m else "us-east-1"

def build_response(status: int, body: dict) -> dict:
    """CORS 付き API Gateway レスポンス"""
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods": "OPTIONS,POST"
        },
        "body": json.dumps(body)
    }

# ========================== ② Bedrock 設定 & 呼び出し ====================== #
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")
_bedrock_client = None

def call_bedrock(prompt: str, history=None) -> str:
    """Bedrock (Nova Lite など) を呼び出す"""
    global _bedrock_client
    if _bedrock_client is None:
        # コンテキスト内リージョンを動的取得
        _bedrock_client = boto3.client("bedrock-runtime",
                                       region_name=extract_region_from_arn(
                                           os.environ["AWS_LAMBDA_FUNCTION_ARN"]
                                       ))

    messages = history or []
    messages.append({"role": "user", "content": prompt})

    # Bedrock のメッセージ形式に変換
    bedrock_messages = [
        {
            "role": m["role"],
            "content": [{"text": m["content"]}]
        } for m in messages
    ]

    payload = {
        "messages": bedrock_messages,
        "inferenceConfig": {
            "maxTokens": 512,
            "temperature": 0.7,
            "topP": 0.9,
            "stopSequences": []
        }
    }

    res = _bedrock_client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json"
    )
    body = json.loads(res["body"].read())
    return body["output"]["message"]["content"][0]["text"]

# ========================== ③ 外部モデル API 呼び出し ======================= #
PREDICT_URL = os.environ.get("https://68f9-34-106-21-190.ngrok-free.app/predict")   # 例: https://xxxx.ngrok-free.app/predict

def call_external_api(prompt: str) -> str:
    """FastAPI などで公開した任意モデルを呼ぶ"""
    if not PREDICT_URL:
        raise RuntimeError("PREDICT_URL not set")

    req_body = json.dumps({"text": prompt}).encode("utf-8")
    req = urllib.request.Request(
        PREDICT_URL,
        data=req_body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        data = json.loads(res.read())
        # 期待するキー名に合わせて取り出す
        return data["answer"]

# ========================== ④ Lambda ハンドラ ============================= #
def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        user_msg = body["message"]
        history  = body.get("conversationHistory", [])

        # ★ 外部 API が設定されていればそちらを優先
        if PREDICT_URL:
            assistant_reply = call_external_api(user_msg)
        else:
            assistant_reply = call_bedrock(user_msg, history)

        history.append({"role": "assistant", "content": assistant_reply})
        return build_response(200, {
            "success": True,
            "response": assistant_reply,
            "conversationHistory": history
        })

    except Exception as e:
        # CloudWatch Logs でトレースしやすいようにスタックも出力
        print("❌ Error:", repr(e))
        return build_response(500, {"success": False, "error": str(e)})
