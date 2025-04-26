import json
import os
import re
import requests

# Lambda コンテキストからリージョンを抽出する関数（今回は不要だけど、残してOK）
def extract_region_from_arn(arn):
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    return "us-east-1"

# FastAPIサーバーのエンドポイントURL
FASTAPI_URL = "https://あなたのngrokのURL.ngrok-free.app/predict"  # 自分の最新URLをここに！

def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))
        
        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])
        
        print("Processing message:", message)
        
        # 会話履歴を使ってまとめたプロンプトを作る
        messages = conversation_history.copy()
        messages.append({
            "role": "user",
            "content": message
        })

        # FastAPIサーバーに送るテキストを作成
        merged_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        # FastAPI推論APIにPOSTリクエストを送る
        response = requests.post(
            FASTAPI_URL,
            json={"text": merged_prompt}
        )

        if response.status_code == 200:
            result = response.json()
            assistant_response = result.get("answer", "No answer returned.")

            # 応答を会話履歴に追加
            messages.append({
                "role": "assistant",
                "content": assistant_response
            })

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
        else:
            return {
                "statusCode": response.status_code,
                "body": json.dumps({
                    "success": False,
                    "error": "Failed to get prediction"
                })
            }
        
    except Exception as error:
        print("Error:", str(error))
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
                "error": str(error)
            })
        }

