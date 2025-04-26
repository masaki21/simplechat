import json, os, urllib.request

PREDICT_URL = os.environ.get("PREDICT_URL")  # Lambda の環境変数に設定

def call_custom_api(msg: str):
    req_body = json.dumps({"text": msg}).encode("utf-8")
    req = urllib.request.Request(
        url=PREDICT_URL,
        data=req_body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        res_body = res.read()
        return json.loads(res_body)["answer"]

def lambda_handler(event, context):
    body = json.loads(event["body"])
    user_msg = body["message"]
    assistant = call_custom_api(user_msg)

    return {
        "statusCode": 200,
        "headers": { "Content-Type": "application/json",
                     "Access-Control-Allow-Origin": "*"},
        "body": json.dumps({"success": True,
                            "response": assistant})
    }
