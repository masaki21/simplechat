#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { BedrockChatbotStack } from '../lib/bedrock-chatbot-stack';

const app = new cdk.App();
const externalApiUrl = 'https://9fc8-34-125-85-61.ngrok-free.app/predict';
new BedrockChatbotStack(app, 'BedrockChatbotStack', {
  // モデルIDをオプションで指定可能
  modelId: 'us.amazon.nova-lite-v1:0',
  //modelId: 'us.amazon.nova-micro-v1:0',
  apiUrl: externalApiUrl,
  
  // 環境変数から取得したリージョンを使用、またはデフォルトとしてus-east-1を使用
  env: { 
    account: process.env.CDK_DEFAULT_ACCOUNT, 
    region: process.env.CDK_DEFAULT_REGION || 'us-east-1'
  },
});

// タグはスタックレベルで追加
cdk.Tags.of(app).add('Project', 'BedrockChatbot');
cdk.Tags.of(app).add('Environment', 'Dev');
