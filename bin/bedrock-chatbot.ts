#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { BedrockChatbotStack, BedrockChatbotStackProps } from '../lib/bedrock-chatbot-stack';

// ======== ★ ここだけ書き換える値 ★ =========================
const PREDICT_URL = 'https://67ad-34-106-21-190.ngrok-free.app/predict'; // ← Colab/ngrok URL
const MODEL_ID    = 'us.amazon.nova-lite-v1:0';                          // ← Bedrock fallback
// ===========================================================

const app = new cdk.App();

const props: BedrockChatbotStackProps = {
  modelId: MODEL_ID,
  predictUrl: PREDICT_URL,
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region : process.env.CDK_DEFAULT_REGION || 'us-east-1',
  },
};

new BedrockChatbotStack(app, 'BedrockChatbotStack', props);

cdk.Tags.of(app).add('Project', 'BedrockChatbot');
cdk.Tags.of(app).add('Environment', 'Dev');

