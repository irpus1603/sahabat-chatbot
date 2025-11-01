#!/bin/bash

# Test script for Ollama streaming API
# This tests the exact payload format that the application sends

ENDPOINT_URL="http://ollama-route-open-webui.apps.iohairan.nokia-airan.ioh.com/api/generate"
MODEL="gemma2-9b-cpt-sahabatai-v1-instruct:Q4_K_M"
TEMPERATURE=0.7
MAX_TOKENS=200

echo "=========================================="
echo "Testing Ollama Streaming API"
echo "=========================================="
echo "Endpoint: $ENDPOINT_URL"
echo "Model: $MODEL"
echo ""

# Test 1: Simple streaming request
echo "TEST 1: Simple Streaming Request"
echo "---"

curl -X POST "$ENDPOINT_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL'",
    "prompt": "user: Hello, how are you?\nassistant: I am doing well, thank you for asking.",
    "temperature": '$TEMPERATURE',
    "max_tokens": '$MAX_TOKENS',
    "stream": true
  }' \
  -v

echo ""
echo ""

# Test 2: With conversation context
echo "TEST 2: Streaming with Conversation Context"
echo "---"

curl -X POST "$ENDPOINT_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL'",
    "prompt": "user: What is the capital of France?\nassistant: The capital of France is Paris.\nuser: Tell me more about Paris.",
    "temperature": '$TEMPERATURE',
    "max_tokens": '$MAX_TOKENS',
    "stream": true
  }' \
  -v

echo ""
echo ""

# Test 3: Non-streaming for comparison
echo "TEST 3: Non-Streaming Request (for comparison)"
echo "---"

curl -X POST "$ENDPOINT_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL'",
    "prompt": "user: What is 2+2?",
    "temperature": '$TEMPERATURE',
    "max_tokens": '$MAX_TOKENS',
    "stream": false
  }' \
  -v

echo ""
echo "=========================================="
echo "Tests completed!"
echo "=========================================="
