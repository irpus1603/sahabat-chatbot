#!/bin/bash

# Simple test for Ollama streaming API

curl -X POST "http://ollama-route-open-webui.apps.iohairan.nokia-airan.ioh.com/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma2-9b-cpt-sahabatai-v1-instruct:Q4_K_M",
    "prompt": "user: create a story about jakarta in summer",
    "temperature": 0.7,
    "max_tokens": 200,
    "stream": true
  }'
