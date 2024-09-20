#!/bin/bash

LLM_HOST=${LLM_HOST:=0.0.0.0}
LLM_PORT=${LLM_PORT:=11434}
API_HOST=${API_HOST:=0.0.0.0}
API_PORT=${API_PORT:=42024}

fastapi run main.py --host=$API_HOST --port=$API_PORT
