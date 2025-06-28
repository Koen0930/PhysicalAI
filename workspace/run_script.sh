#!/bin/bash

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set."
    echo "Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

# Activate virtual environment and run the script
cd "$(dirname "$0")"
source .venv/bin/activate
python3 generate_and_run_python_script.py 