from openai import OpenAI
import subprocess
import os
import sys

# OpenAI クライアントの初期化
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    return OpenAI(api_key=api_key)

# 事前promptをprompt.txtから読み込む
def load_pre_prompt():
    from prompt import write_code_prompt
    return write_code_prompt

# ChatGPT APIでpromptを入力して返信を受け取る
def get_chat_response(client, prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        sys.exit(1)

# 返信の内容を.pyファイルに書いて保存する
def generate_python_script(res):
    python_code = clean_code_blocks(res)
    with open("generated_script.py", "w", encoding="utf-8") as f:
        f.write(python_code)
    print("Generated script saved as 'generated_script.py'")

# コードブロックの文字列を削除する関数
def clean_code_blocks(code_text):
    """
    ChatGPTの応答からコードブロックのマークダウン記法を削除する
    ```python や ``` などを削除
    """
    lines = code_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # コードブロックの開始・終了記号を除外
        if line.strip().startswith('```'):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

# subprocessでpythonスクリプトを実行する
def run_python_script():
    print("Running generated script...")
    try:
        result = subprocess.run(["python3", "generated_script.py"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running script: {result.stderr}")
        else:
            print("Script executed successfully")
            if result.stdout:
                print(f"Output: {result.stdout}")
    except Exception as e:
        print(f"Error executing script: {e}")

def main():
    # OpenAI クライアントの初期化
    client = get_openai_client()
    
    # 事前プロンプトの読み込み
    pre_prompt = load_pre_prompt()
    
    # ユーザーからの指示を入力
    print("Enter your robot control instruction (in English for better accuracy):")
    user_instruction = input("> ")
    
    if not user_instruction.strip():
        print("No instruction provided. Exiting.")
        sys.exit(1)
    
    # 完全なプロンプトを作成
    full_prompt = pre_prompt + "\n" + user_instruction
    
    print("Sending request to ChatGPT...")
    print(f"Instruction: {user_instruction}")
    
    # ChatGPTから応答を取得
    response = get_chat_response(client, full_prompt)
    
    # Pythonスクリプトを生成
    generate_python_script(response)
    
    # 生成されたスクリプトを実行
    run_python_script()

if __name__ == "__main__":
    main() 