#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Vision Pipeline Configuration
パイプラインの設定を管理するファイル
"""

# VLMモデル設定
VLM_MODELS = {
    "small": "Qwen/Qwen2-VL-2B-Instruct",      # 軽量モデル
    "medium": "Qwen/Qwen2.5-VL-3B-Instruct",   # 中サイズモデル
    "large": "Qwen/Qwen2.5-VL-7B-Instruct"     # 大サイズモデル
}

# デフォルト設定
DEFAULT_CONFIG = {
    "loop_interval": 10,                        # ループ間隔（秒）
    "vlm_model": VLM_MODELS["medium"],         # 使用するVLMモデル
    "vlm_max_tokens": 256,                     # VLMの最大トークン数
    "openai_model": "gpt-4.1",                   # OpenAIモデル
    "script_timeout": 30,                      # スクリプト実行タイムアウト（秒）
    "camera_topic": "/camera/image_raw/compressed",  # カメラトピック
    "image_save_dir": "pipeline_images",       # 画像保存ディレクトリ
    "script_save_dir": "generated_scripts",    # スクリプト保存ディレクトリ
}

# プロンプト設定
PROMPTS = {
    "vlm_prompt": """
   You are the visual-navigation agent for a simulated mobile robot.

    Goal  
    1. Reach the deep-green black trash can.  
    2. **Keep moving forward until the trash can occupies (≈) the entire field-of-view.**  
        • Task is complete only when ≥ 90 % of the image is covered by the can.

    If the can is visible, output one line telling the robot how to move toward it.  
    If not, you may need to command the robot to explore the environment.
    If the image is 90% dominated by brown wooden walls, you may need to command the robot to move backward and turn around.

    Output the details of the image and the command to the robot.
    """,
    
    "code_generation_prompt": """
    Your task is to generate a Python script to move the robot to the target position.
    Goal  
    1. Reach the deep-green black trash can.  
    2. **Keep moving forward until the trash can occupies (≈) the entire field-of-view.**  
        • Task is complete only when ≥ 90 % of the image is covered by the can.
    Generate **only** a complete Python script—no comments or explanations—that:

    1. Begin with `import rclpy`.
    2. Publish `geometry_msgs.msg.Twist` to `/cmd_vel` at 10 Hz (`time.sleep(0.1)`).
    3. Execute the given motion sequence *in order* (your choice of velocities).
    4. **Motions must last more than 2 seconds and less than 5 seconds.**
    5. After every segment, and again just before exit, publish a full-stop Twist  
    (all fields = 0.0) for ≥ 0.3 s (≥ 3 consecutive 10 Hz messages).
    6. Wrap the main logic in `try … finally`; in the `finally` block:  
    • send the same full-stop Twist for ≥ 0.3 s,  
    • then call `node.destroy_node()` and `rclpy.shutdown()`.
    7. Use only default QoS settings.
    8. Assign float values to every Twist field (e.g., `msg.linear.x = 0.0`).
    9. Output only runnable Python code—absolutely no additional text.
    
    The description of the environment is as follows:
    """
}

# ログ設定
LOGGING_CONFIG = {
    "log_level": "INFO",
    "save_logs": True,
    "log_file": "pipeline_logs.txt"
} 