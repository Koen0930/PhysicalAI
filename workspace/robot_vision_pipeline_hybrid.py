#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Vision Pipeline (Hybrid Version)
ROS2ロボットのカメラから画像を取得→VLM解析→Geminiでコード生成→実行の統合パイプライン
QwenまたはGeminiを選択可能
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import os
import sys
import time
import threading
import subprocess
from datetime import datetime
from pathlib import Path

# 外部モジュールのインポート
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None
    print("Warning: Gemini not available. Only Qwen mode will work.")

try:
    from qwen_image_inference import QwenImageInference
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    QwenImageInference = None
    print("Warning: Qwen not available. Only Gemini mode will work.")

from pipeline_config import DEFAULT_CONFIG, PROMPTS


class RobotVisionPipelineHybrid(Node):
    def __init__(self, loop_interval=10, model_path="Qwen/Qwen2.5-VL-3B-Instruct", device="auto", experiment_id=None, vlm_mode="qwen"):
        super().__init__('robot_vision_pipeline_hybrid')
        
        # パラメータ設定
        self.loop_interval = loop_interval  # ループ間隔（秒）
        self.model_path = model_path
        self.device = device
        self.vlm_mode = vlm_mode  # "qwen" または "gemini"
        
        # 実験IDの生成（指定されない場合は現在時刻を使用）
        if experiment_id is None:
            self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.experiment_id = experiment_id
        
        # 実験ごとのディレクトリ作成
        self.save_dir = f"experiments/{self.experiment_id}/images"
        self.script_dir = f"experiments/{self.experiment_id}/scripts"
        
        # ディレクトリ作成
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.script_dir, exist_ok=True)
        
        # カメラ画像サブスクライバー
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.compressed_image_callback,
            10)
        
        # 最新の画像データ
        self.latest_image = None
        self.image_count = 0
        self.pipeline_count = 0  # パイプライン実行回数
        
        # 行動履歴
        self.action_history = []  # VLMの出力履歴
        self.max_history = 3      # 保持する履歴の最大数
        
        # VLMモードの検証と初期化
        if self.vlm_mode == "gemini" and not GEMINI_AVAILABLE:
            self.get_logger().error("Gemini mode selected but Gemini is not available!")
            sys.exit(1)
        elif self.vlm_mode == "qwen" and not QWEN_AVAILABLE:
            self.get_logger().error("Qwen mode selected but Qwen is not available!")
            sys.exit(1)
        
        # クライアント初期化
        self.gemini_client = None
        self.vlm_inference = None
        
        if self.vlm_mode == "gemini":
            self.gemini_client = self.get_gemini_client()
        elif self.vlm_mode == "qwen":
            self.vlm_inference = QwenImageInference(model_path=self.model_path, device=self.device, lazy_load=True)
        
        # パイプライン制御
        self.pipeline_running = False
        self.pipeline_thread = None
        
        self.get_logger().info('Robot Vision Pipeline (Hybrid) initialized.')
        self.get_logger().info(f'VLM Mode: {self.vlm_mode}')
        self.get_logger().info(f'Experiment ID: {self.experiment_id}')
        self.get_logger().info(f'Loop interval: {self.loop_interval} seconds')
        self.get_logger().info(f'Device: {self.device}')
        self.get_logger().info(f'Images: {self.save_dir}')
        self.get_logger().info(f'Scripts: {self.script_dir}')
        self.get_logger().info('Commands:')
        self.get_logger().info('- Type "start" to begin the pipeline')
        self.get_logger().info('- Type "stop" to stop the pipeline')
        self.get_logger().info('- Type "once" to run pipeline once')
        self.get_logger().info('- Type "status" to check pipeline status')
        self.get_logger().info('- Type "quit" to exit')

    def get_gemini_client(self):
        """Gemini クライアントの初期化"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.get_logger().error("GEMINI_API_KEY environment variable not set.")
            self.get_logger().error("Please set your Gemini API key:")
            self.get_logger().error("export GEMINI_API_KEY='your-api-key-here'")
            sys.exit(1)
        
        return genai.Client(api_key=api_key)

    def compressed_image_callback(self, msg):
        """圧縮画像コールバック"""
        try:
            # 圧縮画像データをnumpy配列に変換
            np_arr = np.frombuffer(msg.data, np.uint8)
            
            # JPEGデコード
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                self.latest_image = cv_image
            else:
                self.get_logger().error('Failed to decode compressed image')
                
        except Exception as e:
            self.get_logger().error(f'Error processing compressed image: {e}')

    def save_current_image(self):
        """現在の画像を保存"""
        if self.latest_image is None:
            self.get_logger().warning('No image available')
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_dir}/image_{timestamp}_{self.pipeline_count:03d}.jpg"
        
        success = cv2.imwrite(filename, self.latest_image)
        if success:
            self.image_count += 1
            self.get_logger().info(f'Image saved: {filename}')
            return filename
        else:
            self.get_logger().error(f'Failed to save image: {filename}')
            return None

    def analyze_image_with_vlm(self, image_path):
        """VLMで画像を解析（QwenまたはGemini）"""
        try:
            if self.vlm_mode == "qwen":
                return self.analyze_with_qwen(image_path)
            elif self.vlm_mode == "gemini":
                return self.analyze_with_gemini(image_path)
            else:
                self.get_logger().error(f'Unknown VLM mode: {self.vlm_mode}')
                return None
        except Exception as e:
            self.get_logger().error(f'VLM analysis failed: {e}')
            return None

    def analyze_with_qwen(self, image_path):
        """Qwenで画像解析"""
        self.get_logger().info('Analyzing image with Qwen...')
        
        # プロンプトに行動履歴を追加
        enhanced_prompt = self.build_enhanced_prompt(PROMPTS["vlm_prompt"])
        
        result = self.vlm_inference.infer_image(
            image_path, 
            prompt=enhanced_prompt,
            max_tokens=DEFAULT_CONFIG["vlm_max_tokens"]
        )
        self.get_logger().info(f'Qwen analysis result: {result[:100]}...')
        
        # 行動履歴を更新
        self.update_action_history(result)
        
        # 結果をテキストファイルに保存
        self.save_vlm_result(result, image_path, "Qwen")
        return result

    def analyze_with_gemini(self, image_path):
        """Geminiで画像解析"""
        self.get_logger().info('Analyzing image with Gemini...')
        
        # 画像ファイルをバイトとして読み込み
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # プロンプトに行動履歴を追加
        enhanced_prompt = self.build_enhanced_prompt(PROMPTS["vlm_prompt"])
        
        # Geminiで画像解析
        response = self.gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                enhanced_prompt
            ]
        )
        
        result = response.text
        self.get_logger().info(f'Gemini analysis result: {result[:100]}...')
        
        # 行動履歴を更新
        self.update_action_history(result)
        
        # 結果をテキストファイルに保存
        self.save_vlm_result(result, image_path, "Gemini")
        return result

    def build_enhanced_prompt(self, base_prompt):
        """行動履歴を含む拡張プロンプトを構築"""
        if not self.action_history:
            return base_prompt
        
        history_text = "\n\nPrevious Actions History:\n"
        for i, action in enumerate(self.action_history, 1):
            history_text += f"{i}. {action}\n"
        
        history_text += "\nConsidering the above action history, analyze the current situation and decide the next action."
        
        return base_prompt + history_text

    def update_action_history(self, new_action):
        """行動履歴を更新"""
        self.action_history.append(new_action)
        
        # 最大履歴数を超えた場合、古いものを削除
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
        
        self.get_logger().info(f'Action history updated. Current history length: {len(self.action_history)}')

    def build_code_generation_prompt(self, current_vlm_result):
        """コード生成用のプロンプトを構築（行動履歴を含む）"""
        base_prompt = PROMPTS["code_generation_prompt"]
        
        if len(self.action_history) > 1:  # 現在の結果を除く過去の履歴がある場合
            history_text = "\n\nPrevious Actions Context:\n"
            # 現在の結果を除く過去の履歴
            for i, action in enumerate(self.action_history[:-1], 1):
                history_text += f"Action {i}: {action}\n"
            
            history_text += f"\nCurrent Analysis: {current_vlm_result}\n"
            history_text += "\nGenerate robot control code considering the action sequence and current situation."
            
            return base_prompt + history_text + "\n" + current_vlm_result
        else:
            return base_prompt + "\n" + current_vlm_result

    def save_vlm_result(self, vlm_result, image_path, model_type):
        """VLMの解析結果をテキストファイルに保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vlm_filename = f"{self.script_dir}/{model_type.lower()}_analysis_{timestamp}_{self.pipeline_count:03d}.txt"
            
            with open(vlm_filename, "w", encoding="utf-8") as f:
                f.write(f"{model_type} Analysis Result\n")
                f.write(f"{'=' * (len(model_type) + 16)}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Image: {image_path}\n")
                if model_type == "Qwen":
                    f.write(f"Model: {self.model_path}\n")
                    f.write(f"Device: {self.device}\n")
                else:
                    f.write(f"Model: gemini-2.5-flash\n")
                f.write(f"Pipeline Count: {self.pipeline_count}\n")
                f.write(f"Action History Length: {len(self.action_history)}\n")
                f.write(f"\nPrompt:\n{PROMPTS['vlm_prompt']}\n")
                
                # 行動履歴の保存
                if self.action_history:
                    f.write(f"\nAction History:\n")
                    for i, action in enumerate(self.action_history, 1):
                        f.write(f"  {i}. {action}\n")
                
                f.write(f"\n{model_type} Output:\n{vlm_result}\n")
            
            self.get_logger().info(f'{model_type} result saved: {vlm_filename}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to save {model_type} result: {e}')

    def generate_code_with_gemini(self, vlm_result):
        """Geminiでロボット制御コードを生成（行動履歴は参照しない）"""
        try:
            self.get_logger().info('Generating code with Gemini...')
            
            # 行動履歴を含まない純粋なコード生成プロンプトを使用
            full_prompt = PROMPTS["code_generation_prompt"] + "\n" + vlm_result
            
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
            )
            
            generated_code = response.text.strip()
            self.get_logger().info('Code generated successfully')
            return generated_code
            
        except Exception as e:
            self.get_logger().error(f'Code generation failed: {e}')
            return None

    def clean_code_blocks(self, code_text):
        """コードブロックのマークダウン記法を削除"""
        lines = code_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if line.strip().startswith('```'):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def save_and_run_script(self, generated_code):
        """生成されたスクリプトを保存して実行"""
        try:
            # コードをクリーンアップ
            clean_code = self.clean_code_blocks(generated_code)
            
            # スクリプトファイルに保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            script_filename = f"{self.script_dir}/robot_script_{timestamp}_{self.pipeline_count:03d}.py"
            
            with open(script_filename, "w", encoding="utf-8") as f:
                f.write(clean_code)
            
            self.get_logger().info(f'Script saved: {script_filename}')
            
            # スクリプトを実行
            self.get_logger().info('Executing generated script...')
            result = subprocess.run(
                ["python3", script_filename], 
                capture_output=True, 
                text=True,
                timeout=DEFAULT_CONFIG["script_timeout"]
            )
            
            if result.returncode != 0:
                self.get_logger().error(f'Script execution failed: {result.stderr}')
            else:
                self.get_logger().info('Script executed successfully')
                if result.stdout:
                    self.get_logger().info(f'Script output: {result.stdout}')
                    
        except subprocess.TimeoutExpired:
            self.get_logger().warning('Script execution timed out')
        except Exception as e:
            self.get_logger().error(f'Error executing script: {e}')

    def run_pipeline_once(self):
        """パイプラインを1回実行"""
        self.get_logger().info(f'=== Running Pipeline Once ({self.vlm_mode.upper()}) ===')
        
        # パイプライン実行回数をインクリメント
        self.pipeline_count += 1
        self.get_logger().info(f'Pipeline execution #{self.pipeline_count}')
        
        # Step 1: 画像を保存
        image_path = self.save_current_image()
        if image_path is None:
            self.get_logger().error('Failed to save image. Skipping pipeline.')
            return False
        
        # Step 2: VLMで画像解析
        vlm_result = self.analyze_image_with_vlm(image_path)
        if vlm_result is None:
            self.get_logger().error('VLM analysis failed. Skipping pipeline.')
            return False
        
        # Step 3: Geminiでコード生成
        generated_code = self.generate_code_with_gemini(vlm_result)
        if generated_code is None:
            self.get_logger().error('Code generation failed. Skipping pipeline.')
            return False
        
        # Step 4: コード実行
        self.save_and_run_script(generated_code)
        
        self.get_logger().info('=== Pipeline Completed ===')
        return True

    def pipeline_loop(self):
        """パイプラインのループ実行"""
        while self.pipeline_running:
            try:
                self.run_pipeline_once()
                
                # 次のループまで待機
                self.get_logger().info(f'Waiting {self.loop_interval} seconds for next iteration...')
                time.sleep(self.loop_interval)
                
            except Exception as e:
                self.get_logger().error(f'Pipeline error: {e}')
                time.sleep(5)  # エラー時は5秒待機

    def start_pipeline(self):
        """パイプラインを開始"""
        if self.pipeline_running:
            self.get_logger().warning('Pipeline is already running')
            return
        
        self.pipeline_running = True
        self.pipeline_thread = threading.Thread(target=self.pipeline_loop, daemon=True)
        self.pipeline_thread.start()
        self.get_logger().info('Pipeline started')

    def stop_pipeline(self):
        """パイプラインを停止"""
        if not self.pipeline_running:
            self.get_logger().warning('Pipeline is not running')
            return
        
        self.pipeline_running = False
        if self.pipeline_thread:
            self.pipeline_thread.join(timeout=5)
        self.get_logger().info('Pipeline stopped')

    def process_command(self, command):
        """ユーザーコマンドを処理"""
        command = command.strip().lower()
        
        if command == "start":
            self.start_pipeline()
            
        elif command == "stop":
            self.stop_pipeline()
            
        elif command == "once":
            if not self.pipeline_running:
                self.run_pipeline_once()
            else:
                self.get_logger().warning('Stop continuous pipeline first before running once')
                
        elif command == "status":
            status = "RUNNING" if self.pipeline_running else "STOPPED"
            self.get_logger().info(f'Pipeline status: {status}')
            self.get_logger().info(f'VLM Mode: {self.vlm_mode}')
            self.get_logger().info(f'Action history: {len(self.action_history)}/{self.max_history} entries')
            if self.latest_image is not None:
                height, width = self.latest_image.shape[:2]
                self.get_logger().info(f'Latest image size: {width}x{height}')
            else:
                self.get_logger().info('No camera data received yet')
            
            # 行動履歴の表示
            if self.action_history:
                self.get_logger().info('Recent actions:')
                for i, action in enumerate(self.action_history[-3:], 1):
                    self.get_logger().info(f'  {i}. {action[:50]}...' if len(action) > 50 else f'  {i}. {action}')
                
        elif command == "quit":
            self.stop_pipeline()
            self.get_logger().info('Shutting down...')
            return False
            
        elif command == "help":
            self.get_logger().info('Available commands: start, stop, once, status, quit, help')
            
        else:
            self.get_logger().warning(f'Unknown command: {command}. Type "help" for available commands.')
        
        return True


def main(args=None):
    rclpy.init(args=args)
    
    # コマンドライン引数の処理
    import argparse
    parser = argparse.ArgumentParser(description="Robot Vision Pipeline (Hybrid Version)")
    parser.add_argument("--interval", "-i", type=int, default=3,
                       help="Pipeline loop interval in seconds (default: 3)")
    parser.add_argument("--model", "-m", type=str, 
                       default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="VLM model path (for Qwen mode)")
    parser.add_argument("--device", "-d", type=str, 
                       default="auto",
                       help="Device to use (auto, cpu, cuda, cuda:0, cuda:1, etc.)")
    parser.add_argument("--experiment", "-e", type=str, 
                       default=None,
                       help="Experiment ID (default: auto-generated timestamp)")
    parser.add_argument("--vlm-mode", type=str, 
                       choices=["qwen", "gemini"],
                       default="qwen",
                       help="VLM mode: qwen or gemini (default: qwen)")
    
    # sys.argvから取得（ROS2パラメータを除外）
    import sys
    filtered_args = [arg for arg in sys.argv[1:] if not arg.startswith('__')]
    parsed_args = parser.parse_args(filtered_args)
    
    pipeline = RobotVisionPipelineHybrid(
        loop_interval=parsed_args.interval,
        model_path=parsed_args.model,
        device=parsed_args.device,
        experiment_id=parsed_args.experiment,
        vlm_mode=parsed_args.vlm_mode
    )
    
    try:
        # 非ブロッキングでROS2を実行
        def spin_ros():
            rclpy.spin(pipeline)
        
        ros_thread = threading.Thread(target=spin_ros, daemon=True)
        ros_thread.start()
        
        # 初期メッセージ
        print(f"\n=== Robot Vision Pipeline (Hybrid - {parsed_args.vlm_mode.upper()}) ===")
        print(f"VLM Mode: {parsed_args.vlm_mode}")
        print("Waiting for camera data...")
        print("Type 'help' for available commands")
        
        # コマンドライン入力処理
        while True:
            try:
                command = input("\nEnter command: ")
                if not pipeline.process_command(command):
                    break
            except KeyboardInterrupt:
                break
                
    except Exception as e:
        pipeline.get_logger().error(f'Error: {e}')
    finally:
        pipeline.stop_pipeline()
        pipeline.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 