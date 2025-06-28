#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2-VL Image Inference Script
画像を入力として受け取り、Qwen2-VLモデルで推論を行うスクリプト
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenImageInference:
    def __init__(self, model_path="Qwen/Qwen2-VL-2B-Instruct", device="auto", lazy_load=False):
        """
        Qwen2-VL推論クラスの初期化
        
        Args:
            model_path (str): モデルのパス
            device (str): 使用するデバイス ("auto", "cuda", "cpu", "cuda:0", "cuda:1"など)
            lazy_load (bool): Trueの場合、最初の推論時にモデルを読み込む
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.lazy_load = lazy_load
        
        # 遅延読み込みでない場合は即座にモデルを読み込み
        if not lazy_load:
            self.load_model()
    
    def load_model(self):
        """モデルとプロセッサの読み込み"""
        if self.model is not None and self.processor is not None:
            print("Model already loaded, skipping...")
            return
            
        print(f"Loading model from: {self.model_path}")
        
        # モデルの読み込み
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype="auto", 
            device_map=self.device
        )
        
        # プロセッサの読み込み
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        print("Model loaded successfully!")
    
    def infer_image(self, image_path, prompt="この画像を詳しく説明してください。", max_tokens=1024):
        """
        画像に対して推論を実行
        
        Args:
            image_path (str): 画像ファイルのパス
            prompt (str): テキストプロンプト
            max_tokens (int): 生成する最大トークン数
            
        Returns:
            str: 推論結果のテキスト
        """
        # 遅延読み込みの場合、ここでモデルを読み込み
        if self.lazy_load and (self.model is None or self.processor is None):
            self.load_model()
        
        # 画像パスを絶対パスに変換
        image_path = os.path.abspath(image_path)
        
        # 画像パスの確認
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # 画像の読み込みと確認
        try:
            image = Image.open(image_path).convert("RGB")
            print(f"画像を読み込みました: {image_path}")
        except Exception as e:
            raise ValueError(f"画像の読み込みに失敗しました: {e}")
        
        # メッセージの構築
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # 推論の準備
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # デバイスの設定
        target_device = self.device
        if self.device == "auto":
            if torch.cuda.is_available():
                target_device = "cuda"
            else:
                target_device = "cpu"
        
        # GPUが利用可能な場合はGPUに移動
        if torch.cuda.is_available() and target_device != "cpu":
            inputs = inputs.to(target_device)
        
        # 推論の実行
        print("Generating response...")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
            )
        
        # 結果のデコード
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0] if output_text else ""
    
    def infer_batch(self, image_paths, prompts=None, max_tokens=1024):
        """
        複数の画像に対してバッチ推論を実行
        
        Args:
            image_paths (list): 画像ファイルパスのリスト
            prompts (list): プロンプトのリスト（Noneの場合はデフォルトプロンプトを使用）
            max_tokens (int): 生成する最大トークン数
            
        Returns:
            list: 推論結果のテキストのリスト
        """
        if prompts is None:
            prompts = ["この画像を詳しく説明してください。"] * len(image_paths)
        
        if len(image_paths) != len(prompts):
            raise ValueError("画像数とプロンプト数が一致しません")
        
        results = []
        for i, (image_path, prompt) in enumerate(zip(image_paths, prompts)):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            try:
                result = self.infer_image(image_path, prompt, max_tokens)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append(f"Error: {e}")
        
        return results

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Qwen2-VL Image Inference")
    parser.add_argument("--image", "-i", type=str, required=True, 
                       help="画像ファイルのパス")
    parser.add_argument("--prompt", "-p", type=str, 
                       default="Describe the image in detail.",
                       help="テキストプロンプト")
    parser.add_argument("--model", "-m", type=str, 
                       default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="モデルのパス")
    parser.add_argument("--max-tokens", type=int, default=128,
                       help="生成する最大トークン数")
    parser.add_argument("--device", type=str, default="auto",
                       help="使用するデバイス (例: auto, cuda, cpu, cuda:0, cuda:1)")
    parser.add_argument("--output", "-o", type=str,
                       help="結果を保存するファイルパス")
    
    args = parser.parse_args()
    
    try:
        # 推論実行
        inference = QwenImageInference(args.model, args.device)
        result = inference.infer_image(args.image, args.prompt, args.max_tokens)
        
        # 結果の表示
        print("\n" + "="*50)
        print("推論結果:")
        print("="*50)
        print(result)
        print("="*50)
        
        # ファイルに保存（オプション）
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"Image: {args.image}\n")
                f.write(f"Prompt: {args.prompt}\n")
                f.write(f"Result:\n{result}\n")
            print(f"結果をファイルに保存しました: {args.output}")
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 