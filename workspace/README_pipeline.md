# Robot Vision Pipeline

ROS2ロボットのカメラから画像を取得し、VLM（Vision Language Model）で解析して、ChatGPTでロボット制御コードを生成・実行する統合パイプラインです。

## 機能概要

このパイプラインは以下の処理を自動で実行します：

1. **カメラ画像取得**: ROS2の`/camera/image_raw/compressed`トピックから画像を取得
2. **VLM解析**: Qwen2.5-VLモデルで画像を解析し、ロボットの行動指示を生成
3. **コード生成**: ChatGPT APIでロボット制御のPythonコードを生成
4. **コード実行**: 生成されたコードを自動実行してロボットを制御

## ファイル構成

- `robot_vision_pipeline_fixed.py`: メインのパイプラインスクリプト
- `pipeline_config.py`: 設定ファイル
- `qwen_image_inference.py`: VLM推論ライブラリ
- `prompt.py`: プロンプト設定（従来版）

## 必要な環境設定

### 1. 依存パッケージのインストール

```bash
pip install opencv-python numpy rclpy transformers torch torchvision openai
```

### 2. OpenAI APIキーの設定

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. VLMモデルのダウンロード

Qwenモデルは初回実行時に自動でダウンロードされます。以下のモデルが利用可能です：

- `Qwen/Qwen2-VL-2B-Instruct` (軽量)
- `Qwen/Qwen2.5-VL-3B-Instruct` (標準)
- `Qwen/Qwen2.5-VL-7B-Instruct` (高性能)

## 使用方法

### 基本的な実行

```bash
cd workspace
python3 robot_vision_pipeline_fixed.py
```

### コマンドラインオプション

```bash
# ループ間隔を20秒に設定
python3 robot_vision_pipeline_fixed.py --interval 20

# 軽量モデルを使用
python3 robot_vision_pipeline_fixed.py --model "Qwen/Qwen2-VL-2B-Instruct"

# 両方のオプションを指定
python3 robot_vision_pipeline_fixed.py -i 15 -m "Qwen/Qwen2.5-VL-7B-Instruct"
```

### 実行時のコマンド

パイプライン実行中は以下のコマンドが利用できます：

- `start`: 継続的なパイプライン実行を開始
- `stop`: 継続的なパイプライン実行を停止
- `once`: パイプラインを1回だけ実行
- `status`: パイプラインの状態とカメラ情報を表示
- `help`: 利用可能なコマンドを表示
- `quit`: プログラムを終了

## パイプラインの動作フロー

```
1. カメラ画像取得
   ↓
2. 画像を保存 (pipeline_images/)
   ↓
3. VLMで画像解析
   ↓
4. ChatGPTでコード生成
   ↓
5. コードを保存 (generated_scripts/)
   ↓
6. コードを実行
   ↓
7. 指定間隔で次のループへ
```

## 設定のカスタマイズ

`pipeline_config.py`で以下の設定を変更できます：

- ループ間隔
- VLMモデル
- OpenAIモデル
- プロンプト内容
- タイムアウト設定
- 保存ディレクトリ

## トラブルシューティング

### よくある問題

1. **OpenAI APIキーエラー**
   ```
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **カメラトピックが見つからない**
   ```bash
   ros2 topic list | grep camera
   ```

3. **VLMモデルの読み込みエラー**
   - GPU/CPUメモリ不足の可能性
   - より軽量なモデル（2B）を試してください

4. **生成されたコードの実行エラー**
   - 30秒でタイムアウトします
   - エラーログが表示されます

### ログとデバッグ

- 画像は`pipeline_images/`に保存されます
- 生成されたスクリプトは`generated_scripts/`に保存されます
- ROS2ログでパイプラインの状態を確認できます

## 使用例

### 基本的な使用例

1. ROS2ロボットシミュレーターを起動
2. カメラノードを起動
3. パイプラインを実行：
   ```bash
   python3 robot_vision_pipeline_fixed.py
   ```
4. `start`コマンドで自動実行開始

### カスタム設定での使用例

```bash
# 5秒間隔で高性能モデルを使用
python3 robot_vision_pipeline_fixed.py -i 5 -m "Qwen/Qwen2.5-VL-7B-Instruct"
```

## 注意事項

- OpenAI APIの使用量に注意してください
- VLMモデルはGPUメモリを大量に使用します
- 生成されたコードは自動実行されるため、安全性を確認してください
- ロボットの動作範囲と周囲の安全を確保してください

## ライセンス

このプロジェクトの各コンポーネントは、それぞれのライセンスに従います。 