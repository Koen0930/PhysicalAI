# Gemini API を使った TurtleBot3 自然言語制御システム

Gemini APIを使用してGazebo上のTurtleBot3を自然言語で制御するシステムです。

## 事前準備

### 1. Google AI Studio APIキーの取得と設定

1. [Google AI Studio](https://aistudio.google.com/app/apikey)でAPIキーを取得
2. 環境変数に設定：

```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

### 2. 依存関係のインストール

```bash
# Google Generative AI クライアント
pip3 install google-genai

# ROS2 TurtleBot3 シミュレーション（Dockerコンテナ内）
sudo apt install ros-humble-turtlebot3-simulations
```

### 3. 環境変数の設定（TurtleBot3用）

```bash
export TURTLEBOT3_MODEL=waffle
```

## 使用方法

### 1. TurtleBot3 Gazeboシミュレーションの起動

```bash
ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage4.launch.py
```

### 2. 制御スクリプトの実行

```bash
cd /path/to/workspace
python3 robot_vision_pipeline.py
```

## 注意事項

- Geminiは日本語と英語の両方に対応しているため、どちらの言語でも指示可能
- Geminiの応答によっては、生成されたコードが期待通りに動作しない場合があります
- 長時間の動作には注意してください（Ctrl+Cで停止可能）

## トラブルシューティング

### Google AI API エラー
- APIキーが正しく設定されているか確認
- APIの利用制限に達していないか確認
- Google AI Studioでプロジェクトが有効になっているか確認

### ROS2 エラー
- TurtleBot3シミュレーションが起動しているか確認
- ROS2環境が正しくセットアップされているか確認

### Docker環境での実行
このシステムはROS2がインストールされたDocker環境内で実行することを想定しています。 