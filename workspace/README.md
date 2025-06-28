# ChatGPT API を使った TurtleBot3 自然言語制御システム

ChatGPT APIを使用してGazebo上のTurtleBot3を自然言語で制御するシステムです。

## 事前準備

### 1. OpenAI APIキーの取得と設定

1. OpenAI APIキーを取得
2. 環境変数に設定：

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 2. 依存関係のインストール

```bash
# OpenAI APIクライアント
pip3 install openai==0.28.1

# ROS2 TurtleBot3 シミュレーション（Dockerコンテナ内）
sudo apt install ros-humble-turtlebot3-simulations
```

### 3. 環境変数の設定（TurtleBot3用）

```bash
export TURTLEBOT3_MODEL=waffle_pi
```

## 使用方法

### 1. TurtleBot3 Gazeboシミュレーションの起動

```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### 2. 制御スクリプトの実行

```bash
cd /path/to/workspace
python3 generate_and_run_python_script.py
```

### 3. 指示の入力

プロンプトが表示されたら、英語で指示を入力してください：

```
Enter your robot control instruction (in English for better accuracy):
> Move forward at 0.2 m/s for 5 seconds
```

## 指示の例

- `Move forward at 0.2 m/s for 5 seconds`
- `Turn left at 0.5 rad/s for 3 seconds`
- `Move backward slowly for 2 seconds`
- `Rotate clockwise for 4 seconds`
- `Stop the robot`

## トピック確認

別のターミナルで以下のコマンドを実行して、制御コマンドを確認できます：

```bash
ros2 topic echo /cmd_vel
```

## ファイル構成

- `generate_and_run_python_script.py` - メインスクリプト
- `prompt.txt` - ChatGPTへの事前プロンプト
- `generated_script.py` - 生成されたROS2制御スクリプト（実行時に作成）
- `requirements.txt` - Python依存関係
- `README.md` - このファイル

## 注意事項

- より正確な結果を得るために、指示は英語で入力することを推奨
- ChatGPTの応答によっては、生成されたコードが期待通りに動作しない場合があります
- 長時間の動作には注意してください（Ctrl+Cで停止可能）

## トラブルシューティング

### OpenAI API エラー
- APIキーが正しく設定されているか確認
- APIの利用制限に達していないか確認

### ROS2 エラー
- TurtleBot3シミュレーションが起動しているか確認
- ROS2環境が正しくセットアップされているか確認

### Docker環境での実行
このシステムはROS2がインストールされたDocker環境内で実行することを想定しています。 