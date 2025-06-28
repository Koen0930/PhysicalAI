# ROS2 Desktop AI Robot - Docker Compose セットアップ

このディレクトリには、ROS2デスクトップAIロボット環境をDocker Composeで起動するための設定が含まれています。

## ファイル構成

- `docker-compose.yml` - Docker Composeの設定ファイル
- `setup_volumes.sh` - ボリュームマウント用ディレクトリの作成スクリプト
- `manage_container.sh` - コンテナ管理用の便利スクリプト
- `run_ai_robot_container.sh` - 元のDockerスクリプト（参考用）

## 初回セットアップ

1. ボリュームマウント用のディレクトリを作成：
```bash
cd PhysicalAI
./setup_volumes.sh
```

## 使用方法

### 基本的な操作

```bash
# コンテナを起動
./manage_container.sh start

# コンテナを停止
./manage_container.sh stop

# コンテナを再起動
./manage_container.sh restart

# コンテナの状態を確認
./manage_container.sh status

# コンテナのシェルに接続
./manage_container.sh shell

# ログを表示
./manage_container.sh logs
```

### Docker Composeを直接使用

```bash
# コンテナを起動（デタッチモード）
docker-compose up -d

# コンテナを停止
docker-compose down

# ログを確認
docker-compose logs -f

# コンテナのシェルに接続
docker-compose exec ros2-desktop-ai-robot bash
```

## ボリュームマウント

以下のディレクトリがホストとコンテナ間で共有されます：

| ホスト側 | コンテナ側 | 用途 |
|---------|-----------|------|
| `./workspace` | `/home/ubuntu/workspace` | メインの作業スペース（ROS2開発、ソースコード、ログなど全て） |

## アクセス方法

- **noVNC（Webブラウザ）**: http://localhost:6080
- **VNC**: vnc://localhost:15900
- **RDP**: localhost:13389
- **ROS Bridge**: localhost:9090

## 注意事項

- コンテナは`privileged`モードで動作します
- 共有メモリサイズは512MBに設定されています
- コンテナは自動的に再起動するように設定されています（`restart: unless-stopped`）

## トラブルシューティング

### ポートが既に使用されている場合

`docker-compose.yml`のportsセクションを編集して、別のポート番号を使用してください。

### 権限の問題

ワークスペースディレクトリの権限を確認してください：
```bash
ls -la workspace/
```

### コンテナが起動しない場合

ログを確認してください：
```bash
./manage_container.sh logs
``` 