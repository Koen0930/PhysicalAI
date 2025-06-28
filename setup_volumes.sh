#!/bin/bash

# ワークスペースディレクトリを作成
echo "ワークスペースディレクトリを作成中..."

# 必要なディレクトリを作成
mkdir -p workspace

# ディレクトリの権限を設定
chmod 755 workspace

echo "以下のディレクトリが作成されました："
echo "  - workspace/     : メインの作業スペース"
echo ""
echo "Docker Composeでコンテナを起動するには以下のコマンドを実行してください："
echo "  docker-compose up -d" 