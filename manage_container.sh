#!/bin/bash

# Docker Compose管理スクリプト

case "$1" in
    "start" | "up")
        echo "ROS2コンテナを起動中..."
        docker-compose up -d
        echo "コンテナが起動しました。"
        echo "VNC: http://localhost:6080"
        echo "または vnc://localhost:15900"
        ;;
    "stop" | "down")
        echo "ROS2コンテナを停止中..."
        docker-compose down
        echo "コンテナが停止しました。"
        ;;
    "restart")
        echo "ROS2コンテナを再起動中..."
        docker-compose down
        docker-compose up -d
        echo "コンテナが再起動しました。"
        ;;
    "logs")
        echo "コンテナのログを表示中..."
        docker-compose logs -f
        ;;
    "shell" | "bash")
        echo "コンテナのシェルに接続中..."
        docker-compose exec ros2-desktop-ai-robot bash
        ;;
    "status")
        echo "コンテナの状態："
        docker-compose ps
        ;;
    "setup")
        echo "初期セットアップを実行中..."
        ./setup_volumes.sh
        ;;
    *)
        echo "使用方法: $0 {start|stop|restart|logs|shell|status|setup}"
        echo ""
        echo "  start   : コンテナを起動"
        echo "  stop    : コンテナを停止"
        echo "  restart : コンテナを再起動"
        echo "  logs    : ログを表示"
        echo "  shell   : コンテナのシェルに接続"
        echo "  status  : コンテナの状態を表示"
        echo "  setup   : 初期セットアップ（ディレクトリ作成）"
        exit 1
        ;;
esac 