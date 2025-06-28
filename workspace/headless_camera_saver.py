import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import os
from datetime import datetime
import time

class HeadlessCameraSaver(Node):

    def __init__(self):
        super().__init__('headless_camera_saver')
        
        # 画像保存用ディレクトリ
        self.save_dir = "captured_images"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 圧縮画像のサブスクライバー
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.compressed_image_callback,
            10)
        
        self.get_logger().info('Headless camera saver started.')
        self.get_logger().info('Commands:')
        self.get_logger().info('- Type "save" to capture and save an image')
        self.get_logger().info('- Type "auto" to start auto-saving every 3 seconds')
        self.get_logger().info('- Type "stop" to stop auto-saving')
        self.get_logger().info('- Type "quit" to exit')
        
        # 画像カウンター
        self.image_count = 0
        self.latest_image = None
        self.auto_save = False
        self.last_auto_save = time.time()

    def compressed_image_callback(self, msg):
        try:
            # 圧縮画像データをnumpy配列に変換
            np_arr = np.frombuffer(msg.data, np.uint8)
            
            # JPEGデコード
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                self.latest_image = cv_image
                
                # 自動保存モード
                if self.auto_save and time.time() - self.last_auto_save > 3.0:
                    self.save_image(cv_image, auto=True)
                    self.last_auto_save = time.time()
            else:
                self.get_logger().error('Failed to decode compressed image')
                
        except Exception as e:
            self.get_logger().error(f'Error processing compressed image: {e}')

    def save_image(self, cv_image, auto=False):
        """画像を保存する"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "auto" if auto else "manual"
        filename = f"{self.save_dir}/robot_{prefix}_{timestamp}_{self.image_count:04d}.jpg"
        
        success = cv2.imwrite(filename, cv_image)
        if success:
            self.image_count += 1
            mode = "AUTO" if auto else "MANUAL"
            self.get_logger().info(f'[{mode}] Image saved: {filename}')
            
            # 画像情報も表示
            height, width = cv_image.shape[:2]
            self.get_logger().info(f'[{mode}] Image size: {width}x{height}')
        else:
            self.get_logger().error(f'Failed to save image: {filename}')

    def process_command(self, command):
        """ユーザーコマンドを処理"""
        command = command.strip().lower()
        
        if command == "save":
            if self.latest_image is not None:
                self.save_image(self.latest_image)
            else:
                self.get_logger().warning('No image available. Wait for camera data...')
                
        elif command == "auto":
            self.auto_save = True
            self.last_auto_save = time.time()
            self.get_logger().info('Auto-save mode ENABLED (every 3 seconds)')
            
        elif command == "stop":
            self.auto_save = False
            self.get_logger().info('Auto-save mode DISABLED')
            
        elif command == "quit":
            self.get_logger().info('Shutting down...')
            return False
            
        elif command == "help":
            self.get_logger().info('Available commands: save, auto, stop, quit, help')
            
        else:
            self.get_logger().warning(f'Unknown command: {command}. Type "help" for available commands.')
        
        return True

def main(args=None):
    rclpy.init(args=args)
    
    camera_saver = HeadlessCameraSaver()
    
    try:
        # 非ブロッキングでROS2を実行
        import threading
        import sys
        
        def spin_ros():
            rclpy.spin(camera_saver)
        
        ros_thread = threading.Thread(target=spin_ros, daemon=True)
        ros_thread.start()
        
        # コマンドライン入力処理
        while True:
            try:
                command = input("Enter command (save/auto/stop/quit/help): ")
                if not camera_saver.process_command(command):
                    break
            except KeyboardInterrupt:
                break
                
    except Exception as e:
        camera_saver.get_logger().error(f'Error: {e}')
    finally:
        camera_saver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 