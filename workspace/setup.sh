#!/bin/bash

echo "Setting up ChatGPT-ROS2 Control System..."

# Install OpenAI Python package
echo "Installing OpenAI Python package..."
pip3 install openai==0.28.1

# Install TurtleBot3 simulations (if not already installed)
echo "Installing TurtleBot3 simulations..."
sudo apt update
sudo apt install -y ros-humble-turtlebot3-simulations

# Set TurtleBot3 model environment variable
echo "Setting TurtleBot3 model environment variable..."
echo "export TURTLEBOT3_MODEL=waffle_pi" >> ~/.bashrc

# Source ROS2 setup
echo "Sourcing ROS2 setup..."
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Make scripts executable
chmod +x generate_and_run_python_script.py
chmod +x example_generated_script.py

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'"
echo "2. Source your .bashrc: source ~/.bashrc"
echo "3. Launch TurtleBot3 simulation: ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py"
echo "4. Run the control system: python3 generate_and_run_python_script.py" 