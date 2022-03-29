sudo apt-get install ros-noetic-rosbridge-suite
roslaunch rosbridge_server rosbridge_websocket.launch
rosparam set use_sim_time true && rosbag play --rate 1.0 --clock -l $NAME