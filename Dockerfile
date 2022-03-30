FROM ros:noetic
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
 wget \
 git \
 bash-completion \
 build-essential \
 sudo \
 ros-noetic-rtabmap-ros \
 ros-noetic-rosbridge-suite \
 && rm -rf /var/lib/apt/lists/*
# Now create the same user as the host itself
ARG UID=1000
ARG GID=1000
RUN addgroup --gid ${GID} ros
RUN adduser --gecos "ROS User" --disabled-password --uid ${UID} --gid ${GID} ros
RUN usermod -a -G dialout ros
ADD docker/99_aptget /etc/sudoers.d/99_aptget
RUN chmod 0440 /etc/sudoers.d/99_aptget && chown root:root /etc/sudoers.d/99_aptget
# Choose to run as user
ENV USER ros
USER ros 
# Change HOME environment variable
ENV HOME /home/${USER} 
# workspace setup
RUN mkdir -p ${HOME}/ros_ws/src

WORKDIR ${HOME}/ros_ws/src
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash; catkin_init_workspace"
WORKDIR ${HOME}/ros_ws
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash; catkin_make"

# set up environment
COPY docker/update_bashrc.sh /sbin/update_bashrc
RUN sudo chmod +x /sbin/update_bashrc ; sudo chown ros /sbin/update_bashrc ; sync ; /bin/bash -c /sbin/update_bashrc ; sudo rm /sbin/update_bashrc
# Change entrypoint to source ~/.bashrc and start in ~
COPY docker/entrypoint.sh /ros_entrypoint.sh
RUN sudo chmod +x /ros_entrypoint.sh ; sudo chown ros /ros_entrypoint.sh ;

RUN sudo sh \
    -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" \
        > /etc/apt/sources.list.d/ros-latest.list' && \
        wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
RUN sudo apt-get update && sudo apt-get install -y python3-catkin-tools ros-noetic-tf2-sensor-msgs python3-pip ros-noetic-ros-numpy
RUN pip3 install pcl
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Clean image
RUN sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/* 
WORKDIR ${HOME}/ros_ws/


ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]