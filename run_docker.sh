#!/bin/bash
IMAGE=ros_ws # the tag of your built image
NAME=ros_ws
CONTAINER=ros
mkdir -p src

if [ ! "$(docker volume ls -q | grep ${NAME}_src_vol)" ]; then
    echo "Volume does not exists. Creating one..."
        # create a shared volume to store the ros_ws
    docker volume create --driver local \
        --opt type="none" \
        --opt device="${PWD}/src" \
        --opt o="bind" \
        "${NAME}_src_vol"
else
    echo "Volume already exists (${NAME}_src_vol)."
fi

# xhost +

if [ ! "$(docker ps -q -f name=$CONTAINER)" ]; then
    if [ "$(docker ps -af status=exited -f name=$CONTAINER | grep -w $CONTAINER)" ]; then
        echo "Container exists but is stopped. Starting it..."
        docker start $CONTAINER
    else 
        echo "Container does not exists. Creating one..."
        docker run \
            -d \
             -it \
            --name ros \
            --volume="${NAME}_src_vol:/home/ros/ros_ws/src/:rw" \
            --volume="${PWD}/bags:/home/ros/ros_ws/bags:rw" \
            -p "11311:11311" \
            -p "9090:9090" \
            "$IMAGE"
        docker exec -it $CONTAINER /bin/bash
    fi
else 
    echo "Container is already running ($CONTAINER). Attaching..."
    docker exec -it $CONTAINER /bin/bash
fi