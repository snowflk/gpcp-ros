#!/usr/bin/env python
import ctypes
import struct
import rospy
import tf
import numpy as np
import ros_numpy
from PIL import Image
import open3d as o3d
import cv2
import pcl
import cv_bridge
from std_msgs.msg import UInt32MultiArray
from sensor_msgs.msg import CameraInfo
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
import sensor_msgs
import torch
from torchvision import transforms
from pcloud_projection.msg import FilteredPointCloud
import time
current_pc = None
current_pc_indices = None
bridge = cv_bridge.CvBridge()
publisher = None
tf_listener = None
P = None
current_ts = None
IMG_W = None
IMG_H = None


def project2d(X, colors, P,
              w=730, h=530, scale=1.,
              return_point_list=False):
    """
    Project 3D -> 2D
    :param X: List of points (N, 3)
    :param colors: (N, 3)
    :param K: Intrinsic matrix
    :param R: Rotation matrix
    :param t: Translation vector
    :param w: image width
    :param h: image height
    :param scale:
    :return:
    """
    # Extrinsic
    #Rt = np.hstack([R, t])
    # Camera matrix
    #P = K @ Rt
    # print("P", P.shape)
    # Projection
    X = X.swapaxes(0, 1)  # (N, 3) -> (3, N)
    X = np.vstack([X, np.ones((1, X.shape[1]))])
    X_raw = (P @ X)
    # Normalize by Z
    Z = X_raw[2, :]
    Xp = np.swapaxes((X_raw / Z)[:2, :], 0, 1).astype(int)
    # Make image
    w = int(w * scale)
    h = int(h * scale)
    img = Image.new(mode='RGB', size=(w, h), color='black')
    pixels = img.load()  # create the pixel map

    # allset = set(itertools.product(np.arange(0, w), np.arange(0, h)))
    # pointset = set()
    # reg = np.zeros((w, h, 3))

    for i in range(Xp.shape[0]):
        x, y = Xp[i]
        if (x >= w or w < 0) or (y >= h or y < 0):
            continue
        # pointset.add((x, y))
        # reg[x, y, :] = colors[i].astype(int)
        try:
            pixels[x, y] = tuple(colors[i].astype(int))
        except Exception as e:
            print("Error XY", x, y, "WH", w, h)
    if not return_point_list:
        return img
    return img, Xp


def convert_pc_msg_to_np(pc_msg):
    pc_msg.__class__ = sensor_msgs.msg._PointCloud2.PointCloud2
    offset_sorted = {f.offset: f for f in pc_msg.fields}
    pc_msg.fields = [f for (_, f) in sorted(offset_sorted.items())]
    # Conversion from PointCloud2 msg to np array.
    # pc_np = np.array(ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg), dtype=[('x','f4'),('y','f4'),('z', 'f4'), ('rgb', 'i4')])
    pc = ros_numpy.point_cloud2.split_rgb_field(
        ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg))
    # pull out x, y, and z values
    xyz = np.zeros(pc.shape + (3,), dtype='f4')
    rgb = np.zeros(pc.shape + (3,), dtype='u1')

    xyz[:, 0] = pc['x']
    xyz[:, 1] = pc['y']
    xyz[:, 2] = pc['z']
    rgb[:, 0] = pc['r']
    rgb[:, 1] = pc['g']
    rgb[:, 2] = pc['b']
    return xyz, rgb


def pc2_to_o3d(pc_msg):
    xyz = np.array([[0, 0, 0]])
    rgb = np.array([[0, 0, 0]])
    gen = pc2.read_points(pc_msg, skip_nans=True)
    int_data = list(gen)
    for x in int_data:
        test = x[3]
        # cast float32 to int so that bitwise operations are possible
        s = struct.pack('>f', test)
        i = struct.unpack('>l', s)[0]
        # you can get back the float value by the inverse operations
        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = (pack & 0x000000FF)
        # prints r,g,b values in the 0-255 range
        # x,y,z can be retrieved from the x[0],x[1],x[2]
        xyz = np.append(xyz, [[x[0], x[1], x[2]]], axis=0)
        rgb = np.append(rgb, [[r, g, b]], axis=0)
    return xyz, rgb


def perform_segmentation(input_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output = output.argmax(0)

    r = Image.fromarray(output.byte().cpu().numpy()).resize(input_image.size)

    return np.asarray(r)


def res_to_img(res):
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    img = Image.fromarray(res)
    img.putpalette(colors)
    return img


def process():
    global P, IMG_W, IMG_H, publisher, current_ts, current_pc
    if P is None or current_pc is None or current_ts is None:
        print("No data available. Skip")
        return
    print("Processing point cloud")
    # print(f"indices {len(current_pc_indices)}, pc {current_pc}")
    X, colors = current_pc
    img, Xp = project2d(X, colors, P=P, w=IMG_W,
                        h=IMG_H, return_point_list=True)
    cv_img = np.array(img.convert('RGB'))[:, :, ::-1]
    print("Publish image")
    publisher.publish(bridge.cv2_to_imgmsg(cv_img, encoding="bgr8"))


def on_pc_received(msg: FilteredPointCloud):
    global current_pc, current_pc_indices, current_ts
    print("Received point cloud and indices")
    start = time.time()
    #current_pc = pc2_to_o3d(msg.pc)
    xyz, rgb = convert_pc_msg_to_np(msg.pc)
    print(f"Convert time {time.time() - start}")
    current_pc = [xyz, rgb]
    current_ts = msg.pc.header.stamp
    current_pc_indices = msg.indices.data
    start = time.time()
    process()
    print(f"Processing time {time.time() - start}")


def on_camera_info_received(msg: CameraInfo):
    global P, IMG_W, IMG_H, current_ts
    current_ts = msg.header.stamp
    #print("Received camera info, getting projection matrix")
    try:
        IMG_W, IMG_H = msg.width, msg.height
        _P = np.array(msg.P).reshape(3, 4)
        """print("P", _P)
        frame_id = msg.header.frame_id
        fx, fy = _P[0,0], _P[1,1]
        cx, cy = _P[0,2], _P[1,2]
        tx, ty = _P[0,-1], _P[1, -1]
        t_fx, t_fy = coord_transform(fx, fy, frame_id)
        t_cx, t_cy = coord_transform(cx, cy, frame_id)
        t_tx, t_ty = coord_transform(tx, ty, frame_id)
        _P[0,0], _P[1,1] = t_fx, t_fy
        _P[0,2], _P[1,2] = t_cx, t_cy
        _P[0,-1], _P[1,-1] = t_tx, t_ty
        print("TP", _P)"""
        P = _P.copy()
    except Exception as e:
        print("Error", e)
        pass


def coord_transform(x, y, frame_id, target="map"):
    p = PointStamped()
    p.header.frame_id = frame_id
    p.header.stamp = rospy.Time(0)
    p.point.x = x
    p.point.y = y
    transformed_p = tf_listener.transformPoint(target, p)
    return transformed_p.point.x, transformed_p.point.y


if __name__ == "__main__":
    rospy.init_node("pcloud_segmentation_node", anonymous=True)
    rospy.Subscriber("/culling/pc_with_indices_throttle",
                     FilteredPointCloud, on_pc_received, queue_size=1)
    rospy.Subscriber("/camera/rgb/camera_info",
                     CameraInfo, on_camera_info_received)
    # rospy.Subscriber("/rtabmap/odom", Odometry, on_odom_received)
    tf_listener = tf.TransformListener()
    publisher = rospy.Publisher(
        '/projection', sensor_msgs.msg.Image, queue_size=5)
    print("Started point cloud segmenting node")
    rospy.spin()
