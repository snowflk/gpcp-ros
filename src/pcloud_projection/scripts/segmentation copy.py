#!/usr/bin/env python
import ctypes
import struct
import rospy
import tf
import numpy as np
import ros_numpy
from PIL import Image as PILImage
import cv2
import pcl
import cv_bridge
from sensor_msgs.msg import CameraInfo
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PointStamped
import sensor_msgs
import torch
from torchvision import transforms
from pcloud_projection.msg import FilteredPointCloud
import time
import sys
import numpy.lib.recfunctions as rfn

class ImageBuffer:
    def __init__(self):
        self._buf = []
        self._size = 20
    
    def add(self, img: Image):
        if len(self._buf) > 0 and img.header.stamp < self._buf[-1].header.stamp:
            print("Detected jump back in time. Clear image buffer")
            self._buf.clear()

        self._buf.append(img)
        if len(self._buf) >= self._size:
            self._buf = self._buf[-self._size:]

    def lookup(self, stamp: rospy.Time, tolerance=1): 
        stamp_ns = stamp.to_nsec()
        min_diff = sys.float_info.max
        min_idx = -1
        for idx, img in enumerate(self._buf):
            diff = abs(stamp_ns - img.header.stamp.to_nsec())
            if diff < min_diff:
                min_diff = diff
                min_idx = idx
        if min_idx < 0:
            print("Something is wrong")
            return None
        if min_diff > tolerance * (10 ** 9):
            print(f"Cannot find any image for timestamp {stamp}. The nearest timestamp is {self._buf[min_idx].header.stamp}. Time difference is {min_diff}")
            return None
        print(f"Found a image with time diff = {(min_diff / (10 ** 9)):2f}s")
        return self._buf[min_idx]

class PointClassManager:
    def __init__(self):
        self._data = {}

    def add(self, point_idx, class_id):
        if point_idx in self._data:
            freq = self._data[point_idx]
            if class_id in freq:
                freq[class_id] += 1
            else:
                freq[class_id] = 1
            self._data[point_idx] = freq
        else:
            self._data[point_idx] = {class_id: 1}
    
    def get_class_id(self, point_idx):
        if point_idx in self._data:
            freq = self._data[point_idx]
            class_id = max(freq, key=freq.get)
            return class_id
        else:
            return -1

current_pc = None
current_full_pc = None
current_pc_indices = None
pc_frame_id = None
current_pc_array = None
bridge = cv_bridge.CvBridge()
publisher = None
img_publisher = None
seg_publisher = None
pc_publisher = None
tf_listener = None
P = None
current_ts = None
current_full_pc_ts = None
IMG_W = None
IMG_H = None
img_buffer = ImageBuffer()
class_manager = PointClassManager()
model = None
class_mapping = [] # {point_idx: {class_id: count}} 
VOC_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic", "light",
    "fire", "hydrant",
    "stop", "sign",
    "parking", "meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports", "ball",
    "kite",
    "baseball", "bat",
    "baseball", "glove",
    "skateboard",
    "surfboard",
    "tennis", "racket",
    "bottle",
    "wine", "glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot", "dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted", "plant",
    "bed",
    "dining", "table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell", "phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy", "bear",
    "hair", "drier",
    "toothbrush",
]

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
    img = PILImage.new(mode='RGB', size=(w, h), color='black')
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
            pixels[int(x), int(y)] = tuple(colors[i].astype(int))
        except Exception as e:
            print("Error XY", x, y, "WH", w, h, "COLOR", colors[i], colors[i].dtype, type(x))
            print("E=", e)
            raise e
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
        ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg, squeeze=True))
    # pull out x, y, and z values
    xyz = np.zeros(pc.shape + (3,), dtype='f4')
    rgb = np.zeros(pc.shape + (3,), dtype='u1')

    xyz[:, 0] = pc['x']
    xyz[:, 1] = pc['y']
    xyz[:, 2] = pc['z']
    rgb[:, 0] = pc['r']
    rgb[:, 1] = pc['g']
    rgb[:, 2] = pc['b']
    return xyz, rgb, pc

def convert_pc_msg_to_single_np(pc_msg):
    pc_msg.__class__ = sensor_msgs.msg._PointCloud2.PointCloud2
    offset_sorted = {f.offset: f for f in pc_msg.fields}
    pc_msg.fields = [f for (_, f) in sorted(offset_sorted.items())]
    # Conversion from PointCloud2 msg to np array.
    # pc_np = np.array(ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg), dtype=[('x','f4'),('y','f4'),('z', 'f4'), ('rgb', 'i4')])
    pc = ros_numpy.point_cloud2.split_rgb_field(
        ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg, squeeze=True))
    return pc

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

    r = PILImage.fromarray(output.byte().cpu().numpy()).resize(input_image.size)

    return np.asarray(r)


def res_to_img(res):
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    p_colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    p_colors = (p_colors % 255).numpy().astype("uint8")

    img = PILImage.fromarray(res)
    img.putpalette(p_colors)   
    img = img.convert('RGB') 

    return img


def process():
    global P, IMG_W, IMG_H, publisher, current_ts, current_pc, current_full_pc, current_pc_indices
    if P is None or current_pc is None or current_ts is None:
        print("No data available. Skip")
        return
    print("Processing point cloud")
    X, colors = current_pc
    img, Xp = project2d(X, colors, P=P, w=IMG_W,
                        h=IMG_H, return_point_list=True)
    cv_img = np.array(img.convert('RGB'))[:, :, ::-1]
    matched_img = img_buffer.lookup(current_ts)

    if matched_img is not None:
        matched_img_cv2 = cv2.cvtColor(bridge.imgmsg_to_cv2(matched_img, "bgr8"), cv2.COLOR_BGR2RGB)
        matched_img_pil = PILImage.fromarray(matched_img_cv2).convert("RGB")
        result = perform_segmentation(matched_img_pil)
        result_pil = res_to_img(result)
        result_msg = pil2msg(result_pil)
        print("Projection and available. Publishing them")
        img_publisher.publish(matched_img)
        seg_publisher.publish(result_msg)
        publisher.publish(bridge.cv2_to_imgmsg(cv_img, encoding="bgr8"))
        result_rgb = np.array(result_pil)
        
        # Process point cloud result
        X_dsquared = np.square(X).sum(axis=1)
        bg_color = (120, 120, 120)
        
        result_class = np.zeros(current_full_pc.shape[0], dtype=[('class', 'i1')]) 
        if "class" not in current_full_pc.dtype.names:
            current_full_pc = rfn.merge_arrays((current_full_pc, result_class), flatten=True)
        
        px_candidates = np.ones(result.shape + (2,), dtype=np.float) * (-1) # last dim: [point_idx, distance to camera]
        px_candidates[:, :, -1] = sys.float_info.max
        
        for i in range(Xp.shape[0]):
            xi, yi = Xp[i]
            if (xi >= IMG_W or xi < 0) or (yi >= IMG_H or yi < 0):
                continue
            current_candidate_idx, current_candidate_dist = px_candidates[yi, xi]
            current_dist = X_dsquared[i]
            if current_dist < current_candidate_dist or current_candidate_idx < 0:
                px_candidates[yi, xi, :] = [i, current_dist]

        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        p_colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        p_colors = (p_colors % 255).numpy().astype("uint8")
        
        print("Points", len(current_pc_array), "IDX", len(current_pc_indices))
        for i in range(Xp.shape[0]):
            xi, yi = Xp[i]
            if (xi >= IMG_W or xi < 0) or (yi >= IMG_H or yi < 0):
                ci = -1
            elif result[yi, xi] == 0:
                ci = -1
            elif px_candidates[yi, xi][0] != i:
                rgbi = bg_color
                ci = -1
            else:
                rgbi = result_rgb[int(yi), int(xi)]
                ci = result[yi, xi]
            point_idx = current_pc_indices[i]
            class_manager.add(point_idx, ci)
            #res_pc_array['r'][i] = rgbi[0]
            #res_pc_array['g'][i] = rgbi[1]
            #res_pc_array['b'][i] = rgbi[2]
            #res_pc_array['class'][i] = ci
            final_ci = class_manager.get_class_id(point_idx)
            rgbi = p_colors[final_ci] if final_ci >= 0 else bg_color
            current_full_pc['r'][point_idx] = rgbi[0]
            current_full_pc['g'][point_idx] = rgbi[1]
            current_full_pc['b'][point_idx] = rgbi[2]
            current_full_pc['class'][point_idx] = final_ci
        
        result_pc = ros_numpy.point_cloud2.array_to_pointcloud2(ros_numpy.point_cloud2.merge_rgb_fields(current_full_pc), frame_id="map", stamp=current_ts)
        pc_publisher.publish(result_pc)

def pil2msg(im):
    return bridge.cv2_to_imgmsg(np.array(im)[:, :, ::-1].copy())

def on_pc_received(msg: FilteredPointCloud):
    global current_pc, current_pc_indices, current_ts, current_pc_array, pc_frame_id
    print("Received point cloud and indices")
    start = time.time()
    xyz, rgb, current_pc_array = convert_pc_msg_to_np(msg.pc)
    # print(f"Convert time {time.time() - start}")
    current_pc = [xyz, rgb]
    pc_frame_id = msg.pc.header.frame_id
    current_ts = msg.pc.header.stamp
    current_pc_indices = msg.indices.data
    start = time.time()
    process()
    print(f"Processing time {time.time() - start}")


def on_full_pc_received(msg: PointCloud2):
    global current_full_pc, current_full_pc_ts
    if current_full_pc_ts is None:
        current_full_pc_ts = msg.header.stamp
    if msg.header.stamp < current_full_pc_ts:
        print("Detected jump back in time")
        current_full_pc = None
    current_full_pc_ts = msg.header.stamp
    if current_full_pc is None:
        current_full_pc = convert_pc_msg_to_single_np(msg)
    else:
        tmp_full_pc = convert_pc_msg_to_single_np(msg)
        print("LENGTH", len(tmp_full_pc))
        for k in ['x', 'y', 'z']:
            current_full_pc[k] = tmp_full_pc[k][:len(current_full_pc)]
        tmp_rest = tmp_full_pc[len(current_full_pc):]
        result_class = np.zeros(tmp_rest.shape[0], dtype=[('class', 'i1')]) 
        tmp_rest = rfn.merge_arrays((tmp_rest, result_class), flatten=True)
        current_full_pc = rfn.stack_arrays([current_full_pc, tmp_rest])
        print("NEW LENGTH", len(current_full_pc))


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

def on_img_received(msg: Image):
    img_buffer.add(msg)

def coord_transform(x, y, frame_id, target="map"):
    p = PointStamped()
    p.header.frame_id = frame_id
    p.header.stamp = rospy.Time(0)
    p.point.x = x
    p.point.y = y
    transformed_p = tf_listener.transformPoint(target, p)
    return transformed_p.point.x, transformed_p.point.y



if __name__ == "__main__":
    print("Loading Deep Learning model...")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()
    rospy.init_node("pcloud_segmentation_node", anonymous=True)
    rospy.Subscriber("/culling/pc_with_indices_throttle",
                     FilteredPointCloud, on_pc_received, queue_size=1)
    rospy.Subscriber("/camera/rgb/camera_info",
                     CameraInfo, on_camera_info_received)
    rospy.Subscriber("/camera/rgb/image_color", Image, on_img_received)
    rospy.Subscriber("/rtabmap/cloud_map", PointCloud2, on_full_pc_received)
    tf_listener = tf.TransformListener()
    publisher = rospy.Publisher(
        '/projection/image', sensor_msgs.msg.Image, queue_size=5)
    img_publisher = rospy.Publisher('/projection/matching_image', sensor_msgs.msg.Image, queue_size=5)
    seg_publisher = rospy.Publisher('/projection/segmentation', sensor_msgs.msg.Image, queue_size=5)
    pc_publisher = rospy.Publisher('/projection/segmented_cloud', PointCloud2, queue_size=1)
    print("Started point cloud segmenting node")
    rospy.spin()
