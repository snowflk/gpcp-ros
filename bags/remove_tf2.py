import rosbag
import sys
import os 
from tf.msg import tfMessage

if len(sys.argv) < 1:
    print('Usage example: tum_rename_world_kinect_frame.py rgbd_dataset_freiburg3_long_office_household.bag')
    exit

os.rename(sys.argv[1], sys.argv[1] + '.tmp')
with rosbag.Bag(sys.argv[1], 'w') as outbag:
    for topic, msg, t in rosbag.Bag(sys.argv[1] + '.tmp').read_messages():
        if topic == "/tf" and msg.transforms:
            newList = [];
            for m in msg.transforms:
                if m.header.frame_id.startswith('/'):
                    print(f"Rename {m.header.frame_id} -> {m.header.frame_id[1:]}")
                    m.header.frame_id = m.header.frame_id[1:]
                if m.child_frame_id.startswith('/'):
                    m.child_frame_id = m.child_frame_id[1:]
                newList.append(m)
            if len(newList)>0:
                msg.transforms = newList
                outbag.write(topic, msg, t)
        else:
            outbag.write(topic, msg, t)
os.remove(sys.argv[1] + '.tmp')
