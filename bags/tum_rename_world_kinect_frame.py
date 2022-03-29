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
				if m.header.frame_id == "/world":
					print('Renamed /world->' + m.child_frame_id + ' to /world->' + m.child_frame_id + '_gt')
					m.child_frame_id = m.child_frame_id + '_gt'
				newList.append(m)
			if len(newList)>0:
				msg.transforms = newList
				outbag.write(topic, msg, t)
		else:
			outbag.write(topic, msg, t)
os.remove(sys.argv[1] + '.tmp')
