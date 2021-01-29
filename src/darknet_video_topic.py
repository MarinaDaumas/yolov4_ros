from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
import rospy
from threading import Thread, enumerate
from queue import Queue

import rospkg

from cv_bridge import CvBridge, CvBridgeError
from robosub_msgs.msg import Detection, DetectionArray
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rospy.numpy_msg import numpy_msg

publish_img = True

width = None
height = None
network = None
class_names = None
class_colors = None
thresh = None


camera_selection = "front"
bridge = None

detections_pub = None
img_pub = None

frame_queue = None
darknet_image_queue = None
detections_queue = None
fps_queue = None

ext_output = None
dont_show = None

def load_args():
    '''
    Initializes the darknet.
    '''
    # use ros param
    global thresh

    rospack = rospkg.RosPack()
    path_to_darknet = rospack.get_path('darknet_v4')
    

    configPath = path_to_darknet + "/src/cfg/yolov4-tiny.cfg"
    weightPath = path_to_darknet + "/src/weights/yolov4-tiny.weights"
    dataFile = path_to_darknet + "/src/cfg/coco.data"

    thresh = .25 #remove detections with confidence below this value
    
    return configPath, weightPath, dataFile


def check_arguments_errors(configPath, weightPath, dataFile):
    
    if not os.path.exists(configPath):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(configPath))))
    if not os.path.exists(weightPath):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(weightPath))))
    if not os.path.exists(dataFile):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(dataFile))))


def load_network(configPath, weightPath, dataFile):
    global width, height, network, class_names, class_colors

    network, class_names, class_colors = darknet.load_network(
            configPath,
            dataFile,
            weightPath,
            batch_size=1
        )

    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    
    return darknet_image


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def video_capture(frame, frame_queue, darknet_image_queue):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    frame_queue.put(frame_resized)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
    darknet_image_queue.put(darknet_image)



def inference(darknet_image_queue, detections_queue, fps_queue):
    while True:
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
        darknet.print_detections(detections, ext_output)
        darknet.free_image(darknet_image)

        detections_pub.publish(detections)

        print(detections)


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    #video = set_saved_video(cap, args.out_filename, (width, height))
    print('pq n')
    while True:
        frame_resized = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        if frame_resized is not None:
            image = darknet.draw_boxes(detections, frame_resized, class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #if args.out_filename is not None:
            #    video.write(image)
            if not dont_show:
                cv2.imshow('Inference', image)
            if cv2.waitKey(fps) == 27:
                break
        image_pub.publish(bridge.cv2_to_imgmsg(image, "bgr8"))
        print('foi')
        cv2.imshow(image)

    if detections != [] and save_imgs:
        save_image(image)

    video.release()
    cv2.destroyAllWindows()


def selection_callback(data):
    '''
    Sets camera selection to front when data='front' and bottom when data='bottom'.
    '''
    global camera_selection
    camera_selection = data.data
    

def front_callback(data):
    '''
    Calls run_darknet using images from front camera.
    '''
    if camera_selection == "front":
        run_darknet(data)
    else:
        pass


def bottom_callback(data):
    '''
    Calls run_darknet using images from bottom camera.
    '''
    if camera_selection == "bottom":
        run_darknet(data)
    else:
        pass

def run_darknet(data):
    '''
    Runs all darknet functions
    '''

    prev_time = time.time()

    frame = bridge.imgmsg_to_cv2(data)
    
    video_capture(frame, frame_queue, darknet_image_queue)
    inference(darknet_image_queue, detections_queue, fps_queue)
    drawing(frame_queue, detections_queue, fps_queue)

    print(1/(time.time()-prev_time))


def YOLO(): 
    '''
    Main YOLO loop 
    '''
    global darknet_image, bridge, detections_pub, image_pub, frame_queue, darknet_image_queue, detections_queue, fps_queue
    
    configPath, weightPath, dataFile = load_args()
    check_arguments_errors(configPath, weightPath, dataFile)
    darknet_image = load_network(configPath, weightPath, dataFile) 

    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    

    detections_pub = rospy.Publisher('darknetv4', DetectionArray, queue_size=10)
    image_pub = rospy.Publisher("darknet_image", Image, queue_size=10)

    rospy.init_node('Darknetv4Node', anonymous=True)

    rospy.Subscriber('/change_camera', String, selection_callback)

    #rospy.Subscriber('/camera/left/image_raw', Image, front_callback)
    rospy.Subscriber('/usb_cam/image_raw', Image, front_callback)
    rospy.Subscriber('/camera/bottom/left/image_raw', Image, bottom_callback)
    
    bridge = CvBridge()

    print("Starting the YOLO loop...")

    rospy.spin()

if __name__ == '__main__':
  
    YOLO()


   
    

    
    
    
  
   