# Darknet_v4
darknet_v4 is a package that connects darknet and ros. 

# Instalation and Setup
Fist you need cv_bridge to work on python3. Follow [this tutorial](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674) and then add the path of the used workspace to your .bashrc.

# Usage


Video0 and Video1 
## Images from webcams
To run it directly on your webcams use
```bash
roslaunch yolov5 detect_from_cams.launch
```
or
```bash
cd scr/ 
python3 detect_from_cams.py
```

## Images from topics
```bash
roslaunch yolov5 detect_from_topic.launch
```
or
```bash
cd src/
python3 detect_from_topic.py 
```


Using python3 command allows you to use aditional arguments --weigths (path_to_weigts), --save-img, etc.

## Switching between cameras
The nodes can receive input from 2 diferent sources each. They always start getting images from the front source.
To switch between sources use 
```bash 
rostopic pub /change_camera std_msgs/String bottom
```
To switch back repeat changing 'bottom' to 'front'. 

The sources for detect_from_cams are:
- Video0 (front)
- Video1 (bottom) 

and for detect_from_topics:
- /camera/left/image_raw (front) 
- /camera/bottom/left/image_raw (bottom)

It always starts using the inputs from front.

# Message
The detec_from_topic node publishes at /yolov5 and /detection_image topics. 

To see the detections message run
```bash
rostopic echo /yolov5
```

The published message is an array of type Detections, containing the following parameters:

* string label ----- object number
* float64 x -------- X axys relative to box center 
* float64 y -------- Y axys 
* float64 w -------- box width
* float64 h -------- box hight

To see the image with detection boxes run
```bash
rosrun image_view image_view image:=/detection_image
```
# Dependencies
You must have the following package installed [robosub_messages](https://gitlab.com/nautilusufrj/brhue/robosub_msgs). 
