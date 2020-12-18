# Darknet_v4
darknet_v4 is a package that connects darknet and ros. 

# Instalation and Setup
Fist you need cv_bridge to work on python3. Follow [this tutorial](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674) and then add the path of the used workspace to your .bashrc.

# Usage


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
