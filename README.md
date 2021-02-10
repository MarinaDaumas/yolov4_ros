# Darknet_v4
darknet_v4 is a package that connects darknet and ros. 

# Instalation and Setup
First install [OpenCV](https://www.learnopencv.com/install-opencv-3-4-4-on-ubuntu-18-04/) < 4.0 

Then go to /darknet/src directory and run
```
$ make
```
In case it doesn't work go to the Makefile and check if the options of make and architecture are the correct ones. 

# Usage


# Message
The detec_from_topic node publishes at /darknetv4 and /darknet_image topics. 

To see the detections message run
```bash
rostopic echo /darknetv4
```

The published message is an array of type Detections, containing the following parameters:

* string label ----- object number
* float64 x -------- X axys relative to box center 
* float64 y -------- Y axys 
* float64 w -------- box width
* float64 h -------- box hight

To see the image with detection boxes run
```bash
rosrun image_view image_view image:=/darknet_image
```
# Dependencies
You must have the following package installed [robosub_messages](https://gitlab.com/nautilusufrj/brhue/robosub_msgs). 
