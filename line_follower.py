#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
import socket
import struct
import io

class RedPathFollower:
    def __init__(self):
        rospy.init_node("red_path_follower")
        self.bridge = CvBridge()
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.twist = Twist()
        self.latest_frame = None  

        raspberry_pi_ip = "10.9.72.244"
        raspberry_pi_port = 8000

        # Createa socket and connect to the Raspberry Pi
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((raspberry_pi_ip, raspberry_pi_port))
        self.client_socket = self.client_socket.makefile("rb")

        self.image_timer = rospy.Timer(rospy.Duration(0.1), self.process_image)
        self.command_timer = rospy.Timer(rospy.Duration(0.1), self.publish_command)

    def process_image(self, event):
        image_len = struct.unpack("<L", self.client_socket.read(struct.calcsize("<L")))[0]

        if not image_len:
            rospy.loginfo("No image data received. Shutting down.")
            rospy.signal_shutdown("No image data received.")
            return

        image_data = io.BytesIO()
        image_data.write(self.client_socket.read(image_len))
        image_data.seek(0)

        image = np.asarray(bytearray(image_data.read()), dtype=np.uint8)

        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        self.latest_frame = frame 

    def publish_command(self, event):
        if self.latest_frame is None:
            return  

        frame = self.latest_frame  

        new_width = frame.shape[1] // 2
        new_height = frame.shape[0] // 2 
        resized_frame = cv2.resize(frame, (new_width, new_height))

        hsv_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        mask = cv2.inRange(hsv_image, lower_red, upper_red)

        adaptive_threshold = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        kernel = np.ones((5, 5), np.uint8)
        morphological_image = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(morphological_image, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            total_pixels = cv2.contourArea(contour)

            if total_pixels == 0:
                continue

            contour_mask = np.zeros_like(morphological_image)
            cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)
            red_pixels = np.sum(np.logical_and(morphological_image == 255, contour_mask == 255))
            red_percentage = red_pixels / total_pixels

            if red_percentage > 0.8:
                cv2.drawContours(resized_frame, [contour], 0, (0, 255, 0), 2)

        cv2.imshow("Red Lane Detection", resized_frame)
        cv2.waitKey(1)

        if len(contours) == 0:
            rospy.loginfo("No contour detected. Shutting down.")
            self.twist.linear.x = 0
            self.twist.angular.z = 0
            self.cmd_vel_pub.publish(self.twist)
            rospy.signal_shutdown("No contour detected.")

        moments = cv2.moments(morphological_image)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            steer = cx - resized_frame.shape[1] / 2
            self.twist.angular.z = -steer / 100

        
        self.twist.linear.x = 0.2

        
        self.cmd_vel_pub.publish(self.twist)

    def shutdown(self):
        self.client_socket.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    node = RedPathFollower()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()
