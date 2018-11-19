import math
import cv2
import numpy as np
from line_filtering import *
import sys

def main():

	#take two argument from user
	base_image_file_name = str(sys.argv[1])
	projection_image_file_name = str(sys.argv[2])

	#read the base image and resize it if too big
	img = cv2.imread(base_image_file_name)#hall4.jpg
	img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
	base_img = img
	projected_img = cv2.imread(projection_image_file_name)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#get the edge extracted image
	img = cv2.Canny(img,canny_thresh1,canny_thresh2)

	#get the detected line using gradient based Progressive probabilistic Hough transform
	lines =  probabilistic_hough_line(img, 30, 100, 4)

	#set the base points that is going to be projected
	base_pts = get_perspective_points(img,lines)

	#check if the base points is valid
	if base_pts == 1:
			raise ValueError("A problem with the image file occured, quitting...Please try again with a different image")

	height = projected_img.shape[0]
	width = projected_img.shape[1]

	proj_pts = [[1,1],[width-1,1],[width-1,height-1],[1,height-1]]
	project_image(base_img,projected_img,base_pts,proj_pts)

	cv2.imwrite("projected_output.jpg",base_img)
	print("The image is saved as \"projected_output.jpg\" ")

main()
