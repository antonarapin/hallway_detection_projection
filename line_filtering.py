###############################
#
#	line_filtering.py
#	
#	contains the essential line filtering and point extraction functions
#
###############################




# importing all of the required packages
import cv2
import numpy as np
from math import *
from houghTransform import *
from homography_projection import *
from line_filtering_extra_utilities import *
from sklearn.cluster import KMeans


# setting up some essential global variables for easy access
process_floor = True	# floor is chosen for projection if true, otherwise projected plane is ceiling
generate_visualization = True	# an image with visualization of the processing and filtering features will be generated if true
canny_thresh1 = 50	# first Canny threshold
canny_thresh2 = 200	# second Canny threshold
num_clusters = 20	# number of clusters to fit using kmeans
tilt_thresh = 5		# tilt threshold for tilted line filtering

def get_intersections(lines):

	""" Function that records intersection of every line with every other line and 
	returns the list of intersections and corresponding line indexes in the line container
	Input: list of lists of tuples; each entry is a line, described by two endpoints, each is an (x,y) tuple
	Output: two lists; first one is the points of intersection; second one is pairs of indexes of the lines in the input list that correspond to the appropriate intersection in the first list
	"""

	# these are the lists that will be output by the function
	pts = []
	idxs = []
	# we iterate through all lines by index
	for l1 in range(len(lines)):
		line1 = lines[l1]
		
		# eac line can be described in cartesian coordinates as y = kx + b
		# calculating k for the initial line
		# this condition prevents the devision by zero
		if (line1[1][0]-line1[0][0])==0:
			k1 = (line1[1][1]-line1[0][1])/0.0001
		else:
			k1 = (line1[1][1]-line1[0][1])/(line1[1][0]-line1[0][0])

		# calculating b for the initial line
		b1 = line1[1][1] - k1*line1[1][0]

		# iterate through the rest of the lines to find the intersections with every other line
		for l2 in range(len(lines)):
			if l2!=l1:
				line2 = lines[l2]
				# calculating k for the second line
				# condition to prevent division by zero error
				if (line2[1][0]-line2[0][0])==0:
					k2 = (line2[1][1]-line2[0][1])/0.0001
				else:
					k2 = (line2[1][1]-line2[0][1])/(line2[1][0]-line2[0][0])
				
				# calculating b for the othe rline
				b2 = line2[1][1] - k2*line2[1][0]
				pt = [-1,-1]
				
				# we check the slopes, and if they are eaqual, we got the parallel lines
				if (k1-k2)!=0:
					# x coordinate of the intersection of the lines
					pt[0] = (b2-b1)/(k1-k2)
				else:
					# lines are parallel, so intersect at infinity
					pt[0] = 10000000
				# calculate the y coordinate of the intersection
				pt[1] = k1*pt[0]+b1
				# add the intersection and both lines' indexes to the lists
				pts.append(pt)
				idxs.append([l1,l2])

	# return the resulting containers
	return pts, idxs

def filter_clusters(img, inter_pts, l_idxs):
	""" Function that filters out the clusters of the points of intersection that are within the image's bounds
	Input: image, list of points of intersection, list of the indexes that correspond to intersecting lines 
	Output: the points of intersection that are within the bounds, corresponding line indexes, kmeans object
	"""


	# we acquire the image's dimensions and initialize the containers for filteres intersections
	height, width = img.shape[:2]
        filtered_pts = []
        filtered_idxs = []

	# looping through points of intersection
        for pt in inter_pts:
                if pt[1] > 0 and pt[1] < height:
                        if pt[0] > 0 and pt[0] < width:
				# point is within the image's bounds, so append the corresponding line indexes and point to the output lists
                                filtered_pts.append(pt)
                                filtered_idxs.append(l_idxs[inter_pts.index(pt)])
				# draw a small circle at the intersection's location if visualization feature is on
				if generate_visualization:
                                	cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0,250,0),1)



	# formatting the array of points into the numpy way and generating a kmeans object provided by scikit-lear library
        np_pts = np.array(filtered_pts)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(np_pts)
	# if the visualization is on, draw the cluster centers on the image
	if generate_visualization:
        	for kctr in kmeans.cluster_centers_:
                	cv2.circle(img, (int(kctr[0]), int(kctr[1])), 5, (0,0,250),4)
	
	# return the appropriate objects
	return filtered_pts, filtered_idxs, kmeans


def get_mid_cluster_ctr(img, kmeans):
	"""Function that finds the cluster that is closest to the middle of the image
	Input: image, kmeans object
	Output: location of the middle cluster
	"""

	# we determine the midpoint of the image
	height, width = img.shape[:2]
        mid_pt = [width/2,height/2]
	# variables to find the closest cluster
        mid_clust_ctr = None
        min_dist = 100000000
        # iterate through cluster centers to find the one in the middle
        for ctr in kmeans.cluster_centers_:
		# calculating the distance from the cluster center to the center of the image
                dist = math.sqrt((ctr[1]-mid_pt[1])**2 + (ctr[0]-mid_pt[0])**2)
		# updating the variables
                if dist<min_dist:
                        min_dist = dist
                        mid_clust_ctr = ctr
	# paint the central cluster in a different color if the visualization is on
	if generate_visualization:
        	cv2.circle(img, (int(mid_clust_ctr[0]), int(mid_clust_ctr[1])), 5, (200,0,0),4)

	# return middle cluster location
	return mid_clust_ctr	



def filter_tilted_lines(img,lines,mid_clust_lines):
	""" Function that filters out the tilted lines referenced by the indexes in the input list
	Input: image, list of all lines, list of line indexes that have intersections in the central cluster
	Output: list of lines with tilt larger than a given threshold
	"""
	
	# setting up the containers and acquiring the image dimensions 
	tilted_lines = []
        #tilts = []
        height, width = img.shape[:2]
        
	# iterate through the pairs of intersecting lines
        for lidx in mid_clust_lines:
                line1 = lines[lidx[0]]
                line2 = lines[lidx[1]]
                
		# set up the variables that will be set based on lines' inclination
                line1_g = True
                line2_g = True
                tilt1 = 0
                tilt2 = 0


		# prevent division be zero
                if (line1[1][0] - line1[0][0])==0:
			# the line is not tilted, we do not need it
                        tilt1=0
                        line1_g = False
                else:
			# the line is tilted, calculate the absolute value of the angle of the tilt, we do not care about the direction
                        t = (height-line1[1][1])-(height-line1[0][1])
                        b = line1[1][0]-line1[0][0]
                        tilt1 = abs(math.atan2(t,b))
		# prevent division by zero
                if (line2[1][0] - line2[0][0])==0:
			# line is not tilted, we do not need it
                        tilt2=0
                        line2_g = False
                else:
			# the line is tilted, calculate the absolute value of the angle of the tilt, we do not care about the direction
                        t = (height-line2[1][1])-(height-line2[0][1])
                        b = line2[1][0]-line2[0][0]
                        tilt2 = abs(math.atan2(t,b))

		# convert radians to degrees, taking the absolute value
		tilt1 = abs(math.degrees(tilt1))
                tilt2 = abs(math.degrees(tilt2))
		# check the quadrant for both lines
		# we care only about the angle from vertical of from horizontal
                if tilt1 > 90 and tilt1 < 180:
                        tilt1 = 180 - tilt1
                elif tilt1 > 180 and tilt1 < 270:
                        tilt1 = tilt1 - 180
                elif tilt1 > 360:
                        tilt1 = tilt1 - 270

                if tilt2 > 90 and tilt2 < 180:
                        tilt2 = 180 - tilt2
                elif tilt2 > 180 and tilt2 < 270:
                        tilt2 = tilt2 - 180
                elif tilt2 > 270:
                        tilt2 = tilt2 - 270
                
                # compare each line's tilt based on the threshold 
                if tilt1<tilt_thresh or tilt1>(90-tilt_thresh):
			# mark the line as not tilted
                        line1_g = False
			# if line is too close to the horizontal or vertical axis and visualization is on, color it 
			if generate_visualization:
                        	cv2.line(img,(int(line1[0][0]),int(line1[0][1])),(int(line1[1][0]),int(line1[1][1])),(200,60,200),2)
                if tilt2<tilt_thresh or tilt2>(90-tilt_thresh):
			# mark the line as not tilted
                        line2_g = False
			# if line is too close to the horizontal or vertical axis and visualization is on, color it
			if generate_visualization:
                        	cv2.line(img,(int(line2[0][0]),int(line2[0][1])),(int(line2[1][0]),int(line2[1][1])),(200,60,200),2)
		
		# the lines which remain marked as tilted are pushed to the corresponding containers
                if line1_g:
                        tilted_lines.append(line1)
                        #tilts.append(tilt1)
                if line2_g:
                        tilted_lines.append(line2)
                        #tilts.append(tilt2)

	# return the list of tilted lines
	return tilted_lines



def sort_dimensions(lines,img,ctr):
	""" Function that sorts the tilted lines into the approximate planes: right wall, left wall, ceiling and floor
	Input: list of tilted lines, image and the location of the cluster to which lines converge
	Output:	four lists of lines, ech belonging to one of the four planes, and four lists of teo indexes, containing the indexes of the lines in the corresponding planes that are closest to the edges of thir plane
	"""
	
	# setting up the lists that we will fill with lines	
	l_w = []
	r_w = []
	c = []
	f = []
	# setting up the boundaries of the lines that are closest to the edges of their planes based on the angle of inclination
	# also setting up index containers for the two lines on the two edges
	f_ends=[45,135]
	f_e_idxs=[None,None]
	c_ends=[-135,-45]
	c_e_idxs=[None,None]
	r_ends=[135,-135]
	r_e_idxs = [None,None]
	l_ends=[-45,45]
	l_e_idxs=[None,None]
	
	# acquiring the image dimensions and center coordinates
	ctry = ctr[1]
	ctrx = ctr[0]
	height, width = img.shape[:2]

	# iterating through the lines t osort each line in some container
	for line in lines:
		# we determine the approximate location fo the line segment's center based on the endpoints
		lx = (line[1][0]+line[0][0])/2
		ly = ((height-line[1][1])+(height-line[0][1]))/2
		# calculating the angle of inclination of each line
		t = (height-line[1][1])-(height-line[0][1])
                b = line[1][0]-line[0][0]
                angle = math.degrees(math.atan2(t,b))
		# translating the negative angles into positive
		if angle<0:
			angle = 360 - angle
		# filtering out the lines based on tilt and location relative to the center into one of the four planes:
		# floor, ceiling, left wall and right wall
		if (angle>45 and angle<135) and (ly<ctry):
			f.append(line)
			# check if the line is closest to one of the edges of the plane based on what we have seen so far
			if angle>f_ends[0]:
				f_ends[0] = angle
				f_e_idxs[0]= len(f)-1
			if angle<f_ends[1]:
				f_ends[1] = angle
				f_e_idxs[1] = len(f)-1
		elif (angle>45 and angle<135) and (ly>ctry):
                        c.append(line)
			# check if the line is closest to one of the edges of the plane based on what we have seen so far
                        if angle>c_ends[0]:
                                c_ends[0] = angle	
				c_e_idxs[0] = len(c)-1
                        if angle<c_ends[1]:
                                c_ends[1] = angle
				c_e_idxs[1] = len(c)-1
		elif (angle<45 and angle>-45) and (lx<ctrx):
                        l_w.append(line)
			# check if the line is closest to one of the edges of the plane based on what we have seen so far
                        if angle>l_ends[0]:
                                l_ends[0] = angle
				l_e_idxs[0] = len(l_w)-1
                        if angle<l_ends[1]:
                                l_ends[1] = angle
				l_e_idxs[1] = len(l_w)-1

		elif (angle>-45 or angle<45) and (lx>ctrx):
			r_w.append(line)
			# check if the line is closest to one of the edges of the plane based on what we have seen so far
			if angle>r_ends[0]:
				r_ends[0] = angle
				r_e_idxs[0] = len(r_w)-1
			if angle<r_ends[1]:
				r_ends[1] = angle
				r_e_idxs[1] = len(r_w)-1

	# return the lists of lies for each plane, as well as the indexes of the lines that are closest to the edge of their plane
	return l_w,l_e_idxs,r_w,r_e_idxs,c,c_e_idxs,f,f_e_idxs





def get_pts(l1,l2,min_y,max_y,img):
	""" Function that finds the points on two lines that have the minimum and maximum y coordinates 
	Input: two lines, described by the pairs of endpoints, minimum and maximum y coordinates that we are looking for
	Output: the list of points on the extensions of the lines, in circular order
	"""

	# each line has an equation y = kx + b
	# here we calculate the k and b elements for both lines
	k1 = float(l1[0][1] - l1[1][1])/float(l1[0][0] - l1[1][0])
	k2 = float(l2[0][1] - l2[1][1])/float(l2[0][0] - l2[1][0])
	b1 = float(l1[1][1] - float(k1*l1[1][0]))
	b2 = float(l2[1][1] - float(k2*l2[1][0]))
	# having the formula, we simply determine the location of the x coordinates on the extensions of the lines 
	# that correspond to minimum and maximum y values
	x_min_1 = int((min_y-b1)/k1)
	x_min_2 = int((min_y-b2)/k2)
	x_max_1 = int((max_y-b1)/k1)
	x_max_2 = int((max_y-b2)/k2)
	# we return the list of points ordered in a circular manner
	return [[x_min_1,min_y],[x_min_2,min_y],[x_max_2,max_y],[x_max_1,max_y]]	




def get_perspective_points(img,lines):
	""" Function that determines the points in the perspective image onto which the projection is to be applied
	Input: edge image and the list of line segments
	Output: a list of four points for projection on the image determined by the line-based filtering 
	"""

	#lines =  probabilistic_hough_line(img, 30, 100, 4)
	
	# get the points of intersection and the corresponding line indexes from he set of lines
	inter_pts, l_idxs = get_intersections(lines)
	# process the edge image to make it three-channel, enabling the drawing 
	img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

	# get the lines that intersect within image bounds, points of intersection, and kmeans object
	filtered_pts, filtered_idxs, kmeans = filter_clusters(img,inter_pts,l_idxs)

	# get the location of the central cluster based on the image and calculated kmeans
	mid_clust_ctr = get_mid_cluster_ctr(img, kmeans)

	# container to store lines that intersect in the central cluster
	mid_clust_lines = []
	# filter out the pairs of lines that have intersections in the central cluster
	for idx in range(len(filtered_pts)):
		pt = filtered_pts[idx]
		# we can get the predicted cluster location using the kmeans object
		# predict cluster will be the actual cluster, since it was calculated based on the lines that contain the current line
		i = kmeans.predict([pt])	
		clust = kmeans.cluster_centers_[i][0]
		# check if current line's cluster is the central cluster and add the line to the container if it is so
		if clust[0]==mid_clust_ctr[0] and clust[1]==mid_clust_ctr[1]:
			mid_clust_lines.append(filtered_idxs[idx])

	# if the visualization is on, draw the lines that belong to the central cluster 
	if generate_visualization:
		for lidx in mid_clust_lines:
			cv2.line(img,(int(lines[lidx[0]][0][0]),int(lines[lidx[0]][0][1])),(int(lines[lidx[0]][1][0]),int(lines[lidx[0]][1][1])),(0,0,250),1)
			cv2.line(img,(int(lines[lidx[1]][0][0]),int(lines[lidx[1]][0][1])),(int(lines[lidx[1]][1][0]),int(lines[lidx[1]][1][1])),(0,0,250),1)

	# get the lines that belong to the middle cluster and that have their angle of inclination within the threshold
	tilted_lines = filter_tilted_lines(img,lines,mid_clust_lines)
	
	# graph the tilted lines if visualization is on
	if generate_visualization:	
		for line in tilted_lines:
			cv2.line(img,(int(line[0][0]),int(line[0][1])),(int(line[1][0]),int(line[1][1])),(200,180,0),2)
	
	# get the lines that belong to the different planes(floor, ceiling, left wall, right wall) and the edge lines' indexes for each plane
	left_w, lwidxs, right_w, rwidxs, ceiling, cidxs, floor, fidxs = sort_dimensions(tilted_lines,img,mid_clust_ctr)
	
	# if visualization is on, draw the lines in different color for each plane
	if generate_visualization:
		for line in left_w:
                	cv2.line(img,(int(line[0][0]),int(line[0][1])),(int(line[1][0]),int(line[1][1])),(250,0,0),2)
		for line in right_w:
                	cv2.line(img,(int(line[0][0]),int(line[0][1])),(int(line[1][0]),int(line[1][1])),(0,0,250),2)
		for line in ceiling:
                	cv2.line(img,(int(line[0][0]),int(line[0][1])),(int(line[1][0]),int(line[1][1])),(250,0,250),2)
		for line in floor:
                	cv2.line(img,(int(line[0][0]),int(line[0][1])),(int(line[1][0]),int(line[1][1])),(0,250,0),2)


	# if floor is to be processed
	if process_floor:
		#check if there are more than two lines in the floor plane
		if fidxs[0]!=None and fidxs[1]!=None:
			# set the variables and draw the lines if visualization is on
			l1 = floor[fidxs[0]]
			l2 = floor[fidxs[1]]
			if generate_visualization:	
				cv2.line(img,(int(l1[0][0]),int(l1[0][1])),(int(l1[1][0]),int(l1[1][1])),(250,250,250),2)	
				cv2.line(img,(int(l2[0][0]),int(l2[0][1])),(int(l2[1][0]),int(l2[1][1])),(250,250,250),2)
		else:
			# if there are not enough lines in the floor plane, print out the message and exit 
			print("Due to the lack of line features in the floor area, the projection is not possible.")
			return 1
	else:
		#check if there are more than two lines in the ceiling plane
		if cidxs[0]!=None and cidxs[1]!=None:
			# set the variables and draw the lines if visualization is on
                        l1 = ceiling[cidxs[0]]
                        l2 = ceiling[cidxs[1]]
			if generate_vsualization:
                        	cv2.line(img,(int(l1[0][0]),int(l1[0][1])),(int(l1[1][0]),int(l1[1][1])),(250,250,250),2)
                        	cv2.line(img,(int(l2[0][0]),int(l2[0][1])),(int(l2[1][0]),int(l2[1][1])),(250,250,250),2)
                else:
			# if there are not enough lines in the ceiling plane, print out the message and exit 
                        print("Due to the lack of line features in the ceiling area, the projection is not possible.")
                        return 1
	
	# based on all of the lines in the plane, find the minimum and maximum y coordinates	
	max_y = 0
	min_y = 1000000
	for line in floor:
		# extracting and checking y coordinates
		ys = [line[0][1],line[1][1]]
		for y in ys:
			if y > max_y:
				max_y = y
			if y < min_y:
				min_y = y

	# given the minimum and maximum y coordinates, get the locations on the two edge lines in the plane
	# to get the four points required for our projection
	base_pts = get_pts(l1,l2,min_y,max_y,img)
	
	# save the processed visualizations if needed
	if generate_visualization==True:
		cv2.imwrite("process_visualization.jpg",img)	
	
	# return the set of points
	return base_pts

