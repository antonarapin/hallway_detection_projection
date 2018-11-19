import math
import numpy as np

def get_homography(p1,p2):
	""" Function that computes homography given two sets of points
	Input: ptwo sets of corresponding points, each set should contain four two-deminsional points
	Output: a 3x3 numpy array, which is the homography that maps points in p2 to points in p1
	"""

        m = []
	# we need to construct a matrix that has the values in terms of passed in points such that
	# when we multiply this matrix and the homography in a single 9-entry-column form, we get a column of zeros
        for i in range(0, len(p1)):
                x1, y1 = float(p1[i][0]), float(p1[i][1])
                x2, y2 = float(p2[i][0]), float(p2[i][1])
                m.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
                m.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
	# formatting the acquired matrix as a numpy array
        m = np.asarray(m)
	# using the numpy.linalg library to perform SVD decomposition
        u, s, v = np.linalg.svd(m)
	# we noramlize the last column of the V matrix by the smallest value, that happens to be on the bottom,
	# reshape it into 3x3 matrix and return
        return (v[-1,:]/v[-1,-1]).reshape(3, 3)


def project_image(base_img,projected_img,base_pts,proj_pts):
        """ Function that draws the projected transformation of one image onto the other
	Iput: image on which to project, image which to project, four points in the former image and four corresponding points in the latter image
	Output: the projected_image is draw on top of the base image according to the appropriate transformation specified by the points
	"""
	# first we get the homography based on the points
	h = get_homography(proj_pts,base_pts)
	# retrieve the dimensions of the images
        nrows, ncols, nchnls = projected_img.shape
        bnrows, bncols, bnchnls = base_img.shape
	# transform every point in the image that is to be projected
        for i in range(nrows):
                for j in range(ncols):
			# create a vector for the point and apply homography by multiplying 
                        pt = [j,i,1]
                        r = h.dot(pt)
			# the last entry should be 1, thus we normalize the x and y entries based on the last entry
                        r[0] = r[0]/r[2]
                        r[1] = r[1]/r[2]
                        r[2] = r[2]/r[2]
			# check whether the projected coordinate is within the base image's borders
			# and copy the pixels from the image that is to be projected onto the base image
                        if (r[1]>0 and r[1]<bnrows) and (r[0]>0 and r[0]<bncols):
                                base_img[int(r[1]),int(r[0])] = projected_img[j,i]
