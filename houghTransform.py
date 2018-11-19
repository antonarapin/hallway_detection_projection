#import necessary libraries
import numpy as np
import imageio
import matplotlib.pyplot as plot
import math
import random
import cv2


def prob_getLine(img, threshold, line_length, line_gap, width, height, theta):

    """A helper function that returns the lines using probabilistic hough transform
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold : int
        Threshold
    line_length : int
        Minimum accepted length of detected lines.
        Increase the parameter to extract longer lines.
    line_gap : int
        Maximum gap between pixels to still form a line.
        Increase the parameter to merge broken lines more aggresively.
    theta : 1D ndarray
        Angles at which to compute the transform, in radians.
    width: int
        The image's width
    height: int
        The image's height

    Returns
    -------
    lines : list
      List of lines identified, lines in format ((x0, y0), (x1, y0)),
      indicating line start and end.
    """



    # maximum line number to prevent infinite loop
    lines_max = 2 ** 15
    lines = []

    # calculate the image diagonal
    imgDiagnal = 2 * np.ceil((np.sqrt(img.shape[0] * img.shape[0] +img.shape[1] * img.shape[1])))
    accum = np.zeros((int(imgDiagnal), int(theta.shape[0])))
    offset = imgDiagnal / 2
    nthetas = theta.shape[0]
    # compute the bins and allocate the accumulator array
    mask = np.zeros((height, width))
    line_end = np.zeros((2, 2))

    # compute sine and cosine of angles
    cosinTheta = np.cos(theta)
    sinTheta = np.sin(theta)

    # find the nonzero indexes
    yXis, xXis = np.nonzero(img)
    points = list(zip(xXis, yXis))
    # mask all non-zero indexes
    mask[yXis, xXis] = 1
    shift = 16

    while 1:

        # check if the image is empty, quit if no remaining points
        count = len(points)
        if count == 0:
            break

        # select a random non-zero point
        index = random.randint(0,count) % count
        x = points[index][0]
        y = points[index][1]

        # remove the pixel from the image
        del points[index]

        # if previously eliminated, skip
        if not mask[y, x]:
            continue

        #set some constant for the ease of later use
        value = 0
        max_value = threshold - 1
        max_theta = -1

        # apply hough transform on point
        for j in range(nthetas):
            accum_idx = int(round((cosinTheta[j] * x + sinTheta[j] * y)) + offset)
            accum[accum_idx, j] += 1
            value = accum[accum_idx, j]
            if value > max_value:
                max_value = value
                max_theta = j

        #check if the highest value change for this pixel has detected line or not
        if max_value < threshold:
            continue    #if less than the threshold, than skip this point

        # from the random point walk in opposite directions and find the longest line segment continuous
        a = -sinTheta[max_theta]
        b = cosinTheta[max_theta]
        x0 = x
        y0 = y

        # calculate gradient of walks using fixed point math
        xflag = np.fabs(a) > np.fabs(b)
        if xflag:
            if a > 0:
                dx0 = 1
            else:
                dx0 = -1
            dy0 = round(b * (1 << shift) /  np.fabs(a))
            y0 = (y0 << shift) + (1 << (shift - 1))
        else:
            if b > 0:
                dy0 = 1
            else:
                dy0 = -1
            dx0 = round(a * (1 << shift) /  np.fabs(b))
            x0 = (x0 << shift) + (1 << (shift - 1))

        # find the line segment not exceeding the acceptable line gap
        for k in range(2):
            gap = 0
            px = x0
            py = y0
            dx = dx0
            dy = dy0
            if k > 0:
                dx = -dx
                dy = -dy
            while 1:
                if xflag:
                    x1 = px
                    y1 = int(py) >> shift
                else:
                    x1 = int(px) >> shift
                    y1 = py
                # check when line exits image boundary
                if x1 < 0 or x1 >= width or y1 < 0 or y1 >= height:
                    break
                gap += 1
                # if non-zero point found, continue the line
                if mask[y1, x1]:
                    gap = 0
                    line_end[k, 1] = y1
                    line_end[k, 0] = x1
                # if gap to this point was too large, end the line
                elif gap > line_gap:
                    break
                px += dx
                py += dy


        # confirm line length is acceptable
        acceptableLine = abs(line_end[1, 1] - line_end[0, 1]) >= line_length or \
                    abs(line_end[1, 0] - line_end[0, 0]) >= line_length

        # reset the accumulator and points on this line
        for k in range(2):
            px = x0
            py = y0
            dx = dx0
            dy = dy0
            if k > 0:
                dx = -dx
                dy = -dy
            while 1:
                if xflag:
                    x1 = px
                    y1 = int(py) >> shift
                else:
                    x1 = int(px) >> shift
                    y1 = py
                # if non-zero point found, continue the line
                if mask[y1, x1]:
                    if acceptableLine:
                        accum_idx = int(round((cosinTheta[j] * x1 + sinTheta[j] * y1)) + offset)
                        accum[accum_idx, max_theta] -= 1
                        mask[y1, x1] = 0
                # exit when the point is the line end
                if x1 == line_end[k, 0] and y1 == line_end[k, 1]:
                    break
                px += dx
                py += dy

        # add line to the result
        if acceptableLine:
            lines.append(((line_end[0, 0], line_end[0, 1]),
                          (line_end[1, 0], line_end[1, 1])))
            if len(lines) > lines_max:
                return lines

    return lines


def probabilistic_hough_line(img, threshold,
                              line_length, line_gap):
    """Return lines from a progressive probabilistic line Hough transform.
    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold : int
        Threshold
    line_length : int
        Minimum accepted length of detected lines.
        Increase the parameter to extract longer lines.
    line_gap : int
        Maximum gap between pixels to still form a line.
        Increase the parameter to merge broken lines more aggresively.
    theta : 1D ndarray, dtype=double
        Angles at which to compute the transform, in radians.
    Returns
    -------
    lines : list
      List of lines identified, lines in format ((x0, y0), (x1, y0)),
      indicating line start and end.
    """
    #create a list of theta values
    theta = np.pi / 2 - np.arange(180) / 180.0 * np.pi

    #get the dimensions of image
    height = img.shape[0]
    width = img.shape[1]


    #check if the image is too small for meaningful use
    if height < line_length or width < line_length :
        raise ValueError('Image is too small for processing.')

    lines = prob_getLine(img, threshold, line_length, line_gap, width, height, theta)
    if len(lines) > 5:
            return lines
    else:
        raise ValueError("Change parameter to do better line detection")

    return 0





def houghTransform(img):
    """
    The function performs hough transform on a image.
    Input:
        img: a image io object
    Output:
        accumulator: Hough transform accumulator
        theta : an vector of angles (radians) used in computation
        rho : an vector of rho values

    Note:
        it can only detect lines that are white
    """

    #initializing the values:
    theta = np.deg2rad(np.arange(-90, 90, 1)) #initializing a vector of angles in radians
    sinTheta = np.sin(theta)
    cosinTheta = np.cos(theta)
    imgWidth = img.shape [0]
    imgHeight = img.shape [1]
    imgDiagnal = int(math.sqrt(imgWidth * imgWidth + imgHeight * imgHeight)) #get the diagonal length of the image for initializing rho
    rho = np.linspace(-imgDiagnal, imgDiagnal, imgDiagnal*2) #initializing the rho values

    accumulator = np.zeros((2*imgDiagnal, len(theta)))
    points = [ [ 0] * len(theta)] *  (2* imgDiagnal)


    are_edges = img > 5 if True else img < value_threshold
    yXis, xXis = np.nonzero(are_edges)




    #doing hough transform
    for i in range(len(xXis)):
        currentX = xXis[i]
        currentY = yXis[i]

        #loop through all possible angles

        currentRhos = [] #have a rhos to check duplicate x, y
        for j in range(len(theta)):
            currentRho = imgDiagnal + int(currentX  * cosinTheta[j] + currentY*sinTheta[j])


            if points[currentRho][j] == 0 :
                points[currentRho][j] = [ ] * len(theta)

            if not currentRho in currentRhos:
                currentRhos.append(currentRho)
                points[currentRho][j].append([currentX, currentY])


            accumulator[currentRho, j] += 1


    return accumulator, points, theta, rho

def rgb2gray(rgb):

    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
