import cv2
import argparse
import imutils
import numpy as np
import math

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
        vtc = len(approx)
        agl = []

        print(len(approx))
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif 4 <= len(approx) <= 12:

            for i in range(2, vtc + 1):
                agl.append(angle(approx[i % vtc], approx[i - 2], approx[i - 1]))
            agl.sort()

            minagl = agl[0]
            maxagl = agl[-1]
            # compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            
            # a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
            if vtc == 4 and minagl >= -0.15 and maxagl <= 0.15 and  0.95 <=  ar <= 1.05:
                shape = "square"
                
            elif vtc == 4 and minagl >= -0.15 and maxagl <= 0.15:
                shape = "rectangle"
            
            elif vtc == 4 and minagl <= -0.2:
                shape = "trapezoid"

            elif vtc == 5 and minagl >= -0.5 and maxagl <= -0.05:
                shape = "pentagon"

            elif vtc == 5 and minagl >= -0.85 and maxagl <= 0.6:
                shape = "semi-circle"

            elif vtc == 5 and minagl >= -0.1 and maxagl <= 0.6:
                shape = "quarter-circle"

            elif vtc == 6 and minagl >= -0.7 and maxagl <= -0.3:
                shape = "hexa"

            elif vtc == 6 and minagl >= -0.85 and maxagl <= 0.6:
                shape = "semi-circle"
            
            elif vtc == 6 and minagl >= -1 and maxagl <= 0.6:
                shape = "quarter-circle" 
            
            elif vtc == 7:
                shape = "hepta"

            elif vtc == 8:
                area = cv2.contourArea(c)
                r = w / 2

                if abs(1 - (w / h)) <= 0.2 and abs(1 - (area / (np.pi * math.pow(r, 2)))) <= 0.2:
                    shape = "circle"
                
                else:
                    shape = "octa"
            
            elif vtc == 10:
                shape = "star"

            elif vtc == 12:
                shape = "cross"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return name of the shape
        return shape

def angle(pt1, pt2, pt0):	
	dx1 = pt1[0][0] - pt0[0][0]
	dy1 = pt1[0][1] - pt0[0][1]
	dx2 = pt2[0][0] - pt0[0][0]
	dy2 = pt2[0][1] - pt0[0][1]

	return (dx1 * dx2 + dy1 * dy2) / math.sqrt(abs((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=800)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image and initialize the
# shape detector
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

sd = ShapeDetector()

# loop over the contours
for c in contours:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)
 
	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (255, 255, 255), 2)
 
	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)