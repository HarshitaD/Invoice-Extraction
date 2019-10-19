import cv2
import numpy as np
import argparse

def show_wait_destroy(img, winname='default', seconds=0):
	cv2.imshow(winname, img)
	cv2.moveWindow(winname, 5, 0)
	cv2.waitKey(seconds)
	cv2.destroyWindow(winname)
# Generally paper (edges, at least) is white, so you may have better luck by going to a colorspace like YUV which better separates luminosity:
def get_xs(jpg_path, bbox):
	image = cv2.imread(jpg_path)

	imageCopy = image.copy()
	x1,y1,x2,y2=bbox

	image_y = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh_block_size = int(image_y.shape[1]/100)
	if thresh_block_size%2==0:
		thresh_block_size+=1
	# The text on the paper is a problem. Use a blurring effect, to (hopefully) remove these high frequency noises. You may also use morphological operations like dilation as well.

	# image_y = cv2.GaussianBlur(image_y,(3,3),0)
	image_y = cv2.bitwise_not(image_y)
	image_y = cv2.adaptiveThreshold(image_y, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
									cv2.THRESH_BINARY, thresh_block_size, -2)
	# show_wait_destroy(image_y)
	# Dilation and Erosion
	horizontal = np.copy(image_y)
	# [init]
	# [horiz]
	# Specify size on horizontal axis
	cols = horizontal.shape[1]
	horizontal_size = cols // 20
	# Create structure element for extracting horizontal lines through morphology operations
	horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
	horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 2))
	# Apply morphology operations
	horizontal = cv2.erode(horizontal, horizontalStructure)
	horizontal = cv2.dilate(horizontal, horizontalStructure)
	horizontal = cv2.dilate(horizontal, horizontalStructure)
	horizontal = cv2.dilate(horizontal, horizontalStructure)
	horizontal = cv2.dilate(horizontal, horizontalStructure)
	horizontal = cv2.dilate(horizontal, horizontalStructure)

	# Show extracted horizontal lines
	# show_wait_destroy(horizontal)

	# [horiz]
	# [vert]
	vertical = np.copy(image_y)
	# Specify size on vertical axis
	rows = vertical.shape[0]
	verticalsize = rows // 60
	# Create structure element for extracting vertical lines through morphology operations
	verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
	verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (2, verticalsize))
	# Apply morphology operations
	vertical = cv2.erode(vertical, verticalStructure)
	vertical = cv2.dilate(vertical, verticalStructure)
	vertical = cv2.dilate(vertical, verticalStructure)
	vertical = cv2.dilate(vertical, verticalStructure)
	vertical = cv2.dilate(vertical, verticalStructure)
	# Show extracted vertical lines
	# show_wait_destroy(vertical)

	# vertical = cv2.bitwise_not(vertical)
	# show_wait_destroy(vertical, 'vertical' )


	# Step 1
	edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
								cv2.THRESH_BINARY, 3, -2)
	# show_wait_destroy(edges, "edges")
	# Step 2
	kernel = np.ones((2, 2), np.uint8)
	edges = cv2.dilate(edges, kernel)
	# show_wait_destroy(edges, "dilate")
	# Step 3
	smooth = np.copy(vertical)
	# Step 4
	smooth = cv2.blur(smooth, (2, 2))
	# Step 5
	(rows, cols) = np.where(edges != 0)
	vertical[rows, cols] = smooth[rows, cols]
	# Show final result
	# show_wait_destroy(vertical, "smooth - final")

	# You may try to apply a canny edge-detector, rather than a simple threshold. Not necessarily, but may help you:
	image_blurred = cv2.bitwise_and(horizontal, vertical)
	# show_wait_destroy(image_blurred)

	# import numpy as np
	# import cv2

	# # Create a black image
	# img = np.zeros((512,512,3), np.uint8)

	# # Draw a diagonal blue line with thickness of 5 px
	# img = cv2.line(img,(0,0),(511,511),(255,0,0),5)
	# show_wait_destroy(img, "line contour")



	# x1,y1,x2,y2=[4730, 185, 6490, 559]
	h, w, _ = imageCopy.shape # assumes color image
	w_old=10131
	h_old=14506
	x_ratio = w/w_old
	y_ratio = h/h_old
	x1=x1*x_ratio
	x2=x2*x_ratio
	y1=y1*y_ratio
	y2=y2*y_ratio

	dx = x2-x1
	dy=y2-y1

	x1 -= dx/2
	x2 += dx/2
	y1 -=  dy/2
	y2 +=  dy/2
	x1=int(x1)
	x2=int(x2)
	y1=int(y1)
	y2=int(y2)

	p1=(x1, y1)
	p2=(x2, y2)

	# imageCopy=cv2.rectangle(imageCopy, p1, p2 ,(255,0,0), 5)
	imageCopy=cv2.line(imageCopy, (x1,y1), (x2,y1) ,(255,0,0), 5)
	imageCopy=cv2.line(imageCopy, (x1,y2), (x2,y2) ,(255,0,0), 5)



	tight = cv2.Canny(image_blurred, 225, 250)

	# show the images
	# show_wait_destroy(np.hstack([wide, tight, auto]), 'edges')
	# Then find the contours. In my case I only used the extreme outer contours. You may use CHAIN_APPROX_SIMPLE flag to compress the contour

	edges = tight
	contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	table_xs=[]
	# loop over the contours
	for c in contours:
		# compute the center of the contour
		# M = cv2.moments(c)
		# cX = int(M["m10"] / M["m00"])
		# cY = int(M["m01"] / M["m00"])
		cX=sum(x[0][0] for x in c)/len(c)
		cY=sum(x[0][1] for x in c)/len(c)
		cX = int(cX)
		cY = int(cY)
		if x1<cX<x2 and y1<cY<y2:
			table_xs.append((cX,cY))
			# draw the contour and center of the shape on the image
			# cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
			imageCopy = cv2.circle(imageCopy, (cX, cY), 7, (255, 255, 255), -1)
			# cv2.putText(image, "center", (cX - 20, cY - 20),
				# cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	def cv2_to_hocr(p):
		x,y=p
		x=x/x_ratio
		y=y/y_ratio

		x=int(x)
		y=int(y)
		p=(x,y)
		return p
	table_xs=[cv2_to_hocr(x) for x in table_xs]
	# freq=collections.Counter([x[0] for x in table_xs])
	# sorted(freq.keys())
	# for key, value in freq.items(): 
	# 	print(key, " -> ", value)
	return table_xs


# if __name__ == "__main__":
# 	parser = argparse.ArgumentParser(
# 		description=('extract the text within all the ocr_line elements '
# 					'within the hOCR file')
# 	)
# 	parser.add_argument('file', nargs='?', default=sys.stdin)
# 	args = parser.parse_args()
