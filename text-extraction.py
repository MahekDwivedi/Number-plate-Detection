#install easyocr from command
!pip install easyocr

# import the necessary packages
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from easyocr import Reader
import argparse
import cv2

def plt_imshow(title, image):
	# convert the image frame BGR to RGB color space and display it
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.imshow(image)
	plt.title(title)
	plt.show()


#copy the image path
args = {
	"image": "/input/fifty-states-car-license-plates-dataset/fifty states license plates-20190703t183738z-001/Fifty States License Plates/Iowa.jpg",
	"langs": "en",
	"gpu": -1
}

# break the input languages into a comma separated list
langs = args["langs"].split()
print("[INFO] OCR'ing with the following languages: {}".format(langs))

# load the input image from disk
image = cv2.imread(args["image"])

# OCR the input image using EasyOCR
print("[INFO] OCR'ing input image...")
reader = Reader(langs)
results = reader.readtext(image)


#code to display the text extracted from the number plate after detection

# loop over the results
for (bbox, text, prob) in results:
	# display the OCR'd text and associated probability
	print()
	print("{:.3f}: {}".format(prob, text))

	# unpack the bounding box
	(tl, tr, br, bl) = bbox
	tl = (int(tl[0]), int(tl[1]))
	tr = (int(tr[0]), int(tr[1]))
	br = (int(br[0]), int(br[1]))
	bl = (int(bl[0]), int(bl[1]))
	
	cv2.rectangle(image, tl, br, (0, 255, 0), 2)
	cv2.putText(image, text, (tl[0], tl[1] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


plt_imshow("Image", image)

