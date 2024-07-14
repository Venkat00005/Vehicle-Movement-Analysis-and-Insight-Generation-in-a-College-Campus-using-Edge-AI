# importing libraries
import easyocr
import imutils
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
import csv
import warnings
warnings.filterwarnings("ignore")

class NumberPlateRecognizer:
    def __init__(self, image_path):
        
        # Read the image using OpenCV
        self.image_path = cv2.imread(image_path)  

    def Retrieve_Number_plate(self):

        # Convert image to grayscale
        gray = cv2.cvtColor(self.image_path, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filtering for noise reduction
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

        # Apply Canny edge detection to find license plate edges
        edged = cv2.Canny(bfilter, 30, 200)

        # Find contours in the edge image
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        plate_location = None

        # Identify the quadrilateral contour representing the license plate
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4: 
                plate_location = approx
                break

        # Create a mask to isolate the license plate region
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [plate_location], 0, 255, -1)
        new_image = cv2.bitwise_and(self.image_path, self.image_path, mask=mask)

        # Extract coordinates of the license plate bounding box
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))

        # Crop the image based on the bounding box
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        # Use EasyOCR to perform OCR on the cropped image
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)  # English language, no GPU, quiet mode
        result = reader.readtext(cropped_image)

        # Extract and clean the recognized text
        text = result[0][-2]
        text = text.replace('.', "")
        text = text.replace(' ', "")

        # Print and return the recognized license plate number
        return text[1:].upper()