try:
	import cv2 as cv
except:
	import cv

import imageExtractor
import numpy as np
import urllib
import xlrd
import os
 

def url_to_image(url):
	print(url)
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv.imdecode(image, cv.IMREAD_COLOR)
 
	
	return image
def getOnlineFaces(file,resultpath):
	workbook = xlrd.open_workbook(file)

	sheets = workbook.sheet_by_index(0)

	count = 0
	faces_found = 0
	for rowx in range(1,sheets.nrows):
		cols = sheets.row_values(rowx)
		try:

			faces = imageExtractor.getFaces(url_to_image(cols[0]))
			filename = resultpath + imageExtractor.getRandomFileName()
			while os.path.exists(filename):
				filename = resultpath + getRandomFileName()

			count += 1
			_,extension = os.path.splitext(cols[0])


			
			print(count, "images processed.")
			print(faces_found, "faces found.")


			if len(faces) > 1:
				faceCount = 1
				for face in faces:
					print(filename + "-" + str(faceCount) + extension)
					
					cv.imwrite(filename + "-" + str(faceCount) + extension, face)
					faceCount += 1
				faces_found += len(faces)
			elif len(faces) == 1:

				cv.imwrite(filename + extension, faces[0])
				faces_found += 1
		except:
			pass








getOnlineFaces("painting_dataset_2018.xlsx", "excel_images/")


