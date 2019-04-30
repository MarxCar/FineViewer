import os

import cv2 as cv
import numpy as np 
import random
import string

imagepath = "./test/test/"
#face_classifier = cv.CascadeClassifier("/home/sam/anaconda3/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt.xml")
net = cv.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
filename_length = 15


def getRandomFileName():
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(filename_length))


def getFaces(image, conf = 0.5):
    """
    (String, Int) -> (faces[numpy])
    Precondition: path denotes an image that exists
    Returns a list of faces found in the image if none
    are found returns and empty list
    """
    

    
    (h, w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
    net.setInput(blob)
    detections = net.forward()

    face_arr = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > conf:

            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            
            t = image[max(0, int(startY*0.9)):min(int(endY*1.1),image.shape[0]), max(0, int(startX*0.9)):min(int(endX*1.1), image.shape[1])]
            if t.shape[0] == 0 or t.shape[1] == 0:
                pass
            else:
                face_arr.append(t)

    

    return face_arr
def getImage(path):
    return cv.imread(path)

def removeDuplicates(targetdir):
    """
    (String) -> (None)
    Precondition: targetDir is the path to a directory
    """
    if not os.path.exists(targetdir):
        print("Target directory does not exist")

    files = os.listdir(targetdir)

    for indexX, fileX in enumerate(files):
        if os.path.isfile(targetdir + fileX):
            imgX = cv.imread(targetdir + fileX)

            for indexCompare, fileCompare in enumerate(files[indexX+1:]):
                if os.path.isfile(targetdir + fileCompare):
                    imgCompare = cv.imread(targetdir + fileCompare)

                    if imgX.shape == imgCompare.shape and np.linalg.norm(imgCompare - imgX) == 0.0:
                            # remove the duplicate 
                        print("Removing file " + targetdir + fileCompare)
                        os.remove(targetdir + fileCompare)
                    






def imageExtractor(imagepath, resultpath):
    """
    (String, String, Int()) -> None
    Precondition: imagepath and resultpath exist
    Extracts all faces found in the images from imagepath 
    and puts them as seperate images in resultpath
    """
    if not os.path.exists(imagepath):
        print("Image path does not exist")
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)
    files = os.listdir(imagepath)
    
    count = 0
    corruptedCount = 0
    
    no_faces_found = 0
    faces_found = 0



    for file in files:
        
        try: 
            
            faces = getFaces(getImage(imagepath + file))
            filename = resultpath + getRandomFileName()
            while os.path.exists(filename):
                    filename = resultpath + getRandomFileName()
            count += 1
            if count % 100 == 0: 
                print(count, "images processed.")
                print(faces_found, "faces found.")
           
            if len(faces) > 1:
                faceCount = 1
                
                
                
                    
                name, extension = os.path.splitext(resultpath + file)
                
                for face in faces:

                    
                    print(filename +  "-" + str(faceCount) + extension)
                    cv.imwrite(filename +  "-" + str(faceCount) + extension, face)
                    faceCount += 1
                faces_found += len(faces)
            elif len(faces) == 1:
                
                cv.imwrite(filename + extension,faces[0])
                faces_found += 1
            else:
                no_faces_found += 1

        except:
            corruptedCount += 1

    print("Found", count, "usable images and", corruptedCount, "corrupted files")
    print(str(no_faces_found) + " files did not have faces. ")
    print("A total of " + str(faces_found) + " faces.")



