import cv2
import os
import random
import matplotlib.pyplot as plt
from imageai.Detection import ObjectDetection

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath('tiny-yolov3.pt')
detector.loadModel()

peopleImages = os.listdir("images")
randomFile = peopleImages[random.randint(0, len(peopleImages) - 1)]

detectedImage, detections = detector.detectObjectsFromImage(output_type="array", input_image="images/{0}".format(randomFile), minimum_percentage_probability=30)
convertedImage = cv2.cvtColor(detectedImage, cv2.COLOR_RGB2BGR)

plt.figure(figsize=(10,10))
plt.imshow(convertedImage)
plt.axis('off')
plt.show()

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")