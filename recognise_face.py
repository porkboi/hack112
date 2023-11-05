import cv2
import matplotlib.pyplot as plt
import json

def read_json_file(key):
    with open("output/outputcoords.json", "r+") as f:
        data = json.load(f)
        if key not in data:
            data[key] = {"names":[], "value":[]}
            print("Added")
            f.seek(0)
            json.dump(data, f)
            f.truncate()
            return {}
        return data.get(key)

#def add_data_to_file(key, value):
    # Load the existing data from the file
    #with open("fakeDb.json", "r+") as f:
        #data = json.load(f)

def update_json_file(key, value):
    # Load the JSON data from the file
    with open("output/outputcoords.json", "r+") as f:
        data = json.load(f)

    # Update the value for the specified key
    data[key] = value

    # Write the updated data back to the file
    with open("output/outputcoords.json", "r+") as f:
        json.dump(data, f)
        f.truncate()

def labelImage(st):
    imagePath = st
    img = cv2.imread(imagePath)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
    )

    for i in range(len(face)):
        (x, y, w, h) = face[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cropimg = img[y:y+h, x:x+w]
        croppedimg_rgb = cv2.cvtColor(cropimg, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10,10))
        plt.imshow(croppedimg_rgb)
        plt.axis('off')
        #plt.show()
        plt.savefig(f"dataset/{st[7:-5] + str(i)}.png")

    y = face.tolist()
    c = []
    for i in range(len(y)):
        f = [f"person {i}"] + y[i]
        c += f
    update_json_file(st[7:-5], c)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10,10))
    plt.imshow(img_rgb)
    plt.axis('off')
    #plt.show()
    plt.savefig("output/outputimage.png")

def main(r):
    st = "images/" + r
    labelImage(st)

main("photo16991319178.jpeg")
