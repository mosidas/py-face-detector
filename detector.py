import datetime
import cv2
import os
import numpy as np


class Detector:
    def __init__(self):
        # file path: ./resources/*
        self.face_cascades = [
            cv2.CascadeClassifier(
                os.path.join("resources", "haarcascade_frontalface_alt.xml")
            ),
            cv2.CascadeClassifier(
                os.path.join("resources", "haarcascade_frontalface_alt2.xml")
            ),
            cv2.CascadeClassifier(
                os.path.join("resources", "haarcascade_frontalface_alt_tree.xml")
            ),
        ]
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def regiter(self, img):
        # # register face
        # targetImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # # noise reduction
        # targetImg = cv2.equalizeHist(targetImg)
        # # gaussian blur
        # targetImg = cv2.GaussianBlur(targetImg, (5, 5), 0)

        faces = []
        for face_cascade in self.face_cascades:
            faces = face_cascade.detectMultiScale(img, 1.05, 5, 0, (40, 40))
            if len(faces) > 0:
                break

        # save face rect(size:100x100, file path: ./resources/face-yymmdd-hhmmssfff.jpg)
        for x, y, w, h in faces:
            # get face
            face = img[y : y + h, x : x + w]
            # resize
            face = cv2.resize(face, (100, 100))
            # save
            cv2.imwrite(
                os.path.join(
                    "resources",
                    "faces",
                    "face-{}.jpg".format(
                        datetime.datetime.now().strftime("%y%m%d-%H%M%S%f")
                    ),
                ),
                face,
            )

        return img

    def detect(self, img):
        targetImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # noise reduction
        targetImg = cv2.equalizeHist(targetImg)
        # gaussian blur
        targetImg = cv2.GaussianBlur(targetImg, (5, 5), 0)

        for face_cascade in self.face_cascades:
            faces = face_cascade.detectMultiScale(targetImg, 1.05, 5, 0, (40, 40))
            if len(faces) > 0:
                for x, y, w, h in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                break

        return img, faces

    def train(self):
        path = os.path.join("resources", "faces")
        # train
        faces = []
        labels = []
        # train files path: ./resources/faces/face-*.jpg
        for file in os.listdir(path):
            image_path = os.path.join(path, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces.append(image)
            labels.append(0)
        # save trained data
        self.recognizer.train(faces, np.array(labels))

    def recognize(self, img, faces):
        # recognize face
        for x, y, w, h in faces:
            # get face
            face = img[y : y + h, x : x + w]

            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # resize
            gray_face = cv2.resize(gray_face, (100, 100))
            # recognize
            label, confidence = self.recognizer.predict(gray_face)
            # show
            cv2.putText(
                img,
                "Confidence: {}".format(confidence),
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return img
