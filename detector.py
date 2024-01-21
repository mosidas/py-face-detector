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
            # cv2.CascadeClassifier(
            #     os.path.join("resources", "lbpcascade_frontalface.xml")
            # ),
        ]
        radius = 1
        neighbors = 2
        grid_x = 8
        grid_y = 8
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius, neighbors, grid_x, grid_y
        )

    def regiter(self, img):
        # register face
        target = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = []
        for face_cascade in self.face_cascades:
            faces = face_cascade.detectMultiScale(img, 1.1, 5, 0, (50, 50))
            if len(faces) > 0:
                break

        # save face rect(size:100x100, file path: ./resources/face-yymmdd-hhmmssfff.jpg)
        for x, y, w, h in faces:
            # get face
            face = target[y : y + h, x : x + w]
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
        target = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # noise reduction
        target = cv2.equalizeHist(target)
        # gaussian blur
        # targetImg = cv2.GaussianBlur(targetImg, (5, 5), 0)

        for face_cascade in self.face_cascades:
            faces = face_cascade.detectMultiScale(target, 1.1, 6, 0, (40, 40))
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
            _, confidence = self.recognizer.predict(gray_face)
            # show
            cv2.putText(
                img,
                # 少数第二位まで表示
                "Confidence: {}".format(round(confidence, 2)),
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return img
