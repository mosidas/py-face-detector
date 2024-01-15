import glob
import os
from detector import Detector
import cv2


def main():
    # delete previous face images
    filepaths = glob.glob(os.path.join("resources", "faces", "*"))
    for filepath in filepaths:
        os.remove(filepath)

    video = cv2.VideoCapture(0)
    detector = Detector()
    # loop 50 times
    for i in range(50):
        # read frame
        ret, frame = video.read()
        # register
        frame = detector.regiter(frame)
        # show
        cv2.imshow("frame", frame)
        # wait for key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
