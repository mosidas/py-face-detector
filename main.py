from detector import Detector
import cv2


def main():
    video = cv2.VideoCapture(0)
    detector = Detector()
    detector.train()
    # loop
    while True:
        # read frame
        ret, frame = video.read()
        # detect
        frame, faces = detector.detect(frame)
        # recognize
        frame = detector.recognize(frame, faces)
        # show
        cv2.imshow("frame", frame)
        # wait for key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
