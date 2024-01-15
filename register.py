from detector import Detector
import cv2


def main():
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
