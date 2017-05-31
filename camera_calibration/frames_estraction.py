import cv2

def play_video(path):
    cap = cv2.VideoCapture(path)
    i = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            cv2.imwrite("../media/camera_calibration_frames/Frame_" + str(i) + ".jpg", frame);
            i = i+1
            cv2.imshow('untracked_features', frame)
        else:
            break

        if cv2.waitKey( 1 ) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_video('../media/camera_calibration.mp4')