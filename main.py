import cv2
import time
import features_detection
import frame_class
import motion_plot
import motion_estimation

previous_frame = None


def start_motion_estimation(path):
    global previous_frame

    cap = cv2.VideoCapture(path)
    fps_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Reading ", path, " at ", frame_width, "x", frame_height, " @", fps_rate, "fps.")

    trj_plot = motion_plot.Trajectory("PREEMPTIVE RANSAC")

    frames_considered = 0
    time_snap = time.time()
    fps = 0
    processingTime = time.time()
    while cap.isOpened():
        succ, frame = cap.read()

        if succ:  # Frame was correctly acquired
            frames_considered += 1
            
            time_elapsed = time.time() - time_snap
            avg_fps = float("{0:.2f}".format(frames_considered/ time_elapsed))

            new_frame = frame_class.Frame(frame.copy(), 0 if previous_frame is None else previous_frame.get_frame_id()+1)
            new_frame.find_key_points()

            if previous_frame is not None:
                matched_features = features_detection.match_features(previous_frame, new_frame, False)
                hypothesis = motion_estimation.PREEMPTIVE_RANSAC_run(previous_frame, new_frame, matched_features)

                trj_plot.process_motion_hypothesis(hypothesis, matched_features)

                t = time.time()
                fps = round(1/(t-processingTime), 2)
                processingTime = t

            previous_frame = new_frame

            cv2.putText(frame, str(fps) + " FPS", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
            cv2.imshow("Untracked Features", frame)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_motion_estimation('media/test_1.mp4')
