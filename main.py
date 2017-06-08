import cv2
import time
import features_detection
import frame_class
import motion_estimation
import numpy as np
import motion_plot

previous_frame = None


def start_motion_estimation(path):
    global previous_frame

    cap = cv2.VideoCapture(path)
    fps_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Reading ", path, " at ", frame_width, "x", frame_height, " @", fps_rate, "fps.")

    tr1 = motion_plot.Trajectory("Ported RANSAC")
    tr2 = motion_plot.Trajectory("Original RANSAC")

    frames_considered = 0
    time_snap = time.time()
    current_fps = 0
    while cap.isOpened():
        succ, frame = cap.read()

        if succ: # Frame was correctly acquired
            frames_considered += 1
            
            # skip first 300 frames to avoid hand-shaking when video started
            if frames_considered < 300:
            	continue

            time_elapsed = time.time() - time_snap
            current_fps = float("{0:.2f}".format((frames_considered - 300)/ time_elapsed))

            new_frame = frame_class.Frame(frame.copy(), 0 if previous_frame is None else previous_frame.get_frame_id()+1)
            new_frame.find_key_points()

            if previous_frame is not None:
                matched_features = features_detection.match_features(previous_frame, new_frame, False)
                #hypothesis1 = motion_estimation.PREEMPTIVE_RANSAC_run(previous_frame, new_frame, matched_features)
                import ransac
                hypothesis1 = ransac.RANSAC_run(matched_features)
                #hypothesis2 = cv2.findEssentialMat(np.array([x[0].pt for x in matched_features]),
                #                                  np.array([x[1].pt for x in matched_features]),
                #                                  motion_estimation.camera_matrix,
                #                                  cv2.RANSAC,
                #                                  0.999,
                #                                  1.0)[0]
                tr1.process_motion_hypothesis(hypothesis1, matched_features)
                #tr2.process_motion_hypothesis(hypothesis2, matched_features)
                
            cv2.putText(frame, str(current_fps) + " FPS", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
            cv2.imshow("Untracked Features", frame)

            previous_frame = new_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_motion_estimation('media/test_2.mp4')
