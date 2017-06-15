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

    #tr1 = motion_plot.Trajectory("Original RANSAC")
    #tr2 = motion_plot.Trajectory("Ported RANSAC")
    tr3 = motion_plot.Trajectory("PREEMPTIVE RANSAC")

    frames_considered = 0
    time_snap = time.time()
    avg_fps = 0
    min_fps  = 100
    max_fps  = 0
    fps = 0
    processingTime = time.time()
    while cap.isOpened():
        succ, frame = cap.read()

        if succ: # Frame was correctly acquired
            frames_considered += 1
            
            time_elapsed = time.time() - time_snap
            avg_fps = float("{0:.2f}".format((frames_considered)/ time_elapsed))

            new_frame = frame_class.Frame(frame.copy(), 0 if previous_frame is None else previous_frame.get_frame_id()+1)
            new_frame.find_key_points()

            if previous_frame is not None:
                matched_features = features_detection.match_features(previous_frame, new_frame, False)
                #hypothesis3 = motion_estimation.PREEMPTIVE_RANSAC_run(previous_frame, new_frame, matched_features)
                import ransac
                hypothesis2 = ransac.RANSAC_run(matched_features)
                #hypothesis1 = cv2.findEssentialMat(np.array([x[0].pt for x in matched_features]),
                #                                  np.array([x[1].pt for x in matched_features]),
                #                                  motion_estimation.camera_matrix,
                #                                  cv2.RANSAC,
                #                                  0.999,
                #                                  1.0)[0]
                #tr1.process_motion_hypothesis(hypothesis1, matched_features)
                #tr2.process_motion_hypothesis(hypothesis2, matched_features)
                tr3.process_motion_hypothesis(hypothesis2, matched_features)
                t = time.time()
            	fps = round(1/(t-processingTime), 2)
            	processingTime = t
            	if fps < min_fps:
            		min_fps = fps
            	if fps > max_fps:
	            	max_fps = fps
            #cv2.putText(frame, str(avg_fps) + " FPS", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
            previous_frame = new_frame

            cv2.putText(frame, str(fps) + " FPS", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
            cv2.imshow("Untracked Features", frame)
        else:
        	break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    test = 'ported_ransac'
    with open(test+'.txt', 'w') as fp:
    	fp.write('Min FPS: '+str(min_fps)+'\n')
    	fp.write('Max FPS: '+str(max_fps)+'\n')
    	fp.write('Avg FPS: '+str(avg_fps)+'\n')
    	fp.write('Elapsed Time: '+str(round(time_elapsed, 2)))

    print "Min FPS: ", min_fps
    print "Max FPS: ", max_fps
    print "Avg FPS: ", avg_fps
    print "Elapsed Time: ", round(time_elapsed, 2)
    tr3.save(test+'.png')
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_motion_estimation('media/test_1.mp4')
