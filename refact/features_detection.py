import cv2

features_detection_engine = cv2.ORB_create(nfeatures=650)
brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def discover_features(image, draw_key_points=False, key_points_color=(255, 0, 0)):
    key_points, key_points_desc = features_detection_engine.detectAndCompute(image, None)

    if draw_key_points:
        image_copy = image.copy()
        cv2.drawKeypoints(image_copy, key_points, image_copy, key_points_color)
        cv2.putText(image_copy, str(len(key_points)) + " FEATURES", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, key_points_color)
        cv2.imshow("Frame Features", image_copy)

    return key_points, key_points_desc


def brute_force_match(frame_1, frame_2):
    matches = brute_force_matcher.match(frame_1.get_key_points_descriptors(), frame_2.get_key_points_descriptors())

    frame_1_key_points = frame_1.get_key_points()
    frame_2_key_points = frame_2.get_key_points()

    ret = []
    for m in matches:
        ret.append((frame_1_key_points[m.queryIdx], frame_2_key_points[m.trainIdx]))

    return ret


def match_features(frame_1, frame_2, draw_matches=False, matches_color=(0, 255, 0)):
    assert(frame_1 is not None and frame_2 is not None)

    matched_features = brute_force_match(frame_1, frame_2)
    print("Matched ", len(matched_features), "/", len(frame_2.get_key_points()), " features btwn frame ", frame_1.get_frame_id(), "-", frame_2.get_frame_id())

    if draw_matches:
        img_copy = frame_2.get_image().copy()
        cv2.drawKeypoints(img_copy, [x[1] for x in matched_features], img_copy, matches_color)
        cv2.putText(img_copy, str(len(matched_features)) + " MATCHES", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, matches_color)
        cv2.imshow('Matched Features', img_copy)

    return matched_features
