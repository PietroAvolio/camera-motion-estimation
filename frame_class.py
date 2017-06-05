import features_detection


class Frame:
    def __init__(self, image, frame_id, key_points = None, key_points_descriptors = None):
        self.key_points = key_points
        self.key_points_descriptors = key_points_descriptors
        self.frame_id = frame_id
        self.image = image

    def get_key_points(self):
        return self.key_points

    def get_frame_id(self):
        return self.frame_id

    def get_key_points_descriptors(self):
        return self.key_points_descriptors

    def get_image(self):
        return self.image

    def find_key_points(self):
        key_points, key_points_descriptors = features_detection.discover_features(self.image, False)

        self.key_points = key_points
        self.key_points_descriptors = key_points_descriptors
