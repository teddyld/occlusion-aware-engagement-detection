from albumentations.core.transforms_interface import ImageOnlyTransform
from . import functional
import random

class LandmarksDropout(ImageOnlyTransform):
    """LandmarksDropout randomly drops out rectangular regions from the image
    eyes, nose, or mouth features to simulate occlusion from facial 
    accessories or external objects
    
    Args:
        landmarks (List[List[int, int], ...]): Specifies the list of keypoints in order of the 'left_eye', 'right_eye', 'nose', 'left_mouth' and 'right_mouth' labels in the xy format.
        landmarks_weight (Tuple[ScalarType, ScalarType, ScalarType]): Defines the weighted probability of a feature being chosen for dropout in order of (eyes, nose, mouth). The weights are normalized so their probabilities add up to 1.
        dropout_height_range (Tuple[int, int]): Defines the minimum and maximum pixels of the dropout height, providing variability in dropout size.
        fill_value (ColorType, Literal["random"]): Specifies the value used to fill the cropped regions. This can be a constant value, or 'random' which fills the region with random noise.
        p (float): Probability of applying the transform. Default: 0.5.
    
    """
    def __init__(self, landmarks, landmarks_weights=(1, 1, 1), dropout_height_range=(8, 8), fill_value=0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.landmarks = landmarks
        self.landmarks_weights = landmarks_weights
        self.dropout_height_range = dropout_height_range
        self.fill_value = fill_value
        self.validate_landmarks(landmarks)
        self.validate_landmarks_weights(landmarks_weights)
        self.validate_range(dropout_height_range, 'dropout_height_range')
        
    def apply(self, img, **params):
        height, _ = img.shape[:2]
        dropout_height = self.calculate_dropout_height(
            height,
            self.dropout_height_range,
        )

        feature = self.get_random_feature()
        for keypoint in self.landmarks:
            if not any(keypoint):
                return functional.landmarks_dropout(img, self.landmarks, feature, dropout_height, self.fill_value)
        # All keypoints are [0, 0] meaning that the image had no detected face
        return img
    
    def get_random_feature(self):
        """Get the randomly selected feature by choosing from the normalized probability of landmarks_weight"""
        feature_labels = ['eyes', 'nose', 'mouth']
        return random.choices(feature_labels, weights=self.landmarks_weights)[0]
    
    @staticmethod
    def calculate_dropout_height(height, height_range):
        """Calculate random dropout height based on the provided range"""
        if isinstance(height_range[0], int):
            min_height = height_range[0]
            max_height = height_range[1]

            max_height = min(max_height, height)
            dropout_height = random.randint(int(min_height), int(max_height))

        else:  # Assume float
            dropout_height = int(height * random.uniform(height_range[0], height_range[1]))
        return dropout_height
    
    @staticmethod
    # Validation for landmarks
    def validate_landmarks(landmarks):
        if not isinstance(landmarks, tuple):
            return ValueError(f"Expected landmarks to be a tuple of keypoints in xy format. Got: {type(landmarks)} {landmarks}")
        
    @staticmethod
    # Validation for landmarks weights
    def validate_landmarks_weights(landmarks_weights):
        if not isinstance(landmarks_weights, tuple) or not len(landmarks_weights) == 3:
            raise ValueError(f"landmarks_weights expected a tuple of four numbers. Got: {landmarks_weights}")
        
        for num in landmarks_weights:
            if not isinstance(num, (int, float)):
                raise ValueError(f"{num} of type {type(num)} is not a valid weight. Expected {num} to be of type int or float")
            
    @staticmethod
    # Validation for dropout dimension ranges
    def validate_range(range_value, range_name, minimum=0):
        if not minimum <= range_value[0] <= range_value[1]:
            raise ValueError(
                f"First value in {range_name} should be less or equal than the second value "
                f"and at least {minimum}. Got: {range_value}",
            )
        if isinstance(range_value[0], float) and not all(0 <= x <= 1 for x in range_value):
            raise ValueError(f"All values in {range_name} should be in [0, 1] range. Got: {range_value}")