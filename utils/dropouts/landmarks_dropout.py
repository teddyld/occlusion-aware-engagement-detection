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
        fill_value (ColorType, Literal["random"]): Specifies the value used to fill the cropped regions. This can be a constant value, or 'random' which fills the region with random noise.
        p (float): Probability of applying the transform. Default: 0.5.
    
    """
    def __init__(self, landmarks, landmarks_weights=(1, 1, 1), fill_value=0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.landmarks = landmarks
        self.landmarks_weights = landmarks_weights
        self.fill_value = fill_value
        self.validate_landmarks(landmarks)
        self.validate_landmarks_weights(landmarks_weights)
        
    def apply(self, img, **params):
        feature = self.get_random_feature()
        for keypoint in self.landmarks:
            if not any(keypoint):
                return functional.landmarks_dropout(img, self.landmarks, feature, self.fill_value)
        # All keypoints are [0, 0] meaning that the image had no detected face
        return img
    
    def get_random_feature(self):
        """Get the randomly selected feature by choosing from the normalized probability of landmarks_weight"""
        feature_labels = ['eyes', 'nose', 'mouth']
        return random.choices(feature_labels, weights=self.landmarks_weights)[0]
    
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