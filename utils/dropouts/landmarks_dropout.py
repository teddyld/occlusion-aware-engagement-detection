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
        dropout_width_range (Tuple[int, int]): Defines the minimum and maximum pixels of the dropout width, providing variability in dropout size.
        fill_value (ColorType, Literal["random"]): Specifies the value used to fill the cropped regions. This can be a constant value, or 'random' which fills the region with random noise.
        inverse (bool): Defines whether to apply the dropout to regions outside of the facial features. If True, zeroes the probability of applying dropout to the 'nose' landmark.
        p (float): Probability of applying the transform. Default: 0.5.
    
    """
    def __init__(self, landmarks, landmarks_weights=(1, 1, 1), dropout_height_range=(4, 4), dropout_width_range=(4, 4), fill_value=0, inverse=False, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.landmarks = landmarks
        
        if inverse:
            landmarks_weights = list(landmarks_weights)
            landmarks_weights[1] = 0
            landmarks_weights = tuple(landmarks_weights)

        self.landmarks_weights = landmarks_weights
        self.dropout_height_range = dropout_height_range
        self.dropout_width_range = dropout_width_range
        self.fill_value = fill_value
        self.inverse = inverse
        self.validate_landmarks(landmarks)
        self.validate_landmarks_weights(landmarks_weights)
        self.validate_range(dropout_height_range, 'dropout_height_range')
        self.validate_range(dropout_width_range, 'dropout_width_range')

        
    def apply(self, img, **params):
        height, width = img.shape[:2]
        proposed_height, proposed_width = self.calculate_dropout_dimensions(
            height,
            width,
            self.dropout_height_range,
            self.dropout_width_range,
        )
        
        feature, landmarks = self.get_random_feature_landmarks()
        
        dropout_height, dropout_width = self.calculate_dropout_limits(
            height,
            width,
            proposed_height,
            proposed_width,
            feature,
            landmarks,
        )

        zero_count = 0
        for keypoint in self.landmarks:
            x, y = keypoint[0], keypoint[1]
            # Sometimes CompreFace returns landmarks outside of the range of the image
            if x < 0 or x > width or y < 0 or y > height:
                return img
            
            if not any(keypoint):
                zero_count += 1
                
        # All keypoints are [0, 0] meaning that the image had no detected face
        if zero_count == len(landmarks):
            return img
        else:
        # Face detected
            return functional.landmarks_dropout(img, landmarks, feature, dropout_height, dropout_width, self.fill_value, self.inverse)
    
    def get_random_feature_landmarks(self):
        """Get the randomly selected feature and its landmarks by choosing from the normalized probability of landmarks_weight"""
        feature_labels = ['eyes', 'nose', 'mouth']
        feature = random.choices(feature_labels, weights=self.landmarks_weights)[0]
        if feature == 'eyes':
            landmarks = [self.landmarks[0], self.landmarks[1]]
        elif feature == 'nose':
            landmarks = [self.landmarks[2]]
        else:
            landmarks = [self.landmarks[3], self.landmarks[4]]
        return feature, landmarks
    
    @staticmethod
    def calculate_dropout_dimensions(height, width, height_range, width_range):
        """Calculate random dropout height based on the provided range"""
        if isinstance(height_range[0], int):
            min_height = height_range[0]
            max_height = height_range[1]

            min_width = width_range[0]
            max_width = width_range[1]
            max_height = min(max_height, height)
            max_width = min(max_width, width)
            dropout_height = random.randint(int(min_height), int(max_height))
            dropout_width = random.randint(int(min_width), int(max_width))

        else:  # Assume float
            dropout_height = int(height * random.uniform(height_range[0], height_range[1]))
            dropout_width = int(width * random.uniform(width_range[0], width_range[1]))
        return dropout_height, dropout_width
    
    @staticmethod
    def calculate_dropout_limits(height, width, proposed_height, proposed_width, feature, landmarks):
        """Calculate the dropout dimension limits based on the proposed dimensions and feature landmarks to avoid list index out of range error"""
        # Dropout dimensions for eyes and mouth features
        if feature in ('eyes', 'mouth'):
            left_key, right_key = landmarks[0], landmarks[1]
            
            # Width dimensions
            left_limit, right_limit = proposed_width, proposed_width
            if left_key[0] - proposed_width < 0:
                left_limit = left_key[0]
            
            if right_key[0] + proposed_width > width:
                right_limit = width - right_key[0]      

            dropout_width = min(left_limit, right_limit)
            
            # Height dimensions
            left_upper_limit, right_upper_limit = proposed_height, proposed_height
            if left_key[1] - proposed_height < 0:
                left_upper_limit = left_key[1]
            
            if right_key[1] - proposed_height < 0:
                right_upper_limit = right_key[1]      

            upper_limit = min(left_upper_limit, right_upper_limit)

            left_lower_limit, right_lower_limit = proposed_height, proposed_height
            if left_key[1] + proposed_height > height:
                left_lower_limit = height - left_key[1]
            
            if right_key[1] + proposed_height > height:
                right_lower_limit = height - right_key[1]
                
            lower_limit = min(left_lower_limit, right_lower_limit)
            
            dropout_height = min(upper_limit, lower_limit)    
            
        else: # Dropout dimensions for nose feature
            key = landmarks[0]
            # Width dimensions
            left_limit, right_limit = proposed_width, proposed_width
            if key[0] - proposed_width < 0:
                left_limit = key[0]
            
            if key[0] + proposed_width > width:
                right_limit = width - key[0]      

            dropout_width = min(left_limit, right_limit)
            
            # height dimensions
            upper_limit, lower_limit = proposed_height, proposed_height
            if key[1] - proposed_height < 0:
                upper_limit = key[1]

            if key[1] + proposed_height > height:
                lower_limit = height - key[1]

            dropout_height = min(upper_limit, lower_limit)
            
        return dropout_height, dropout_width
    
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