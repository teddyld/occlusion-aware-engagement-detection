
import utils.config as config
from albumentations.core.transforms_interface import ImageOnlyTransform
from . import functional
import os
import random

class ALOTDropout(ImageOnlyTransform):
    """ALOTDropout randomly drops out square regions from the image and fills 
    the dropout regions with textures from the ALOT dataset 
    to simulate artificial occlusion
    
    Args:
        data_path (string): Specifies the path that stores subdirectories containing ALOT images
        num_holes_range (Tuple[int, int]): Specifies the range (minmum and maximum) of the number of rectangular regions to zero out. This allows for dynamic variation in the number of regions removed per transformation instance.
        hole_height_range (Tuple[int, int]): Defines the minimum and maximum heights of the dropout regions, providing variability in their vertical dimensions
        hole_width_range (Tuple[int, int]): Defines the minimum and maximum widths of the dropout regions, providing variability in their horizontal dimensions
        p (float): Probability of applying the transform. Default: 0.5.
        
    Reference:
        https://doi.org/10.1016/j.patrec.2008.10.005
    """
    def __init__(self, data_path=config.ALOT_DATA_PATH, num_holes_range=(1, 1), hole_height_range=(8, 8), hole_width_range=(8, 8),  always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.data_path = data_path
        self.num_holes_range = num_holes_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range
        
        self.validate_data_path(self.data_path)
        self.validate_range(self.num_holes_range, 'num_holes_range', minimum=1)
        self.validate_range(self.hole_height_range, 'hole_height_range')
        self.validate_range(self.hole_width_range, 'hole_width_range')

    def apply(self, img, **params):
        holes = self.get_holes(img)
        return functional.alot_dropout(img, self.data_path, holes)

    def get_holes(self, img):
        height, width = img.shape[:2]
        holes = []
        num_holes = random.randint(self.num_holes_range[0], self.num_holes_range[1])
        
        for _ in range(num_holes):
            hole_height, hole_width = self.calculate_hole_dimensions(
                height,
                width,
                self.hole_height_range,
                self.hole_width_range,
            )
            
            y1 = random.randint(0, height - hole_height)
            x1 = random.randint(0, width - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            holes.append((x1, y1, x2, y2))
        
        return holes
    
    @staticmethod
    def calculate_hole_dimensions(height, width, height_range, width_range):
        """Calculate random hole dimensions based on the provided range"""
        if isinstance(height_range[0], int):
            min_height = height_range[0]
            max_height = height_range[1]

            min_width = width_range[0]
            max_width = width_range[1]
            max_height = min(max_height, height)
            max_width = min(max_width, width)
            hole_height = random.randint(int(min_height), int(max_height))
            hole_width = random.randint(int(min_width), int(max_width))

        else:  # Assume float
            hole_height = int(height * random.uniform(height_range[0], height_range[1]))
            hole_width = int(width * random.uniform(width_range[0], width_range[1]))
        return hole_height, hole_width
    
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
        
    @staticmethod
    # Validation for data path
    def validate_data_path(data_path):
        if not isinstance(data_path, str):
            raise TypeError(
                f"Input should be a valid string. Got: {type(data_path)}"
            )
        
        if not os.path.isdir(data_path):
            raise ValueError(f"data_path was not a valid directory")