from albumentations.core.transforms_interface import ImageOnlyTransform
from . import functional
import random

class EdgeDropout(ImageOnlyTransform):
    """EdgeDropout drops out a random edge (top, left, right, or bottom) of the image and fills the dropout region with a constant value or random noise.
    
    Args:
        edge_height_range (Tuple[int, int]): Defines the minimum and maximum pixels of the edge height dropout, providing variability in dropout size from the top and bottom
        edge_width_range (Tuple[int, int]): Defines the minimum and maximum pixels of the edge width dropout, providing variability in dropout size from the left and right
        edge_weights (Tuple[ScalarType, ScalarType, ScalarType, ScalarType]): Defines the weighted probability of each side being chosen for dropout in order of (top, right, bottom, left). The weights are normalized so their probabilities add up to 1.
        fill_value (ColorType, Literal["random"]): Specifies the value used to fill the cropped regions. This can be a constant value, or 'random' which fills the region with random noise.
        p (float): Probability of applying the transform. Default: 0.5.
    
    """
    def __init__(self, edge_height_range=(12, 12), edge_width_range=(12, 12), edge_weights=(1, 1, 1, 1), fill_value=0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.edge_height_range = edge_height_range
        self.edge_width_range = edge_width_range
        self.edge_weights = edge_weights
        self.fill_value = fill_value
        
        self.validate_range(edge_height_range, "edge_height_range")
        self.validate_range(edge_width_range, "edge_width_range")
        self.validate_edge_weights(edge_weights)

    def apply(self, img, **params):
        height, width = img.shape[:2]
        edge_height, edge_width = self.calculate_edge_size(
            height,
            width,
            self.edge_height_range,
            self.edge_width_range,
        )
        edge_name = self.get_random_edge()
        return functional.edge_dropout(img, edge_height, edge_width, edge_name, self.fill_value)

    def get_random_edge(self):
        """Get the randomly selected edge by choosing from the normalized probability of edge_weights"""
        edge_labels = ['top', 'left', 'right', 'bottom']
        return random.choices(edge_labels, weights=self.edge_weights)[0]
    
    @staticmethod
    def calculate_edge_size(height, width, height_range, width_range):
        """Calculates the edge dimensions based on the provided range"""
        if isinstance(height_range[0], int):
            min_height = height_range[0]
            max_height = height_range[1]

            min_width = width_range[0]
            max_width = width_range[1]
            max_height = min(max_height, height)
            max_width = min(max_width, width)
            edge_height = random.randint(int(min_height), int(max_height))
            edge_width = random.randint(int(min_width), int(max_width))

        else:  # Assume float
            edge_height = int(height * random.uniform(height_range[0], height_range[1]))
            edge_width = int(width * random.uniform(width_range[0], width_range[1]))
        return edge_height, edge_width

    @staticmethod
    # Validation for edge weights
    def validate_edge_weights(edge_weights):
        if not isinstance(edge_weights, tuple) or not len(edge_weights) == 4:
            raise ValueError(f"edge_weights expected a tuple of four numbers. Got: {edge_weights}")
        
        for num in edge_weights:
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