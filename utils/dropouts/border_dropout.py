from albumentations.core.transforms_interface import ImageOnlyTransform
from . import functional
import random

class BorderDropout(ImageOnlyTransform):
    """BorderDropout drops out the bordering edges of the image and fills the dropout region with a constant value or applies a Gaussian blur.
    
    Args:
        border_range (Tuple[int, int]): Defines the minimum and maximum pixels of the border dropout, providing variability in dropout size.
        fill_value (ColorType, Literal["blur"]): Specifies the value used to fill the cropped regions. This can be a constant value, or 'random' which fills the region with random noise.
        blur_limit (Tuple[int, int] | int): Controls the range for the Gaussian kernel size if fill_value is "blur", providing variability in blur intensity.
            - If a single int is provided, the kernel size will be randomly chosen between 0 and that value.
            - If a tuple of two ints is provided, it defines the inclusive range
              of possible kernel sizes.
            Default: (3, 7).
        p (float): Probability of applying the transform. Default: 0.5.
    """
    def __init__(self, border_range=(4, 8), fill_value="blur", blur_limit=(3, 7), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.border_range = border_range
        self.fill_value = fill_value
        self.blur_limit = blur_limit
        
        self.validate_range(border_range, "border_range")
        self.validate_limits(blur_limit)

    def apply(self, img, **params):
        height, width = img.shape[:2]
        border_size = self.calculate_border_size(
            height,
            width,
            self.border_range,
        )
        
        ksize = self.calculate_ksize(self.blur_limit)

        return functional.border_dropout(img, border_size, self.fill_value, ksize)

    @staticmethod
    def calculate_border_size(height, width, size_range):
        """Calculates the border dimensions based on the provided range"""
        if isinstance(size_range[0], int):
            min_size = size_range[0]
            max_size = min(size_range[1], height, width)

            border_size = random.randint(int(min_size), int(max_size))

        else:  # Assume float
            border_size = int(height * random.uniform(size_range[0], size_range[1]))

        return border_size
    
    @staticmethod
    def calculate_ksize(blur_limit):
        """Calculates the kernel size for the Gaussian blur based on the provided blur limit"""
        if isinstance(blur_limit, int):
            ksize = random.randint(0, blur_limit)
        else:
            ksize = random.randint(blur_limit[0], blur_limit[1])
        
        # Deal with even kernel size
        if ksize != 0 and ksize % 2 != 1:
            ksize = ksize + 1
        
        return ksize

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
    # Validation for blur limit
    def validate_limits(blur_limit):
        if isinstance(blur_limit, tuple):
            for v in blur_limit:
                if v != 0 and v % 2 != 1:
                    raise ValueError(f"Blur limit must be 0 or odd. Got: {blur_limit}")
        elif not isinstance(blur_limit, int):
            raise ValueError(f"Blur limit must be an integer or a tuple of two integers. Got: {blur_limit}")
    