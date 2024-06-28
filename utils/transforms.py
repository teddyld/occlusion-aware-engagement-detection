import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
from utils.config import ALOT_DATA_PATH

# Simulate facial accessories and external occlusions by applying random dropout to face landmarks i.e. across the eyes, nose, and mouth
class RandomLandmarksDropout(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        
    def apply(self, img, **params):
        # TODO
        return img

# Simulate self-occlusion by cropping a random edge of the image
class RandomCropEdges(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        
    def apply(self, img, **params):
        # TODO
        return img

# Simulate artificial occlusion by randomly erasing the image with a random ALOT texture
class RandomALOTDropout(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.data = ALOT_DATA_PATH
        
    def apply(self, img, **params):
        # TODO
        return img

# Test and valid dataset transform
simple_tf = A.Compose([
    A.Normalize(mean=(0.485, ), std=(0.229, )),
    ToTensorV2(),
])

# Non-occlusion-aware training dataset transform
baseline_tf = A.Compose([
    A.RandomResizedCrop(size=(48, 48), scale=(0.8, 1.0), interpolation=cv2.INTER_LINEAR),
    A.Affine(translate_percent=(-0.2, 0.2), interpolation=cv2.INTER_NEAREST, mode=cv2.BORDER_REPLICATE),
    A.HorizontalFlip(),
    A.Rotate(limit=10, border_mode=cv2.BORDER_REPLICATE),
    A.Normalize(mean=(0.485, ), std=(0.229, )),
    ToTensorV2(),
])

# Occlusion-aware training dataset transform
occlusion_aware_tf = A.Compose([
    # Simulate facial accessories and external occlusions
    
    # Simulate limited field of view and self-occlusion
    A.Rotate(limit=10, border_mode=cv2.BORDER_REPLICATE, p=0.5),

    # Simulate artificial occlusion
    A.OneOf([
        A.GaussNoise(),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0),
        A.CoarseDropout(num_holes_range=(10, 20), hole_height_range=(3, 3), hole_width_range=(3, 3)),
    ], p=0.5),
    # Simulate extreme illumination
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True, p=0.5),

    A.Resize(height=48, width=48, interpolation=cv2.INTER_LINEAR),
    A.Normalize(mean=(0.485, ), std=(0.229, )),
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy'))

test_tf = A.Compose([
    RandomCropEdges(),
    A.Normalize(mean=(0.485, ), std=(0.229, )),
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy'))
