import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from utils.dropouts.alot_dropout import ALOTDropout
from utils.dropouts.landmarks_dropout import LandmarksDropout
from utils.dropouts.edge_dropout import EdgeDropout

# Simple transform
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
    # Simulate limited field of view and self-occlusion
    A.Rotate(limit=10, border_mode=cv2.BORDER_REPLICATE, p=0.5),
    EdgeDropout(edge_height_range=(8, 24), edge_width_range=(8, 24), fill_value="random"),
    # Simulate artificial occlusion
    A.OneOf([
        ALOTDropout(num_holes_range=(1, 1), hole_height_range=(16, 24), hole_width_range=(16, 24)),
        A.GaussNoise(),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0),
        A.CoarseDropout(num_holes_range=(10, 20), hole_height_range=(3, 3), hole_width_range=(3, 3)),
    ], p=0.5),
    # Simulate extreme illumination
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True, p=0.5),

    A.Resize(height=48, width=48, interpolation=cv2.INTER_LINEAR),
    A.Normalize(mean=(0.485, ), std=(0.229, )),
    ToTensorV2(),
])

def get_transform(tf_name):
    """Return the transform parsed by tf_name
        
        Args:
            tf_name (Literal["simple", "baseline", "occlusion_aware"]): Specifies the transform to return
    """
    if tf_name == 'simple':
        return simple_tf
    elif tf_name == 'baseline':
        return baseline_tf
    elif tf_name == 'occlusion_aware':
        return occlusion_aware_tf
    else:
        return None