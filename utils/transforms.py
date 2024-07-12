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
    # Baseline transforms
    A.RandomResizedCrop(size=(48, 48), scale=(0.8, 1.0), interpolation=cv2.INTER_LINEAR),
    A.Affine(translate_percent=(-0.2, 0.2), interpolation=cv2.INTER_NEAREST, mode=cv2.BORDER_REPLICATE),
    A.HorizontalFlip(),
    A.Rotate(limit=10, border_mode=cv2.BORDER_REPLICATE, p=0.5),
    
    # Occlusion+ transforms
    A.OneOf([
        # Simulate limited field of view and self-occlusion
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0),
        
        # Simulate artificial occlusion
        A.GaussNoise(),
    
        # Simulate extreme illumination
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True, p=0.5),
    ], p=0.5),

    A.Normalize(mean=(0.485, ), std=(0.229, )),
    ToTensorV2(),
])

# MaskedAutoEncoder transform
mae_tf = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), interpolation=cv2.INTER_LINEAR),
    A.HorizontalFlip(),
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