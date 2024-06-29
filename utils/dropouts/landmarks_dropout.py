from albumentations.core.transforms_interface import ImageOnlyTransform

class LandmarksDropout(ImageOnlyTransform):
    """LandmarksDropout randomly drops out rectangular regions from the image 
    eyes, nose, or mouth landmarks to simulate occlusion from facial 
    accessories or external objects 
    
    Args:
        p: Probability of applying the transform. Default: 0.5.
    
    """
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        
    def apply(self, img, **params):
        return img