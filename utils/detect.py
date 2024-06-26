import os
from dotenv import load_dotenv
load_dotenv()
import requests
import matplotlib.pyplot as plt

COMPREFACE_APIKEY = os.getenv('COMPREFACE_API_KEY')

header = {
    "x-api-key": COMPREFACE_APIKEY
}

def detect_face_features(image_path, verbose=False):
    '''
    Returns bbox and landmarks of a face in image_path using CompreFace service. Returns faces with a minimum confidence threshold of 0.6.
    Arguments:
        image_path (string): Path to image
        verbose (boolean): When asserted will print a message if image_path did not contain a detected face
    '''
    try:
        file = open(image_path, 'rb')
        files = {
            "file": file
        }
        
        # Send request to CompreFace detection API
        response = requests.post('http://localhost:8000/api/v1/detection/detect?face_plugins=landmarks&limit=1&det_prob_threshold=0.6', headers=header, files=files)

        response = response.json()
        
        # No face was found in the given image
        if 'code' in response:
            if verbose:
                print(f"{response['message']}: {image_path}")
            return None
        
        result = response['result'][0]
        
        x_min, y_min, x_max, y_max = result['box']['x_min'], result['box']['y_min'], result['box']['x_max'], result['box']['y_max']
        bbox = [x_min, y_min, x_max, y_max]
        
        landmarks = result['landmarks']
        
        return bbox, landmarks
    # Critical error - no image at image_path exists or CompreFace was not available
    except Exception as e: 
        print(e)
        return None

def bbox_to_rect(bbox, color='r'):
    """
    Convert bounding box to matplotlib format i.e.
    (upper-left x, upper-left y, lower-right x, lower-right y)
    to
    ((upper-left x, upper-left y), width, height)
    """
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1], fill=False, edgecolor=color, linewidth=1)
