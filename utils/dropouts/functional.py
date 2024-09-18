import os
import random
import cv2
from . import utils
import numpy as np
import networkx as nx

def edge_dropout(img, edge_height, edge_width, edge_name, fill_value):
    """Dropout the edge of an image and fill it with fill_value
    
    Args:
        img (np.ndarray): The image to augment
        edge_height (int): The height of the dropout region when cutting from the top or bottom edge
        edge_width (int): The width of the dropout region when cutting from the left or right edge
        fill_value (ColorType, Literal["random"]): The fill value to use for the dropout. Can be a single integer or the string "random" to fill with random noise
    """
    img = img.copy()
    
    height, width = img.shape[:2]
    if edge_name in ('top', 'bottom'):
        dropout_shape = (edge_height, width)
    elif edge_name in ('left', 'right'):
        dropout_shape = (height, edge_width)

    if edge_name == 'top':
        img[0:edge_height, :] = generate_fill(dropout_shape, fill_value, img.dtype)
    elif edge_name == 'bottom':
        img[height-edge_height:height, :] = generate_fill(dropout_shape, fill_value, img.dtype)
    elif edge_name == 'left':
        img[:, 0:edge_width] = generate_fill(dropout_shape, fill_value, img.dtype)
    elif edge_name == 'right':
        img[:, width-edge_width:width] = generate_fill(dropout_shape, fill_value, img.dtype)
    else:
        raise ValueError(f"Edge name must be top, bottom, left, or right. Got:{edge_name}")
    return img

def landmarks_dropout(img, landmarks, feature, dropout_height, dropout_width, fill_value, inverse):
    """Dropout a facial feature of an image using the image landmarks and fill it with fill_value. For example, if the feature is 'eyes' it will dropout the eyes with a rectangular region.  
    
    Args:
        img (np.ndarray): The image to augment
        landmarks (List[List[int, int], List[int, int]], List[List[int, int]]): Specifies the list of keypoints in the xy format. Contains only one keypoint in the case that the feature is 'nose' and two elements otherwise
        feature (string): Specifies the facial feature to dropout. One of 'eyes', 'nose' or 'mouth'
        dropout_height (int): The height of the dropout region
        dropout_width (int): The width of the dropout region.
        fill_value (ColorType, Literal["random"]): The fill value to use for the dropout. Can be a single integer or the string "random" to fill with random noise
        inverse (bool): Whether to dropout the facial feature or the 'inverse' of the facial feature
    """
    original_img = img.copy()
    img = img.copy()
    height, width = img.shape[:2]
    # Create one-indexed 2d graph
    G = nx.grid_2d_graph(width + 1, height + 1)
    if feature in ('eyes', 'mouth'):
        left_key, right_key = tuple(landmarks[0]), tuple(landmarks[1])
        # Generate shortest path of points between landmarks
        path = nx.bidirectional_shortest_path(G, source=left_key, target=right_key)
        img = dropout_along_path(img, dropout_height, dropout_width, path, fill_value)
    elif feature == 'nose':
        x, y = landmarks[0][0], landmarks[0][1]
        dropout_shape = (dropout_height * 2, dropout_width * 2)
        img[y - dropout_height:y + dropout_height, x - dropout_width:x + dropout_width] = generate_fill(dropout_shape, fill_value, img.dtype)
    
    if inverse:
        img = original_img - img
        
    return img

def generate_fill(dropout_shape, fill_value, dtype):
    """Generate a fill based on the dropout shape, its dtype and the specified fill_value"""
    if isinstance(fill_value, str) and fill_value == "random":
        random_fill = np.empty(shape=dropout_shape)
        for i in range(dropout_shape[0]):
            for j in range(dropout_shape[1]):
                random_fill[i, j] = random.randint(0, utils.MAX_VALUES_BY_DTYPE[dtype])
        return random_fill

    return fill_value

def dropout_along_path(img, dropout_height, dropout_width, path, fill_value):
    """Generate a fill based on the path between keypoints. Each keypoint in path is a node that will be dropped along with its neighbouring pixels specified by dropout_height and dropout_width"""
    for node in path:
        x, y = node[0], node[1]
        dropout_shape = (2 * dropout_height, 2 * dropout_width)
        img[
            y - dropout_height:y + dropout_height, 
            x - dropout_width:x + dropout_width
        ] = generate_fill(dropout_shape, fill_value, img.dtype)

    return img

def alot_dropout(img, data_path, holes):
    """Apply cutout augmentation by cutting out holes and filling them with random images from the ALOT dataset
    
    Args:
        img (np.ndarray): The image to augment
        data_path (string): The path to the ALOT dataset directory
        holes (Iterable[Tuple[int, int, int, int]]): An iterable of tuples where each tuple contains the coordinates of the top-left and bottom-right corners of the rectangular hole (x1, y1, x2, y2).
    """
    img = img.copy()
    for x1, y1, x2, y2 in holes:
        alot_img = get_random_alot_image(data_path, img.dtype, x2 - x1, y2 - y1)
        for i, row in enumerate(range(y1, y2)):
            for j, col in enumerate(range(x1, x2)):
                img[row, col] = alot_img[i, j]

    return img

def get_random_alot_image(data_path, dtype, width, height):
    """
    Return random ALOT image from data_path resized to (width, height). 
    Expected data_path looks like
    data_path
        /1
            /1_cli.png
            /1_c1l1.png
            ...
        /2
            /2_cli.png
            /2_c1l1.png
            ...
        /3
        ...
        /250
    
    """
    try:
        alot_dir = os.listdir(data_path)

        # Get random subdirectory path
        random_alot_subdir_path = f"{data_path}/{random.randint(1, len(alot_dir) - 1)}"
        random_alot_subdir = os.listdir(random_alot_subdir_path)

        # Get random image path
        random_alot_img_path = os.path.abspath(f"{random_alot_subdir_path}/{random_alot_subdir[random.randint(0, len(random_alot_subdir) - 1)]}")

        # Read ALOT image from disk
        alot_img = cv2.imread(random_alot_img_path, cv2.IMREAD_GRAYSCALE)
        # Resize ALOT image
        alot_img = cv2.resize(alot_img, dsize=(width, height))
        return alot_img.astype(dtype=dtype)
    except:
        raise ValueError(f'The directory {data_path} did not have the expected file structure')
    
def border_dropout(img, border_size, fill_value, ksize):
    """Dropout the border of an image and fill it with fill_value
    
    Args:
        img (np.ndarray): The image to augment
        border_size (int): The height/width of the dropout region when cutting from border of the image
        fill_value (ColorType, Literal["blur"]): The fill value to use for the dropout. Can be a single integer or the string "blur" to fill the border with a Gaussian blur
        ksize (int): The kernel size for the Gaussian blur
    """
    img = img.copy()
    
    height, width = img.shape[:2]
    
    # Apply Gaussian blur to border of image
    if isinstance(fill_value, str) and fill_value == "blur":
        blurred_img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        img[0:border_size, :]  = blurred_img[0:border_size, :]
        img[height-border_size:height, :] = blurred_img[height-border_size:height, :]
        img[:, 0:border_size] = blurred_img[:, 0:border_size]
        img[:, width-border_size:width] = blurred_img[:, width-border_size:width]
    else:
        img[0:border_size, :] = fill_value
        img[height-border_size:height, :] = fill_value
        img[:, 0:border_size] = fill_value
        img[:, width-border_size:width] = fill_value

    return img
