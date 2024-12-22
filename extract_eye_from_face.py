import dlib
import cv2
import numpy as np
from PIL import Image
import os


def extract_eye_area(image, output_path=None):  
    """
    Extract the eye area from an image using dlib's facial landmarks.

    Args:
        image (str, np.ndarray, or PIL.Image.Image): Input image. It can be a file path,
                                                     an OpenCV image, or a PIL image.
        output_path (str, optional): Path to save the extracted eye area. Defaults to None.

    Returns:
        np.ndarray: The extracted eye area as an OpenCV image.
    """

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('eye_shape/shape_predictor_68_face_landmarks.dat')

    if isinstance(image, str): 
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Image file not found: {image}")
    elif isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise ValueError("Unsupported image format. Provide a file path, PIL Image, or OpenCV image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        raise ValueError("No faces detected in the image.")

    face = faces[0]
    landmarks = predictor(gray, face)

    points = []
    for n in list(range(17, 27)) + list(range(36, 48)): 
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points.append((x, y))
        
    points = np.array(points)

    min_x = np.clip(np.min(points[:, 0]) - 20, 0, img.shape[1])
    max_x = np.clip(np.max(points[:, 0]) + 20, 0, img.shape[1])
    min_y = np.clip(np.min(points[:, 1]), 0, img.shape[0])
    max_y = np.clip(np.max(points[:, 1]) + 45, 0, img.shape[0])

    eye_area = img[min_y:max_y, min_x:max_x]

    if output_path is not None:
        if not os.path.isdir(output_path):
            raise ValueError(f"Output path must be a directory. Got: {output_path}")

        original_name = os.path.splitext(os.path.basename(image))[0]

        new_file_name = f"{original_name}_eye-extracted.jpg"

        full_output_path = os.path.join(output_path, new_file_name)

        cv2.imwrite(full_output_path, eye_area)


    return eye_area