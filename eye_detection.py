import cv2
import os
import numpy as np
from PIL import Image


def eye_detect(input_img, output_path, face_cascade_path='eye_shape/haarcascade_frontalface_default.xml', eye_cascade_path='eye_shape/haarcascade_eye.xml'):
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError("Haar Cascades could not be loaded. Check the file paths.")

    if isinstance(input_img, str):
        img = cv2.imread(input_img)
        if img is None:
            raise FileNotFoundError(f"Image file not found: {input_img}")
    elif isinstance(input_img, Image.Image):
        img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
    elif isinstance(input_img, np.ndarray):
        img = input_img
    else:
        raise ValueError("Unsupported input type. Must be a file path or NumPy array.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        raise ValueError("No faces detected in the image.")

    for (x, y, w, h) in faces:
        eye_region_y_start = int(0.2 * h)
        eye_region_y_end = int(0.5 * h)
        eye_region = gray[y + eye_region_y_start:y + eye_region_y_end, x:x + w]

        eyes = eye_cascade.detectMultiScale(eye_region, scaleFactor=1.1, minNeighbors=10)

        if len(eyes) > 0:
            min_x = min([ex for (ex, ey, ew, eh) in eyes])
            max_x = max([ex + ew for (ex, ey, ew, eh) in eyes])
            min_y = min([ey for (ex, ey, ew, eh) in eyes])
            max_y = max([ey + eh for (ex, ey, ew, eh) in eyes])

            padding = 10
            eye_x1 = max(x + min_x - padding, 0)
            eye_y1 = max(y + eye_region_y_start + min_y - padding, 0)
            eye_x2 = min(x + max_x + padding, img.shape[1])
            eye_y2 = min(y + eye_region_y_start + max_y + padding, img.shape[0])

            eye_roi = img[eye_y1:eye_y2, eye_x1:eye_x2]

            if output_path is not None:
                if not os.path.isdir(output_path):
                    raise ValueError(f"Output path must be a directory. Got: {output_path}")

                # original_name = os.path.splitext(os.path.basename(input_img if isinstance(input_img, str) else "image"))[0]
                save_path = f"image_eye-region.png"
                full_path = os.path.join(output_path, save_path)

                cv2.imwrite(full_path, eye_roi)
            return eye_roi

    return None
