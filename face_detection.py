import cv2
import os


def eye_extract(input_img, output_path):
    """
    Extract the eye area from the input image using Haar Cascade.

    Args:
        input_img (str): Path to the input image file.
        output_path (str): Directory to save the extracted eye area.

    Returns:
        numpy.ndarray: The extracted eye area as an OpenCV image.
    """

    eye_cascade = cv2.CascadeClassifier('eye_shape/haarcascade_eye.xml')
    if eye_cascade.empty():
        raise RuntimeError("Haar Cascade for eyes could not be loaded. Check the path.")

    img = cv2.imread(input_img)
    if img is None:
        raise FileNotFoundError(f"Image file not found: {input_img}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(eyes) == 0:
        raise ValueError("No eyes detected in the image.")

    if output_path is not None:
        if not os.path.isdir(output_path):
            raise ValueError(f"Output path must be a directory. Got: {output_path}")

    for (x, y, w, h) in eyes:
        eye_part = img[y:y + h, x:x + w]

        if output_path is not None:
            original_name = os.path.splitext(os.path.basename(input_img))[0]
            save_path = f"{original_name}_eye-extracted.jpg"
            full_path = os.path.join(output_path, save_path)

            cv2.imwrite(full_path, eye_part)
        return eye_part

    return None
