import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from pathlib import Path

from EyeNet import EyeNet
from FacialSkinColorNet import FacialSkinColorNet

from extract_eye_from_face import extract_eye_area


title = "Baymax! Medical Image Classification 🏥"
description = """
    Personal robot to detect visual facial diseases like Skin-related, Eye-related, Facial Skin color, and Fatigue or Stress.
    Upload an image or take a photo using your camera!
"""

classes_for_skin_color = ['Facial Redness', 'Normal', 'Pale/Grayish Skin Tone', 'Yellowish Skin Tone']
classes_for_eye = ['Jaundice', 'Conjunctivitis', 'Dark Circles Under Eyes']


def load_model(model_path, device):
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        if 'skin' in model_path.name.lower():
            model = FacialSkinColorNet().to(device)
        elif 'eye' in model_path.name.lower():
            model = EyeNet().to(device)
        else:
            raise ValueError("Unknown model type in file name.")

        checkpoint = torch.load(str(model_path), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def test_single_image(model, image, device, classes):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        class_probs = {class_name: float(prob) for class_name, prob in zip(classes, probabilities[0])}
        return classes[predicted_class], class_probs
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None


SKIN_COLOR_MODEL_PATH = 'models/skin_color_checkpoint.pth'
EYE_MODEL_PATH = 'models/eye_model_checkpoint.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skin_model = load_model(SKIN_COLOR_MODEL_PATH, device)
eye_model = load_model(EYE_MODEL_PATH, device)


def analyze_image(input_img):
    if input_img is None:
        return "No image uploaded."

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes = axes.ravel()


    skin_predicted_class, skin_probabilities = test_single_image(skin_model, input_img, device, classes_for_skin_color)
    if skin_predicted_class and skin_probabilities:
        labels = list(skin_probabilities.keys())
        probs = list(skin_probabilities.values())
        axes[0].bar(labels, [p * 100 for p in probs], color=['red', 'green', 'gray', 'yellow'])
        axes[0].set_title(f"Skin Prediction: {skin_predicted_class}")
        axes[0].set_ylabel("Confidence (%)")
        axes[0].set_ylim(0, 100)
        axes[0].set_xticklabels(labels, rotation=45, ha='right')


    extracted_eye_part = extract_eye_area(input_img, 'eye_output')
    
    eye_predicted_class, eye_probabilities = test_single_image(eye_model, extracted_eye_part, device, classes_for_eye)
    if eye_predicted_class and eye_probabilities:
        labels = list(eye_probabilities.keys())
        probs = list(eye_probabilities.values())
        axes[1].bar(labels, [p * 100 for p in probs], color=['blue', 'orange', 'purple'])
        axes[1].set_title(f"Eye Prediction: {eye_predicted_class}")
        axes[1].set_ylabel("Confidence (%)")
        axes[1].set_ylim(0, 100)
        axes[1].set_xticklabels(labels, rotation=45, ha='right')

    plt.tight_layout()
    return fig


demo = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(sources=['upload', 'webcam', 'clipboard'], type="pil", label="Upload or Take Photo"),
    outputs=gr.Plot(label="Prediction Results"),
    title=title,
    description=description,
    theme="default"
)

if __name__ == "__main__":
    demo.launch()