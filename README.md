# skin-disease-classification
AI project for Facial Skin disease classification

### 🤖 Baymax! Medical Image classification
Personal robot to detect visual facial diseases like Skin-related, Eye-related, Facial Skin color, and Fatigue or Stress. Upload an image or take a photo using your camera!

![Baymax](https://lumiere-a.akamaihd.net/v1/images/pp_baymax_herobanner_22586_755e6499.jpeg?region=0,0,2048,878)

## Installation
1. Cloning the repository:
```bash
git clone https://github.com/Navfalbek/skin-disease-classification.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained models  `.pth` for Facial and Eye disease classification:
   
   You can download them from [Google Drive](https://drive.google.com/drive/folders/16U4Y35bLDBHgkDyiHoBk1CqXXtu7lORb?usp=sharing) ~ 1.1 GB.
   After downloading the models locate the in `models` folder inside the repo.

## Usage
Simply run and then open in localhost `127.0.0.1:7860`:
```bash
python app.py
```

## NN Structure
```scss
Input (3 x 224 x 224)
 |-> Conv2d(3 -> 32, 3x3, pad=1)
 |-> BatchNorm2d(32)
 |-> ReLU
 |-> MaxPool2d(2)         --> (32 x 112 x 112)
 |-> Conv2d(32 -> 64, 3x3, pad=1)
 |-> BatchNorm2d(64)
 |-> ReLU
 |-> MaxPool2d(2)         --> (64 x 56 x 56)
 |-> Conv2d(64 -> 128, 3x3, pad=1)
 |-> BatchNorm2d(128)
 |-> ReLU
 |-> MaxPool2d(2)         --> (128 x 28 x 28)
 |-> Dropout(0.3)
 |-> Flatten              --> (128*28*28) = 100,352
 |-> Linear(100,352 -> 512)
 |-> BatchNorm1d(512)
 |-> ReLU
 |-> Dropout(0.4)
 |-> Linear(512 -> 4)
Output: (N, 4)
```


## License
This project is licensed under the [MIT License](LICENSE).

## Presentation link
[Canva Link](https://www.canva.com/design/DAGaB6qc2lk/bUo8p6mag0H1ZfF1zRkHmA/edit?utm_content=DAGaB6qc2lk&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
