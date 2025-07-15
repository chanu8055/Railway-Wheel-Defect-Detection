# 🚆 Railway Wheel Defect Detection using CNN

This project uses image-based classification powered by Convolutional Neural Networks (CNN) to detect defective railway wheels. It helps in early detection of cracks or issues to avoid accidents.

## 🧠 Features

- Detects damaged vs healthy wheels
- Trains CNN using TensorFlow/Keras
- Can be trained on real or synthetic datasets

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy / Matplotlib
- Pillow

## 🚀 How to Run

1. Install requirements:
```
pip install -r requirements.txt
```

2. Train the model:
```
python wheel_defect_cnn.py
```

3. Model is saved as `model.h5`

## 📁 Dataset Included

This folder includes a basic sample dataset under `train/` and `test/`. You can add more real wheel images for better training.

## ✅ Sample Output

- `damaged` → detected as defective
- `healthy` → detected as safe
