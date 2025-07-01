# bean-leaf-lesion-classifier
Deep learning model for classifying bean leaf lesions using GoogLeNet. Trained on real plant images to detect disease types with high accuracy.
# 🫘 Bean Leaf Lesions Classification Using GoogLeNet

This project is a deep learning-based image classification model that detects and classifies **lesions on bean plant leaves** using the **GoogLeNet (Inception v1)** architecture. It is trained on a labeled dataset of bean leaf images and aims to assist in **early detection of plant diseases**, which is critical for maintaining crop health and yield. The model achieves high accuracy and can be used to predict lesion types from new, unseen images.

---

## 📁 Dataset

- **Source**: [Kaggle - Bean Leaf Lesions Classification](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)
- The dataset contains annotated images of bean leaves with various disease categories.
- Files:
  - `train.csv`: Training image paths and labels.
  - `val.csv`: Validation image paths and labels.

---

## 🧰 Requirements

pip install opendatasets torch torchvision scikit-learn matplotlib pandas
🛠️ How It Works
Download the dataset using opendatasets.
Preprocess the images to resize and normalize.
Encode labels using LabelEncoder.
Use a pretrained GoogLeNet model and fine-tune it on the bean leaf dataset.
Train the model using cross-entropy loss and Adam optimizer.
Evaluate on validation set.
Predict custom images uploaded by the user.

📊 Training Setup
Model: GoogLeNet (Inception v1)
Image Size: 128x128
Epochs: 15
Batch Size: 4
Optimizer: Adam
Loss Function: CrossEntropyLoss

📈 Sample Training Output
Epoch 1/15, Train Loss: 1.2345, Train Accuracy: 83.75%
...
Epoch 15/15, Train Loss: 0.2341, Train Accuracy: 97.92%

🎯 Final Validation Accuracy
Displayed at the end of training:
Accuracy Score is: 95.63%

🖼️ Predict Custom Image
Upload a new image and run prediction:
predicted_label, confidence = predict_image(image_path, model, transform, label_encoder, device)
print(f"Predicted Class: {predicted_label}, Confidence: {confidence:.2f}%")

💾 Save the Model
torch.save(model.state_dict(), '/content/drive/MyDrive/googlenet_model.pth')

🔍 Example Prediction Output
Predicted Class: Angular Leaf Spot
Confidence: 98.52%

🧠 Built With
Python
PyTorch
TorchVision
Scikit-learn
Matplotlib
Google Colab

📌 Author
Developed by [Abdo Noaman ]

