from flask import Flask, request, jsonify
import torch
from torch import nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import io
import os

app = Flask(__name__)

# تحديد الجهاز (GPU أو CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Available device is: {device}")

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # تغيير حجم الصورة إلى 128x128
    transforms.ToTensor(),  # تحويل الصورة إلى تنسور PyTorch
    transforms.ConvertImageDtype(torch.float),
])

# تحميل LabelEncoder
classes = ['angular_leaf_spot', 'bean_rust', 'healthy']
label_encoder = LabelEncoder()
label_encoder.fit(classes)

# تحميل النموذج
def load_model(model_path, num_classes):
    model = models.googlenet(weights=None, aux_logits=False, init_weights=True)  # تعطيل aux_logits
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # تعديل الطبقة الأخيرة
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # تجاهل المفاتيح غير المتطابقة
    model.to(device)
    model.eval()  # وضع النموذج في وضع التقييم
    return model


model_path ='D:/deep-learning project/image classification/image classification/googlenet_model.pth'
num_classes = len(classes)
googlenet_model = load_model(model_path, num_classes)

def predict_image(image, model, transform, label_encoder, device):
    image = transform(image).to(device) / 255.0 
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        confidence = probabilities[0][predicted_class].item() * 100
    return predicted_label, confidence

# End Point
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No choosen file'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        predicted_label, confidence = predict_image(image, googlenet_model, transform, label_encoder, device)
        return jsonify({
            'predicted_class': predicted_label,
            'confidence': round(confidence, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)