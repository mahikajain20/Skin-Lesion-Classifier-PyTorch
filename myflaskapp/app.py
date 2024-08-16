from flask import Flask, request, jsonify
from flask import render_template
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from PIL import Image
from PIL import UnidentifiedImageError
import io

#Model 2 - Pretrained, DenseNet121
class CustomDenseNet121(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.1):
        super(CustomDenseNet121, self).__init__()
        
        # Load pre-trained DenseNet121
        self.model = models.densenet121(pretrained=True)
        
        # Replace the classifier with a custom one
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.model.classifier.in_features, 256, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, num_classes, bias=False),
            nn.BatchNorm1d(num_classes)
        )
        
        print(f'DenseNet121 created with {num_classes} classes and dropout rate of {dropout_rate}')
        
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([p.numel() for p in model_parameters])
        print(f'Model has {params} trainable params.')

    def forward(self, x):
        return self.model(x)


#Importing the model

path = '/Users/mahikajain/Desktop/lhl/skin-lesion-classifier/models/bestmodels/best_model_20240814-042148.pth'
model_x = CustomDenseNet121(dropout_rate=0.3)
# Load your trained model
checkpoint = torch.load(path, map_location=torch.device('cpu'))
model_x.load_state_dict(checkpoint['model_state_dict'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_x.to(device)
model_x.eval()

#Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

#Inference function

def infer(model, image: Image, device='cpu'):
    """
    Perform inference on a single image using the trained model.
    
    Args:
        model: The trained PyTorch model.
        image_file (FileStorage): Uploaded image file.
        device (str): Device to run the inference on ('cuda' or 'cpu').
        
    Returns:
        pred_label (int): Predicted label for the input image.
        pred_probs (torch.Tensor): Class probabilities for the input image.
    """
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the required input size
        transforms.ToTensor(),        # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    # Apply the transformations
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():  # Disable gradient calculation
        # Perform the forward pass
        outputs = model(image)
        
        # Get the predicted class probabilities
        pred_probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the predicted label
        _, pred_label = torch.max(outputs, 1)
    
    return pred_label.item(), pred_probs.squeeze().tolist()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        try:
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except UnidentifiedImageError:
            return jsonify({'error': 'Uploaded file is not a valid image'}), 400
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
        
        try:
            pred_label, pred_probs = infer(model_x, image, device)
            class_names = ['benign', 'malignant']  # Replace with your actual class names
            prediction = class_names[pred_label]
            return jsonify({
                'prediction': prediction,
                'probabilities': dict(zip(class_names, pred_probs))
            }), 200
        except Exception as e:
            return jsonify({'error': f'Error during model inference: {str(e)}'}), 500
        
    return render_template('upload_pretty.html')


if __name__ == '__main__':
    app.run(debug=True)








