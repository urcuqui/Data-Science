import torch
from torchvision import transforms
import base64
import io
from PIL import Image
from flask import Flask, request, jsonify

# load the model
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True, verbose=False)

# put model in evaluation mode
model.eval()

# define the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# put the model on a GPU if available, otherwise CPU
model.to(device)

with open("../data/labels.txt", 'r') as f:
    labels = [label.strip() for label in f.readlines()]

# Define the transforms for preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256
    transforms.CenterCrop(224),  # Crop the image to 224x224 about the center
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Normalize the image with the ImageNet dataset mean values
        std=[0.229, 0.224, 0.225]  # Normalize the image with the ImageNet dataset standard deviation values
    )
])
    

def predict(x):
    img_tensor = x.to(device)

    with torch.no_grad():
        output = model(img_tensor)
    output = torch.nn.Softmax(dim=1)(output)
    label = labels[output[0].argmax()]
                   
    return label, output

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def run_infer():
    data = request.json
    img_bytes = base64.urlsafe_b64decode(data['image'])
    img = Image.open(io.BytesIO(img_bytes))
    
    img_tensor = preprocess(img).unsqueeze(0) #preprocess input
    label, output = predict(img_tensor) # predict
    
    output_list = output.tolist()

    response = {'message': 'Success', 'label': label, 'probs': output_list}
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(port=2718)