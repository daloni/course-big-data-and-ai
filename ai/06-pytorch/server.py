import torch
from torch import nn
from flask import Flask, request, jsonify

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(24, 300),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(300, 300),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(300, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        return predicted_classes, probabilities

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
model.to(device)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        columns = [
            'age', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca',
            'sex_male', 'sex_female', 'cp_type_0', 'cp_type_1', 'cp_type_2',
            'cp_type_3', 'restecg_type_0', 'restecg_type_1', 'restecg_type_2',
            'slope_type_0', 'slope_type_1', 'slope_type_2', 'thal_type_0', 'thal_type_1',
            'thal_type_2', 'thal_type_3'
        ]

        form_data = request.form

        input_data = {col: 0 for col in columns}

        input_data['age'] = float(form_data.get('age', 0))
        input_data['trestbps'] = float(form_data.get('trestbps', 0))
        input_data['chol'] = float(form_data.get('chol', 0))
        input_data['fbs'] = float(form_data.get('fbs', 0))
        input_data['thalach'] = float(form_data.get('thalach', 0))
        input_data['exang'] = float(form_data.get('exang', 0))
        input_data['oldpeak'] = float(form_data.get('oldpeak', 0))
        input_data['ca'] = float(form_data.get('ca', 0))

        sex = form_data.get('sex', 'male')
        input_data[f'sex_{sex}'] = 1

        cp_type = form_data.get('cp_type', '0')
        input_data[f'cp_type_{cp_type}'] = 1

        restecg_type = form_data.get('restecg_type', '0')
        input_data[f'restecg_type_{restecg_type}'] = 1

        slope_type = form_data.get('slope_type', '0')
        input_data[f'slope_type_{slope_type}'] = 1

        thal_type = form_data.get('thal_type', '0')
        input_data[f'thal_type_{thal_type}'] = 1

        input_features = [input_data[col] for col in columns]
        input_tensor = torch.tensor([input_features], dtype=torch.float32)
        input_features = input_tensor.to(device)

        predicted_classes, probabilities = model.predict(input_features)
        result = 'Probable' if probabilities.argmax(1).item() else 'Improbable'

        return jsonify({
            'prediction': result,
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
