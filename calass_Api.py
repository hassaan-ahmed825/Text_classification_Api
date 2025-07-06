from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import re
import pickle
import joblib

# --------------------------
# Device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Load vocab
# --------------------------
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# --------------------------
# Load label encoder
# --------------------------
le = joblib.load('label_encoder.pkl')

# --------------------------
# Model config
# --------------------------
vocab_size = len(vocab)
embed_dim = 32  # same as used in training
num_class = len(le.classes_)

# --------------------------
# Define model
# --------------------------
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# --------------------------
# Load model weights
# --------------------------
model = TextClassificationModel(vocab_size, embed_dim, num_class).to(device)
model.load_state_dict(torch.load('text_classification.pt', map_location=device))
model.eval()

# --------------------------
# Flask app
# --------------------------
app = Flask(__name__)

# --------------------------
# Simple tokenizer
# --------------------------
def simple_tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())

# --------------------------
# Text pipeline (safe for dict or torchtext.vocab.Vocab)
# --------------------------
UNK_IDX = vocab['<unk>'] if '<unk>' in vocab else 0

def text_pipeline(x):
    return [vocab[token] if token in vocab else UNK_IDX for token in simple_tokenizer(x)]

# --------------------------
# Predict route
# --------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_text = data['text']

    text_tensor = torch.tensor(text_pipeline(input_text), dtype=torch.int64)
    offsets = torch.tensor([0])

    text_tensor = text_tensor.to(device)
    offsets = offsets.to(device)

    with torch.no_grad():
        output = model(text_tensor, offsets)
        pred = output.argmax(1).item()
        class_name = le.inverse_transform([pred])[0]

    return jsonify({'predicted_sentiment': class_name})

# --------------------------
# Run app
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)
