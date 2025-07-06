# Text-Classification-API

**“Flask API for text sentiment/topic classification using PyTorch.”**

---

##  **Text Classification API**

A production-ready **Flask REST API** for predicting the **sentiment** or **topic** of any input text, built with **PyTorch** and a custom NLP model.

---

##  **Project Highlights**

- **Custom NLP Model:** Trained in PyTorch on your text dataset.
- **Preprocessing:** Uses a saved vocabulary (`vocab.pkl`) and label encoder (`label_encoder.pkl`).
- **REST API:** Predict text category with a simple POST request.
- **Postman:** User-friendly tool to test the API with real JSON requests — import our collection and send test inputs easily.
- **Ready for Deployment:** Can be hosted on any server (Heroku, Railway, Render, AWS).

---

## **Project Structure**

├── text_classification.pt # Trained PyTorch model
├── class_Api.py # Flask API script
├── vocab.pkl # Saved tokenizer/vocab
├── label_encoder.pkl # Saved label encoder
├── requirements.txt # Dependencies
├── README.md # Project documentation (this file!)
└── venv/ # Virtual environment (should be in .gitignore)



---

## 🗂️ **Classes**

Example output classes:
- Positive
- Negative
- Neutral  
*(Or your custom categories if you trained on specific topics)*

---

## ⚙️ **How It Works**

### 📌 **1️⃣ Train Model**
- The PyTorch model was trained on your dataset using tokenization & embeddings.
- The trained model is saved as `text_classification.pt`.
- Vocabulary and label mappings are stored as `vocab.pkl` and `label_encoder.pkl`.

### 📌 **2️⃣ API Endpoint**
- **`/predict`** — Accepts raw text in JSON.
- The API tokenizes the text, runs the model, and returns the predicted class.

 ### *3️⃣ Test with Postman

Open Postman on your computer.
Create a POST request to http://127.0.0.1:5000/predict.
In the Body tab, select form-data (for files) or raw JSON (for text).
Add your text input (or upload your image if it’s an image API).
Click Send — see the prediction in the JSON response instantly.

## ✅ **Example Request**

**Request:**  
`POST /predict`

**Body:**
```json
{
  "text": "I love this product, it works really well!"
}


Response:

{
  "predicted_class": "positive"
}


// "How To Run Locally":

 Clone this repo
 git clone https://github.com/YOUR_USERNAME/text-classification-api.git
cd text-classification-api

// "Create virtual environment"

python -m venv venv
source venv/bin/activate    # Linux/macOS
# OR
venv\Scripts\activate       # Windows
 

//   Install dependencies
  pip install -r requirements.txt

//  Run the API

python class_Api.py


// Open Postman

Make a POST request to: http://127.0.0.1:5000/predict


//  Deployment

You can easily deploy this API on:

Heroku

Render

Railway

AWS / GCP

// How To Use
Send single or batch text for prediction.

Get JSON responses instantly.

Integrate with your website, chatbot, or app.


//What’s Included
✔ Well-documented Flask API
✔ Trained PyTorch model & encoders
✔ Swagger docs for easy testing
✔ Example usage for real applications

// Accuracy Tips
Accuracy can be improved by:

Adding Early Stopping to prevent overfitting.

Using Data Augmentation for text (synonym replacement, back translation).

Fine-tuning embeddings.

Training longer on more data.

Built by: Hassaan Ahmed
 Email: hassaanahmed80400@gmail.com
 GitHub: hassaan-ahmed825