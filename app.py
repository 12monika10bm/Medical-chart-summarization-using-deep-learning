from flask import Flask, request, render_template
import pickle
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the model and vectorizer
model = tf.keras.models.load_model("SymptomtoSpecialist.h5")
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

labels = [
    "Psoriasis - Dermatologist",
    "Varicose Veins - Vascular Surgeon or Phlebologist",
    "Peptic Ulcer Disease - Gastroenterologist",
    "Drug Reaction - Allergist or Immunologist",
    "Gastroesophageal Reflux Disease (GERD) - Gastroenterologist",
    "Allergy - Allergist or Immunologist",
    "Urinary Tract Infection (UTI) - Urologist or Infectious Disease Specialist",
    "Malaria - Infectious Disease Specialist",
    "Jaundice - Hepatologist or Gastroenterologist",
    "Cervical Spondylosis - Orthopedic Surgeon or Neurologist",
    "Migraine - Neurologist",
    "Hypertension - Cardiologist or Nephrologist",
    "Bronchial Asthma - Pulmonologist or Allergist",
    "Acne - Dermatologist",
    "Arthritis - Rheumatologist",
    "Dimorphic Hemorrhoids - Proctologist or Colorectal Surgeon",
    "Pneumonia - Pulmonologist or Infectious Disease Specialist",
    "Common Cold - Primary Care Physician (General Practitioner/Family Doctor)",
    "Fungal Infection - Dermatologist or Infectious Disease Specialist",
    "Dengue - Infectious Disease Specialist or Hematologist",
    "Impetigo - Dermatologist or Pediatrician",
    "Chickenpox - Pediatrician or Infectious Disease Specialist",
    "Typhoid - Infectious Disease Specialist or Gastroenterologist",
    "Diabetes - Endocrinologist or Diabetologist"
]

@app.route('/')
def home():
    return render_template('index.html')
    # User is not loggedin redirect to login page

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('result.html')

@app.route('/category')
def category():
    return render_template('category.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptom_description = request.form['symptom_description']
        data = [symptom_description]
        vect = vectorizer.transform(data).toarray()
        prediction = model.predict(vect)
        predicted_label = labels[np.argmax(prediction)]
        return render_template('result.html', prediction_text=predicted_label, symptom_description=symptom_description)

if __name__ == "__main__":
    app.run(debug=True)
