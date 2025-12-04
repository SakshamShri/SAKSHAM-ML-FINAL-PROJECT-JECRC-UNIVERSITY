# Disease Prediction System

A machine learning-based web application that predicts diseases based on user-reported symptoms and provides comprehensive health recommendations including precautions, medications, diet plans, and workout suggestions.

## Team Members
- **Saksham Shrivastava** (22BCON333)
- **Gungun Choudhary** (22BCON322)
- **Saurabh Soni** (22BCON273)
- **Nikunj Varshaney** (22BCOL1413)

**Institution:** JECRC University

## Overview

This project uses a Support Vector Machine (SVM) classifier trained on a dataset of 4,920 medical records with 132 symptoms to predict among 41 different diseases. The system achieves 100% accuracy on the test dataset and provides a user-friendly Flask web interface for symptom input and disease prediction.

## Features

- **Disease Prediction**: Predicts diseases based on user-entered symptoms using an SVC (Support Vector Classifier) model
- **Comprehensive Health Information**: Provides detailed information for each predicted disease including:
  - Disease description
  - Recommended precautions
  - Suggested medications
  - Diet recommendations
  - Workout plans
- **Interactive Web Interface**: Clean, responsive UI for easy symptom input and result visualization
- **REST API**: JSON endpoints for integration with other applications
- **Multiple Disease Coverage**: Supports prediction of 41 different diseases

## Supported Diseases

The system can predict the following diseases:

- Fungal infection
- Allergy
- GERD
- Chronic cholestasis
- Drug Reaction
- Peptic ulcer disease
- AIDS
- Diabetes
- Gastroenteritis
- Bronchial Asthma
- Hypertension
- Migraine
- Cervical spondylosis
- Paralysis (brain hemorrhage)
- Jaundice
- Malaria
- Chicken pox
- Dengue
- Typhoid
- Hepatitis (A, B, C, D, E)
- Alcoholic hepatitis
- Tuberculosis
- Common Cold
- Pneumonia
- Dimorphic hemorrhoids (piles)
- Heart attack
- Varicose veins
- Hypothyroidism
- Hyperthyroidism
- Hypoglycemia
- Osteoarthritis
- Arthritis
- Vertigo (Paroxysmal Positional Vertigo)
- Acne
- Urinary tract infection
- Psoriasis
- Impetigo

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn (SVC with linear kernel)
- **Data Processing**: pandas, NumPy
- **Model Persistence**: pickle
- **Frontend**: HTML, CSS, JavaScript

## Project Structure

```
neural network project/
├── main.py                    # Flask application with routes and prediction logic
├── svc.pkl                    # Trained SVC model (pickled)
├── Untitled.ipynb            # Jupyter notebook with model training and evaluation
├── datasets/
│   ├── Training.csv          # Training dataset (4,920 records, 133 columns)
│   ├── symtoms_df.csv        # Symptom information
│   ├── precautions_df.csv    # Precautions for each disease
│   ├── workout_df.csv        # Workout recommendations
│   ├── description.csv       # Disease descriptions
│   ├── medications.csv       # Medication recommendations
│   ├── diets.csv            # Diet recommendations
│   └── Symptom-severity.csv  # Symptom severity ratings
├── templates/
│   ├── index.html           # Main prediction interface
│   ├── about.html           # About page
│   ├── contact.html         # Contact page
│   ├── developer.html       # Developer information
│   └── blog.html            # Blog page
└── static/
    └── css/                 # CSS stylesheets
```

## Installation

1. **Clone or download the project**

2. **Install dependencies**:
```bash
pip install flask numpy pandas scikit-learn==1.4.1.post1
```

3. **Verify dataset files**: Ensure all CSV files are present in the `datasets/` directory

4. **Verify model file**: Ensure `svc.pkl` exists in the root directory

## Usage

### Running the Web Application

1. **Start the Flask server**:
```bash
python main.py
```

2. **Access the application**: Open your browser and navigate to `http://localhost:5000`

3. **Enter symptoms**: Type your symptoms (comma-separated) in the input field

4. **View results**: The system will display the predicted disease along with recommendations

### API Endpoints

#### Get All Symptoms
```http
GET /symptoms
```
Returns a JSON array of all available symptoms (sorted alphabetically).

#### Predict Disease (JSON)
```http
POST /predict
Content-Type: application/json

{
  "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"]
}
```

Response:
```json
{
  "disease": "Fungal infection",
  "description": "Fungal infection is a common skin condition caused by fungi.",
  "precautions": ["bath twice", "use detol or neem in bathing water", "keep infected area dry", "use clean cloths"],
  "medications": ["Antifungal Cream", "Fluconazole", "Terbinafine", "Clotrimazole", "Ketoconazole"],
  "diets": ["Antifungal Diet", "Probiotics", "Garlic", "Coconut oil", "Turmeric"],
  "workouts": ["Avoid sugary foods", "Consume probiotics", "Increase intake of garlic", ...]
}
```

#### Predict Disease (Form)
```http
POST /predict
Content-Type: application/x-www-form-urlencoded

symptoms=itching,skin_rash,nodal_skin_eruptions
```
Returns rendered HTML with prediction results.

## Model Details

### Training Process

The model was trained using the following approach (see `Untitled.ipynb`):

1. **Dataset**: 4,920 medical records with 132 symptom features and 1 target (disease)
2. **Train-Test Split**: 70-30 split with random_state=20
3. **Label Encoding**: Disease names encoded to numerical values
4. **Models Evaluated**:
   - Support Vector Classifier (SVC) - **Selected**
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - K-Nearest Neighbors
   - Multinomial Naive Bayes

5. **Model Selection**: SVC with linear kernel was selected (all models achieved 100% accuracy)

### Model Performance

- **Accuracy**: 100% on test set
- **Confusion Matrix**: Perfect diagonal (no misclassifications)
- **Feature Vector**: Binary encoding (132 dimensions)

### Symptoms Dictionary

The model uses 132 possible symptoms, including:
- `itching`, `skin_rash`, `nodal_skin_eruptions`
- `continuous_sneezing`, `shivering`, `chills`
- `joint_pain`, `stomach_pain`, `acidity`
- `high_fever`, `headache`, `nausea`
- `fatigue`, `weight_loss`, `cough`
- And 117 more symptoms...

## Example Usage

### Example 1: Fungal Infection
**Input Symptoms**: `itching, skin_rash, nodal_skin_eruptions`

**Output**:
- **Disease**: Fungal infection
- **Description**: Fungal infection is a common skin condition caused by fungi.
- **Precautions**: Bath twice, use detol or neem in bathing water, keep infected area dry, use clean cloths
- **Medications**: Antifungal Cream, Fluconazole, Terbinafine, Clotrimazole, Ketoconazole
- **Diet**: Antifungal Diet, Probiotics, Garlic, Coconut oil, Turmeric
- **Workout**: Avoid sugary foods, consume probiotics, increase intake of garlic, etc.

### Example 2: Common Cold
**Input Symptoms**: `continuous_sneezing, runny_nose, congestion`

**Output**:
- **Disease**: Common Cold
- **Description**: The common cold is a viral infection of the upper respiratory tract.
- Plus relevant precautions, medications, diet, and workout recommendations

## Development

### Training a New Model

1. Open `Untitled.ipynb` in Jupyter Notebook
2. Run all cells to train the model
3. The trained model will be saved as `svc.pkl`

### Adding New Diseases

To add support for new diseases:

1. Update `datasets/Training.csv` with new training data
2. Add disease information to:
   - `datasets/description.csv`
   - `datasets/precautions_df.csv`
   - `datasets/medications.csv`
   - `datasets/diets.csv`
   - `datasets/workout_df.csv`
3. Retrain the model using the Jupyter notebook
4. Update `diseases_list` dictionary in `main.py`

## Important Notes

- **Medical Disclaimer**: This system is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.
- **Model Accuracy**: While the model shows 100% accuracy on the test set, this may indicate overfitting. Use with caution and validate predictions with medical professionals.
- **Symptom Input**: Symptoms must match the exact format in the symptoms dictionary (use underscores instead of spaces)

## Requirements

- Python 3.x
- Flask
- NumPy
- pandas
- scikit-learn 1.4.1.post1

## Future Enhancements

- Add symptom auto-complete and suggestion features
- Implement user authentication and prediction history
- Add multi-language support
- Include severity assessment
- Integrate with telemedicine services
- Mobile application development
- Improve model with more diverse training data
- Add confidence scores to predictions

## License

This project is intended for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or support, please use the contact page in the web application.

## Team

This project was developed by:
- **Saksham Shrivastava** (22BCON333)
- **Gungun Choudhary** (22BCON322)
- **Saurabh Soni** (22BCON273)
- **Nikunj Varshaney** (22BCOL1413)

**JECRC University - Final Year Project**
