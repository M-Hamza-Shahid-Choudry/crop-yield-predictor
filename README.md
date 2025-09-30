🌾 Crop Yield Predictor

An interactive Streamlit web app that predicts crop yield (kg/ha) using machine learning.
This project was built by M Hamza Shahid for Uraan AI Techathton 1.0.

The model uses historical data on rainfall, pesticides, temperature, country, and crop type to provide accurate yield predictions.

🚀 Features

🖥️ Web App built with Streamlit

📊 Input Parameters:

Country / Area

Crop Type

Year

Average Rainfall (mm/year)

Pesticides (tonnes)

Average Temperature (°C)

🤖 Machine Learning Pipeline:

Preprocessing functions (temperature categories, proxy humidity, etc.)

Encoders & scalers for numeric/categorical data

Custom feature selector (CorrelationThresholdSelector)

📈 Predicted Yield results in both hg/ha and kg/ha

💡 Interpretation Messages:

Below-average yield warning

Normal yield info

Excellent yield success

📂 Project Structure
├── app.py                   # Main Streamlit app
├── CropYieldPredictor.pkl   # Trained ML model (required)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

⚙️ Installation

Clone the repository:

git clone https://github.com/<your-username>/crop-yield-predictor.git
cd crop-yield-predictor


Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows


Install dependencies:

pip install -r requirements.txt

▶️ Usage

Make sure the trained model file CropYieldPredictor.pkl is in the same directory as app.py.

Run the Streamlit app:

streamlit run app.py


Open the link shown in your terminal (usually http://localhost:8501).

📊 Example Input
Area	Item	Year	Average Rainfall	Pesticides	Avg Temp
India	Maize	2023	800.0	5000.0	20.0
🧾 Output

Predicted Yield (in kg/ha and hg/ha)

Interpretation of yield quality

Debug info (expected features vs. input features)

🛠️ Tech Stack

Python

Streamlit

scikit-learn

pandas

NumPy

📌 Notes

Ensure that the feature names in your input data exactly match those expected by the trained model.

The model file is ~120 MB, so it may take a few seconds to load.

❤️ Acknowledgments

Built with love by M Hamza Shahid
for Uraan AI Techathton 1.0 🌾
