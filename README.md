# SMS / Email Spam & Phishing Detector (NLP Basics)

An interactive web app built with Python, scikit-learn, and Streamlit to classify SMS and email messages as legitimate (`ham`) or spam/phishing using text normalization, TF-IDF features, and Logistic Regression (with an optional Multinomial Naive Bayes mode).

## Project Structure

- `data/messages.csv`: Labeled SMS and email dataset (`text`, `label`)
- `src/preprocess.py`: Text cleaning and token normalization
- `src/train.py`: Trains TF-IDF + classifier and saves model artifacts to `models/`
- `app.py`: Interactive Streamlit web UI
- `requirements.txt`: Python dependencies
- `.gitignore`: Ignored virtual environment, cache, and local files

## Setup & Usage

1. Activate your virtual environment and install dependencies:
   `.\.venv\Scripts\Activate.ps1`
   `pip install -r requirements.txt`

2. Train the model (default `logreg`, or `nb` for `MultinomialNB`):
   `python src/train.py`
   (or `python src/train.py --model nb`)

3. Run the interactive web app:
   `streamlit run app.py`
