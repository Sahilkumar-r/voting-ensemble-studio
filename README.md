# Voting Ensemble Studio 🗳️🚀

A robust, interactive Streamlit web application that allows users to build, configure, and evaluate Voting Ensembles (combining multiple heterogeneous algorithms) on their own datasets without writing any code.

## 🌟 Features

*   **Custom Data Upload:** Upload any clean, preprocessed CSV dataset.
*   **Dynamic Configuration:** Automatically detects columns to let you easily select your target and features.
*   **Task Versatility:** Supports both Classification and Regression tasks.
*   **Multiple Base Estimators:** Select and combine different algorithms into one powerful ensemble:
    *   Decision Trees
    *   Linear/Logistic Regression
    *   Support Vector Machines (SVM)
    *   K-Nearest Neighbors (KNN)
*   **Voting Parameters:** Choose between **Hard Voting** (majority rule) or **Soft Voting** (averaging probabilities) for classification tasks.
*   **Validation Strategies:** Evaluate your model using either a standard Train-Test Split or K-Fold Cross Validation.

## 🛠️ Installation

First, clone this repository to your local machine and navigate into the directory:
```bash
git clone https://github.com/yourusername/voting-ensemble-studio.git
cd voting-ensemble-studio
```

Choose one of the following methods to install the dependencies:

### Option 1: Standard Installation using `pip`
It is recommended to set up a virtual environment first, then use the provided `requirements.txt` file:
```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Ultra-fast Installation using `uv`
If you use [uv](https://github.com/astral-sh/uv) (an extremely fast Python package installer and resolver), you can initialize a project environment and add the dependencies directly:
```bash
# Initialize a uv project
uv init

# Add the required packages
uv add streamlit pandas numpy scikit-learn
```
*(Note: You can also simply run `uv pip install -r requirements.txt` to use the existing file with uv).*

## 🚀 Usage

Run the Streamlit application from your terminal:

```bash
streamlit run app.py
```

This will open a new tab in your default web browser where you can interact with the Voting Ensemble Studio.

## 📝 How to use the App
1. **Upload Data:** Use the file uploader to provide your clean `.csv` file.
2. **Configure Data:** Select if your task is Classification or Regression, then define your Target variable and Feature columns.
3. **Select Base Algorithms:** Pick two or more different algorithms from the multiselect box to combine their unique strengths.
4. **Set Voting Config:** For classification, decide if you want the ensemble to use Hard or Soft voting.
5. **Evaluate:** Choose your validation method and click "Train & Evaluate Ensemble" to see your metrics!
