import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

st.set_page_config(page_title="Voting Ensemble Studio", layout="wide")

st.title("Voting Ensemble Studio 🗳️")
st.markdown("Upload a dataset, select multiple heterogeneous base models, and combine their predictions using a Voting Ensemble.")

# --- 1. File Upload ---
uploaded_file = st.file_uploader("Upload preprocessed dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head(), use_container_width=True)

    # --- 2. Data Configuration ---
    st.sidebar.header("1. Data Configuration")
    task_type = st.sidebar.selectbox("Task Type", ["Classification", "Regression"])
    
    target_col = st.sidebar.selectbox("Select Target Column", options=df.columns)
    feature_cols = st.sidebar.multiselect("Select Feature Columns", 
                                          options=[col for col in df.columns if col != target_col], 
                                          default=[col for col in df.columns if col != target_col])

    if feature_cols and target_col:
        X = df[feature_cols]
        y = df[target_col]

        # --- 3. Base Algorithms Selection ---
        st.sidebar.header("2. Select Base Models")
        st.sidebar.markdown("Choose the models to include in your ensemble:")
        
        available_models = ["Decision Tree", "Linear/Logistic Regression", "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)"]
        selected_models = st.sidebar.multiselect("Base Estimators", available_models, default=["Decision Tree", "Linear/Logistic Regression"])
        
        # --- 4. Voting Configuration ---
        st.sidebar.header("3. Voting Configuration")
        voting_type = 'hard'
        if task_type == "Classification":
            voting_type = st.sidebar.radio("Voting Type", ["hard", "soft"], 
                                           help="Hard voting uses majority rule. Soft voting averages predicted probabilities (requires models that support probability).")

        # --- 5. Validation Configuration ---
        st.sidebar.header("4. Validation Strategy")
        val_method = st.sidebar.radio("Method", ["Train-Test Split", "K-Fold Cross Validation"])
        
        if val_method == "Train-Test Split":
            test_size = st.sidebar.slider("Test Size Ratio", 0.1, 0.5, 0.2, step=0.05)
        else:
            k_folds = st.sidebar.slider("Number of Folds (K)", 2, 10, 5)

        # --- 6. Execution ---
        if st.button("Train & Evaluate Ensemble", type="primary"):
            if not selected_models:
                st.error("Please select at least one base model to train.")
            else:
                with st.spinner("Training model..."):
                    
                    # Build the list of estimators
                    estimators = []
                    if "Decision Tree" in selected_models:
                        if task_type == "Classification":
                            estimators.append(('dt', DecisionTreeClassifier(random_state=42)))
                        else:
                            estimators.append(('dt', DecisionTreeRegressor(random_state=42)))
                            
                    if "Linear/Logistic Regression" in selected_models:
                        if task_type == "Classification":
                            estimators.append(('lr', LogisticRegression(max_iter=1000, random_state=42)))
                        else:
                            estimators.append(('lr', LinearRegression()))
                            
                    if "Support Vector Machine (SVM)" in selected_models:
                        if task_type == "Classification":
                            # Soft voting requires probability=True for SVC
                            prob = True if voting_type == 'soft' else False
                            estimators.append(('svc', SVC(probability=prob, random_state=42)))
                        else:
                            estimators.append(('svr', SVR()))
                            
                    if "K-Nearest Neighbors (KNN)" in selected_models:
                        if task_type == "Classification":
                            estimators.append(('knn', KNeighborsClassifier()))
                        else:
                            estimators.append(('knn', KNeighborsRegressor()))

                    # Initialize Voting Model
                    if task_type == "Classification":
                        ensemble = VotingClassifier(estimators=estimators, voting=voting_type)
                    else:
                        ensemble = VotingRegressor(estimators=estimators)

                    st.subheader("Model Evaluation Results")
                    st.write(f"**Ensemble composed of:** {', '.join([name for name, _ in estimators])}")

                    if val_method == "Train-Test Split":
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                        ensemble.fit(X_train, y_train)
                        y_pred = ensemble.predict(X_test)
                        
                        col1, col2 = st.columns(2)
                        if task_type == "Classification":
                            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                            col2.metric("F1 Score (Weighted)", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
                        else:
                            col1.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
                            col2.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.4f}")

                    else: # K-Fold
                        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                        
                        if task_type == "Classification":
                            scores = cross_val_score(ensemble, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
                            metric_name = "Accuracy"
                        else:
                            scores = cross_val_score(ensemble, X, y, cv=kf, scoring='r2', n_jobs=-1)
                            metric_name = "R² Score"
                            
                        col1, col2 = st.columns(2)
                        col1.metric(f"Mean {metric_name} (across {k_folds} folds)", f"{scores.mean():.4f}")
                        col2.metric("Standard Deviation", f"{scores.std():.4f}")
                        
                        st.write(f"**Fold-by-fold {metric_name} scores:**")
                        st.bar_chart(scores)