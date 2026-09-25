import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, SGDClassifier, SGDRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error, 
                             r2_score, confusion_matrix, roc_curve, auc)

st.set_page_config(page_title="Voting Ensemble Studio", layout="wide")

st.title("Voting Ensemble Studio")
st.markdown("Build a robust voting ensemble by selecting and tuning multiple heterogeneous base models, complete with advanced evaluation graphics.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload preprocessed dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    with st.expander("Preview Dataset"):
        st.dataframe(df.head(), use_container_width=True)

    # --- Data Configuration ---
    st.sidebar.header("1. Data Configuration")
    task_type = st.sidebar.selectbox("Task Type", ["Classification", "Regression"])
    
    target_col = st.sidebar.selectbox("Select Target Column", options=df.columns)
    feature_cols = st.sidebar.multiselect("Select Feature Columns", 
                                          options=[col for col in df.columns if col != target_col], 
                                          default=[col for col in df.columns if col != target_col])

    if feature_cols and target_col:
        X = df[feature_cols]
        y = df[target_col]

        # --- Base Algorithm Selection & Tuning ---
        st.sidebar.header("2. Base Models Configuration")
        
        if task_type == "Classification":
            available_models = ["Decision Tree", "Logistic Regression", "SVC", "Gaussian Naive Bayes", "SGD Classifier"]
        else:
            available_models = ["Decision Tree", "Linear Regression", "Ridge", "Lasso", "SVR", "SGD Regressor"]
            
        selected_models_names = st.sidebar.multiselect("Select Base Algorithms", available_models, default=available_models[:2])
        
        estimators = []
        
        for model_name in selected_models_names:
            st.sidebar.markdown(f"**Tune {model_name}**")
            
            if model_name == "Decision Tree":
                criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"] if task_type=="Classification" else ["squared_error", "absolute_error"], key=f"dt_crit")
                max_depth = st.sidebar.slider("Max Depth", 1, 50, 10, key=f"dt_depth")
                min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2, key=f"dt_split")
                if task_type == "Classification":
                    estimators.append((model_name, DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split)))
                else:
                    estimators.append((model_name, DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split)))

            elif model_name == "Logistic Regression":
                C = st.sidebar.number_input("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0, key="lr_c")
                penalty = st.sidebar.selectbox("Penalty", ["l2", "none"], key="lr_pen")
                pen = None if penalty == "none" else penalty
                estimators.append((model_name, LogisticRegression(C=C, penalty=pen, solver='lbfgs', max_iter=2000)))

            elif model_name == "Linear Regression":
                estimators.append((model_name, LinearRegression()))
                
            elif model_name in ["Ridge", "Lasso"]:
                alpha = st.sidebar.number_input("Regularization Strength (Alpha)", 0.01, 10.0, 1.0, key=f"{model_name}_alpha")
                if model_name == "Ridge":
                    estimators.append((model_name, Ridge(alpha=alpha)))
                else:
                    estimators.append((model_name, Lasso(alpha=alpha)))

            elif model_name in ["SVC", "SVR"]:
                C = st.sidebar.number_input("Regularization Parameter (C)", 0.1, 10.0, 1.0, key=f"svc_c")
                kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"], key=f"svc_kernel")
                prob = True if task_type == "Classification" else False # Needed for soft voting
                if task_type == "Classification":
                    estimators.append((model_name, SVC(C=C, kernel=kernel, probability=prob)))
                else:
                    estimators.append((model_name, SVR(C=C, kernel=kernel)))
                    
            elif model_name == "Gaussian Naive Bayes":
                estimators.append((model_name, GaussianNB()))
                
            elif model_name in ["SGD Classifier", "SGD Regressor"]:
                loss = st.sidebar.selectbox("Loss Function", ["hinge", "log_loss", "modified_huber"] if task_type=="Classification" else ["squared_error", "huber"], key=f"sgd_loss")
                alpha = st.sidebar.number_input("Alpha (Penalty)", 0.0001, 0.1, 0.0001, format="%.4f", key=f"sgd_alpha")
                if task_type == "Classification":
                    estimators.append((model_name, SGDClassifier(loss=loss, alpha=alpha, max_iter=1000)))
                else:
                    estimators.append((model_name, SGDRegressor(loss=loss, alpha=alpha, max_iter=1000)))

        # --- Voting Configuration ---
        st.sidebar.header("3. Voting Configuration")
        if task_type == "Classification":
            voting_type = st.sidebar.radio("Voting Method", ["hard", "soft"])
            st.sidebar.caption("*Note: Soft voting requires all base models to support probability estimates.*")
        else:
            voting_type = "hard" # Not applicable for regression, just a placeholder

        # --- Validation Configuration ---
        st.sidebar.header("4. Validation Strategy")
        val_method = st.sidebar.radio("Method", ["Train-Test Split", "K-Fold Cross Validation"])
        if val_method == "Train-Test Split":
            test_size = st.sidebar.slider("Test Size Ratio", 0.1, 0.5, 0.2, step=0.05)
        else:
            k_folds = st.sidebar.slider("Number of Folds (K)", 2, 10, 5)

        # --- Execution ---
        if len(estimators) < 2:
            st.warning("Please select at least 2 base algorithms to form a voting ensemble.")
        elif st.button("Train & Evaluate Voting Ensemble", type="primary"):
            with st.spinner("Training ensemble model..."):
                
                # Initialize Voting Model
                if task_type == "Classification":
                    ensemble = VotingClassifier(estimators=estimators, voting=voting_type, n_jobs=-1)
                else:
                    ensemble = VotingRegressor(estimators=estimators, n_jobs=-1)

                st.subheader("Model Evaluation Results")

                if val_method == "Train-Test Split":
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    
                    try:
                        ensemble.fit(X_train, y_train)
                        y_pred = ensemble.predict(X_test)
                        
                        # Metrics Display
                        col1, col2 = st.columns(2)
                        if task_type == "Classification":
                            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                            col2.metric("F1 Score (Weighted)", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
                            
                            # Graphics for Classification
                            st.divider()
                            st.markdown("### Evaluation Graphics")
                            g_col1, g_col2 = st.columns(2)
                            
                            with g_col1:
                                st.markdown("**Confusion Matrix**")
                                fig, ax = plt.subplots(figsize=(5,4))
                                cm = confusion_matrix(y_test, y_pred)
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                plt.ylabel('Actual')
                                plt.xlabel('Predicted')
                                st.pyplot(fig)
                                
                            with g_col2:
                                if voting_type == 'soft' and len(np.unique(y)) == 2:
                                    st.markdown("**ROC Curve**")
                                    y_prob = ensemble.predict_proba(X_test)[:, 1]
                                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                                    roc_auc = auc(fpr, tpr)
                                    
                                    fig2, ax2 = plt.subplots(figsize=(5,4))
                                    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                                    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                                    plt.xlabel('False Positive Rate')
                                    plt.ylabel('True Positive Rate')
                                    plt.legend(loc="lower right")
                                    st.pyplot(fig2)
                                else:
                                    st.info("ROC Curve is available for binary classification with 'Soft' voting enabled.")

                        else: # Regression
                            col1.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
                            col2.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.4f}")
                            
                            # Graphics for Regression
                            st.divider()
                            st.markdown("### Evaluation Graphics")
                            g_col1, g_col2 = st.columns(2)
                            
                            with g_col1:
                                st.markdown("**Actual vs Predicted**")
                                fig, ax = plt.subplots(figsize=(5,4))
                                sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.6)
                                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                                plt.xlabel('Actual Values')
                                plt.ylabel('Predicted Values')
                                st.pyplot(fig)
                                
                            with g_col2:
                                st.markdown("**Residual Plot**")
                                residuals = y_test - y_pred
                                fig2, ax2 = plt.subplots(figsize=(5,4))
                                sns.histplot(residuals, kde=True, ax=ax2, color='purple')
                                plt.xlabel('Residual Error')
                                st.pyplot(fig2)

                    except Exception as e:
                        st.error(f"Error during training: {e}. If using soft voting, ensure all models (like SVC) are set to compute probabilities or choose algorithms that inherently support it.")

                else: # K-Fold
                    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                    
                    try:
                        if task_type == "Classification":
                            scores = cross_val_score(ensemble, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
                            metric_name = "Accuracy"
                        else:
                            scores = cross_val_score(ensemble, X, y, cv=kf, scoring='r2', n_jobs=-1)
                            metric_name = "R² Score"
                            
                        col1, col2 = st.columns(2)
                        col1.metric(f"Mean {metric_name} (across {k_folds} folds)", f"{scores.mean():.4f}")
                        col2.metric("Standard Deviation", f"{scores.std():.4f}")
                        
                        st.markdown(f"**Fold-by-fold {metric_name} distribution:**")
                        fig, ax = plt.subplots(figsize=(8,3))
                        sns.lineplot(x=range(1, k_folds+1), y=scores, marker='o', ax=ax)
                        plt.xticks(range(1, k_folds+1))
                        plt.xlabel("Fold")
                        plt.ylabel(metric_name)
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error during cross-validation: {e}")
