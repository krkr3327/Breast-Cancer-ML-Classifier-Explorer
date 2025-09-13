import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Page Setup ----------------
st.set_page_config(page_title="🩺 Cancer Classifier", layout="wide")
st.title(" Breast Cancer Wisconsin ML Explorer ")
st.markdown("An elegant ML app to **Preprocess → Reduce → Classify → Evaluate** 💻✨")

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ App Controls")
st.sidebar.write("Customize steps here 👇")

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

with st.expander("🔍 Dataset Preview", expanded=True):
    st.write(f"Shape: {X.shape}")
    st.dataframe(X.head())

# ---------------- Cleaning ----------------
with st.expander("🧹 Data Cleaning"):
    st.write("This dataset is already clean, but let's pretend 😉")
    X = X.drop_duplicates()
    st.success(f"After dropping duplicates → Shape: {X.shape}")

# ---------------- Feature Selection ----------------
with st.expander("🎯 Feature Selection & Scaling", expanded=True):
    features = st.multiselect("Select Feature Columns", X.columns, default=X.columns[:8])
    if features:
        X = X[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        st.success("✅ Features scaled successfully!")

# ---------------- PCA ----------------
if 'features' in locals() and features:
    with st.expander("📉 PCA Visualization", expanded=True):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["Target"] = y.values

        fig, ax = plt.subplots()
        sns.scatterplot(x="PC1", y="PC2", hue="Target", data=pca_df, palette="Set1", s=60, ax=ax)
        ax.set_title("PCA - First 2 Components", fontsize=14)
        st.pyplot(fig)
        st.info(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

    # ---------------- Train/Test Split ----------------
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # ---------------- Model Selection ----------------
    model_choice = st.sidebar.selectbox("🤖 Choose Classifier", 
                                        ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"])

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=500)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = SVC(probability=True)

    # ---------------- Train Model ----------------
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # ---------------- ROC Curve ----------------
    with st.expander("📈 ROC Curve & AUC", expanded=True):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color="darkblue", lw=2, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve ({model_choice})")
        ax.legend(loc="lower right")
        st.pyplot(fig)

        st.success(f"🌟 Model: {model_choice} → AUC = {roc_auc:.3f}")

    # ---------------- Classification Report ----------------
    with st.expander("📋 Classification Report", expanded=False):
        y_pred = model.predict(X_test)
        st.text(classification_report(y_test, y_pred))
