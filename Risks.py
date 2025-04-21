import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

st.title("Road Accident Risk Feature Selection using Information Gain")

uploaded_file = st.file_uploader("Upload your Excel dataset", type=["xlsx"])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    data.dropna(inplace=True)

    # Encode categorical columns
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Change this to your actual target column name
    if 'Accident_Risk_Level' not in data.columns:
        st.error("Expected column 'Accident_Risk_Level' not found in dataset.")
    else:
        X = data.drop('Accident_Risk_Level', axis=1)
        y = data['Accident_Risk_Level']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = DecisionTreeClassifier(criterion='entropy')
        model.fit(X_train, y_train)

        feature_importance = model.feature_importances_

        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Information_Gain': feature_importance
        }).sort_values(by='Information_Gain', ascending=False)

        st.subheader("Feature Importances (Information Gain)")
        st.dataframe(importance_df)

        threshold = st.slider("Select Information Gain Threshold", 0.0, 0.2, 0.01, 0.01)

        selected_features = importance_df[importance_df['Information_Gain'] > threshold]['Feature']
        st.write(f"Selected Features at threshold {threshold}:", list(selected_features))

        # Train again using selected features
        X_train_sel = X_train[selected_features]
        X_test_sel = X_test[selected_features]

        final_model = DecisionTreeClassifier(random_state=42)
        final_model.fit(X_train_sel, y_train)
        y_pred = final_model.predict(X_test_sel)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')

        st.success(f"Accuracy: {acc:.2f}")
        st.success(f"Precision: {prec:.2f}")
        st.success(f"Recall: {rec:.2f}")
