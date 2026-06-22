import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Thumbnail Click Probability", layout="wide") 
st.title("Thumbnail Click Probability Prediction")
st.write("This app predicts the probability of a user clicking on a thumbnail based on various features.")  

@st.cache_data
def load_data():
    df = pd.read_csv('thumbnail-data.csv')
    X = df.drop('clicked', axis=1)
    y = df['clicked']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    return model, df

model, df = load_data()

st.sidebar.header("Input Features")

user_brightness = st.sidebar.slider("Brightness level(0-100)", 0, 100, 50)
text_length = st.sidebar.slider("Text length(0-100)", 0, 100, 50)

prediction = model.predict([[user_brightness, text_length]])[0]
probability = model.predict_proba([[user_brightness, text_length]])[0][1]*100

st.subheader("Prediction Result")
if prediction == 1:
    st.success(f"The model predicts that the user will click on the thumbnail with a probability of {probability:.2f}%.")

else:
    st.error(f"The model predicts that the user will not click on the thumbnail with a probability of {100 - probability:.2f}%.")

st.subheader("Feature Importance")
st.write("This S-curve shows how the probability changes as thumbnails get brighter and text gets shorter.")


smooth_brightness = np.linspace(0, 100, 100)
constant_text = np.full(100, text_length)
X_smooth = np.column_stack((smooth_brightness, constant_text))
y_smooth_prob = model.predict_proba(X_smooth)[:, 1]

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(smooth_brightness, y_smooth_prob, color='blue', label='Probability Curve')

# Plot the user's specific thumbnail as a red dot
ax.scatter([user_brightness], [probability/100], color='red', s=100, zorder=5, label='Your Thumbnail')

# Draw a dashed line at 50% (The decision boundary)
ax.axhline(0.5, color='gray', linestyle='--', label='50% Threshold')

ax.set_xlabel("Brightness Level")
ax.set_ylabel("Probability of Click")
ax.set_title("Click Probability vs. Brightness")
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)

st.pyplot(fig)