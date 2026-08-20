import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np
from utils import N_SUBJECTS, TRAIN_PER_SUBJECT, TEST_PER_SUBJECT, IMG_SIZE, IMG_SHAPE

st.set_page_config(
    page_title="Face Recognition Lab | Fisherfaces",
    page_icon=":material/familiar_face_and_zone:"
)

@st.cache_resource
def load(path: str):
    if path[-3:] == "npz":
        with np.load(path) as dataset:
            return dataset["train_faces"], dataset["test_faces"]
    elif path[-3:] == "pkl":
        model = joblib.load(path)
        return model
    elif path[-3:] == "png":
        return plt.imread(path)
    else:
        raise ValueError("Expected path to either npz or pkl file.")

st.title("Face Recognition using Fisherfaces", text_alignment="center")
st.markdown("<h4 align='center'>Check the confusion matrix below to see where the model fails.</h4>", unsafe_allow_html=True)

st.selectbox("Select subject", options=[f"Subject {i}" for i in range(N_SUBJECTS)], key="subject")
st.pills("Select Image", options=[f"Image {i}" for i in range(TEST_PER_SUBJECT)], default="Image 0", required=True, key="image")
st.session_state["test_face_idx"] = TEST_PER_SUBJECT*int(st.session_state["subject"].split(" ")[-1]) + int(st.session_state["image"].split(" ")[-1])

lda = load(r"web_app/assets/models/lda/lda.pkl")
train_faces, test_faces = load(r"web_app/assets/datasets/olivetti_faces_dataset.npz")

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.imshow(test_faces[st.session_state["test_face_idx"]].reshape(IMG_SHAPE), cmap="gray", interpolation="bilinear")
    ax.set_title(f"{st.session_state['image']} of {st.session_state['subject']}")
    ax.set_axis_off()
    st.pyplot(fig)
    plt.close(fig)
with col2:
    fig, ax = plt.subplots()
    rng = np.random.default_rng()
    random_img_idx = rng.integers(0, TRAIN_PER_SUBJECT, size=1)
    predicted_class = lda.predict(test_faces[st.session_state["test_face_idx"]].reshape(1, -1))
    predicted_class_subject = train_faces[predicted_class*TRAIN_PER_SUBJECT+random_img_idx].reshape(IMG_SHAPE)

    fig, ax = plt.subplots()
    ax.imshow(predicted_class_subject, cmap="gray", interpolation="bilinear")
    ax.set_title(f"Predicted class: Subject {predicted_class[0]}")
    ax.set_axis_off()
    st.pyplot(fig)
    plt.close(fig)

if int(st.session_state["subject"].split(" ")[-1]) == predicted_class:
    st.markdown("<p align='center' style='font-size: 24px;'>Prediction is correct!</p>", unsafe_allow_html=True)
else:
    st.markdown("<p align='center' style='font-size: 24px;'>Prediction is incorrect!</p>", unsafe_allow_html=True)

st.markdown("<h3 align='center'>Confusion Matrix:</h3>", unsafe_allow_html=True)
confusion_matrix = load("web_app/assets/models/lda/confusion_matrix.png")
st.image(confusion_matrix)

if st.button("Home", icon=":material/home:"):
    st.switch_page("landing_page.py")