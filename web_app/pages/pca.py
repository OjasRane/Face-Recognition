import streamlit as st
from utils import N_SUBJECTS, TRAIN_PER_SUBJECT, TEST_PER_SUBJECT, IMG_SIZE, IMG_SHAPE
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Face Recognition Lab | Eigenfaces",
    page_icon=":material/familiar_face_and_zone:"
)

@st.cache_resource
def load(path: str):
    if path[-3:] == "npz":
        with np.load(path) as dataset:
            return dataset["train_faces"], dataset["test_faces"]
    elif path[-3:] == "pkl":
        payload = joblib.load(path)
        return payload["model"], payload["projected"]
    elif path[-3:] == "png":
        return plt.imread(path)
    else:
        raise ValueError("Expected path to either npz or pkl file.")

def find_match(test_image_vector, model, projection):
    test_image_vector = test_image_vector.reshape(-1, IMG_SIZE)
    projection_of_test_image = model.transform(test_image_vector)
    distance = np.linalg.norm(projection - projection_of_test_image, axis=1)
    min_distance_idx = np.argmin(distance)
    return min_distance_idx, distance[min_distance_idx]

st.title("Face Recognition using Eigenfaces", text_alignment="center")
st.markdown("<h4 align='center'>Check the confusion matrix below to see where the model fails.</h4>", unsafe_allow_html=True)

st.selectbox("Select subject", options=[f"Subject {i}" for i in range(N_SUBJECTS)], key="subject")
st.pills("Select Image", options=[f"Image {i}" for i in range(TEST_PER_SUBJECT)], default="Image 0", required=True, key="image")
st.session_state["test_face_idx"] = TEST_PER_SUBJECT*int(st.session_state["subject"].split(" ")[-1]) + int(st.session_state["image"].split(" ")[-1])

pca, projection = load("web_app/assets/models/pca/pca.pkl")
train_faces, test_faces = load("web_app/assets/datasets/olivetti_faces_dataset.npz")

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.imshow(test_faces[st.session_state["test_face_idx"]].reshape(IMG_SHAPE), cmap="gray", interpolation="bilinear")
    ax.set_title(f"{st.session_state['image']} of {st.session_state['subject']}")
    ax.set_axis_off()
    st.pyplot(fig)
    plt.close(fig)
with col2:
    min_distance_idx, distance = find_match(test_faces[st.session_state["test_face_idx"]], pca, projection)
    fig, ax = plt.subplots()
    ax.imshow(train_faces[min_distance_idx].reshape(IMG_SHAPE), cmap="gray", interpolation="bilinear")
    ax.set_title(f"Nearest match: Subject {min_distance_idx // TRAIN_PER_SUBJECT}")
    ax.set_axis_off()
    st.pyplot(fig)
    plt.close(fig)

if int(st.session_state["subject"].split(" ")[-1]) == min_distance_idx // TRAIN_PER_SUBJECT:
    st.markdown("<p align='center' style='font-size: 24px;'>Prediction is correct!</p>", unsafe_allow_html=True)
else:
    st.markdown("<p align='center' style='font-size: 24px;'>Prediction is incorrect!</p>", unsafe_allow_html=True)
st.markdown(f"<p align='center' style='font-size: 24px;'>Distance between projection of both images: {np.round(distance, 4)} units</p>", unsafe_allow_html=True)

st.markdown("<h3 align='center'>Confusion Matrix:</h3>", unsafe_allow_html=True)
confusion_matrix = load("web_app/assets/models/pca/confusion_matrix.png")
st.image(confusion_matrix)

if st.button("Home", icon=":material/home:"):
    st.switch_page("landing_page.py")