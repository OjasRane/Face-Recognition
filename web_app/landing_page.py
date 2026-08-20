import streamlit as st

st.set_page_config(
    page_title="Face Recognition Lab",
    page_icon=":material/familiar_face_and_zone:"
)

st.title("Face Recognition Lab", text_alignment="center")
st.markdown("<h2 align='center'>Explore face recognition projects here!</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True, key="PCA", horizontal_alignment="center"):
        st.markdown("<h2 align='center'>Face Recognition using PCA (Eigenfaces)</h2>", unsafe_allow_html=True)
        st.text("Explore the eigenfaces algorithm for face recognition trained on Olivetti Faces dataset.", text_alignment="center")
        if st.button("Explore Now!", key="explore_pca"):
            st.switch_page("pages/pca.py")

with col2:
    with st.container(border=True, key="LDA", horizontal_alignment="center"):
        st.markdown("<h2 align='center'>Face Recognition using LDA (Fisherfaces)</h2>", unsafe_allow_html=True)
        st.text("Explore the fisherfaces algorithm for face recognition trained on Olivetti Faces dataset.",text_alignment="center")
        if st.button("Explore Now!", key="explore_lda"):
            st.switch_page("pages/lda.py")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True, key="notebooks", horizontal=True, horizontal_alignment="center", height=350):
        st.markdown("<h2 align='center'>Check out the mathematics</h2>", unsafe_allow_html=True)
        st.text("Check out my notebooks on Face Recognition using PCA and LDA", text_alignment="center")
        st.link_button("Face Recognition using PCA",
                       url="https://OjasRane.github.io/Face-Recognition/face_recognition_using_PCA.html",
                       icon=":material/import_contacts:")
        st.link_button("Face Recognition using LDA",
                       url="https://OjasRane.github.io/Face-Recognition/face_recognition_using_LDA.html",
                       icon=":material/import_contacts:")

with col2:
    with st.container(border=True, key="repo", horizontal=True, horizontal_alignment="center", height=350):
        st.markdown("<h2 align='center'>GitHub Repository</h2>", unsafe_allow_html=True)
        st.text("Check out the GitHub repository for source code.", text_alignment="center")
        st.link_button("GitHub Repository", url="https://github.com/OjasRane/Face-Recognition")

with st.container(border=True, height=250, key="connect_with_me", horizontal_alignment="center"):
    st.markdown("## Connect with me", text_alignment="center")
    st.image("https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white",
             link="https://github.com/OjasRane", width=180)
    st.image("https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white",
             link="https://linkedin.com/in/ojasrane-in", width=180)