import streamlit as st
from PIL import Image
from classification import classify_nsfw_images

# Title of the Streamlit app
st.title("Inappropriate Image Classifier")

# Upload image section

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    file_name= uploaded_image.name
# Process image and extract text
if uploaded_image is not None:
    
    # Display the uploaded image
    image = Image.open(uploaded_image)
    
    if st.button("Show Image"):
        st.image(image, caption="Uploaded Image", use_container_width=True)
    file_name= uploaded_image.name
    result= classify_nsfw_images(image, file_name)
    # print(result)
    
    # Display the extracted text
    st.subheader("Result")
    st.write(result)
else:
    st.info("Please upload an image.")

