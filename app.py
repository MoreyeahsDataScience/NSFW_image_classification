import streamlit as st
from PIL import Image
from classification import classify_nsfw_images

st.title("NSFW Image Classification DEMO")
st.write("This APP uses a pre-trained deep learning model to classify uploaded images into NSFW (Not Safe for Work) categories. This api helps us identify explicit digital content that's inappropriate for viewing in public or at work. It includes multiple categories such as 18+, and results images with flag such as Safe, Questionable , or Unsafe . It returns the model's classification label, confidence score, and NSFW category for each image.")


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

