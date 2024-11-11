import os
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoProcessor, FocalNetForImageClassification
from typing import List



# Model and feature extractor setup
model_path = "MichalMlodawski/nsfw-image-detection-large"
feature_extractor = AutoProcessor.from_pretrained(model_path)
model = FocalNetForImageClassification.from_pretrained(model_path)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Mapping from model labels to NSFW categories
label_to_category = {
    "LABEL_0": "Safe",
    "LABEL_1": "Questionable",
    "LABEL_2": "Unsafe"
}

# @app.post("/classify-nsfw")
def classify_nsfw_images(image, file_name):
    results = []

   
    image_tensor = transform(image).unsqueeze(0)
    
    # Process image using feature_extractor
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Prediction using the model
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get the label and category
    label = model.config.id2label[predicted.item()]
    category = label_to_category.get(label, "Unknown")
    emoji = {"Safe": "✅", "Questionable": "⚠️", "Unsafe": "🔞"}.get(category, "❓")
    confidence_percentage = confidence.item() * 100
    
    results.append({
        "file_name": file_name,
        "model_label": label,
        "confidence": f"{confidence_percentage:.2f}%",
    })

    return (results[0])


# Run the app with `uvicorn <filename>:app --reload`
