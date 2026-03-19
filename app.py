import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import requests

# Load the model once when the app starts
@st.cache_resource ##it makes sure the model is only downloaded and loaded once, not every time someone uploads an image
def load_model():
    model = models.resnet18(weights='DEFAULT')
    model.eval()
    return model

@st.cache_resource
def load_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    return requests.get(url).json()

def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def get_class_name(idx):
    # Download the ImageNet class labels
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels = requests.get(url).json()
    return labels[idx]

def classify(model, img_tensor):
    cat_indices = range(281, 286)
    with torch.no_grad():
        output = model(img_tensor)
    _, predicted_idx = torch.max(output, 1)
    idx = predicted_idx.item()
    return idx, idx in cat_indices

# --- The actual web page ---
st.title("🐱 Cat Detector")
st.write("Upload an image and I'll tell you if it's a cat!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Your image", use_column_width=True)

    with st.spinner("Analysing..."):
        model = load_model()
        img_tensor = preprocess(img)
        idx, is_cat = classify(model, img_tensor)

    if is_cat:
        st.success(f"✅ That's a CAT! ({get_class_name(idx)})")
    else:
        st.error(f"❌ Not a cat! The model thinks it's a **{get_class_name(idx)}** (class #{idx})")