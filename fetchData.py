import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import requests

# 1. SETUP THE PATH
# This looks in the 'data/train' folder we extracted from Kaggle
#os.path.join builds a file path in an OS-safe way (data/train/cat.1.jpg). The function then checks it actually exists before doing anything.
img_path = os.path.join("data", "test1", "3.jpg")

def run_classifier():
    if not os.path.exists(img_path):
        print(f"Error: Could not find the image at {img_path}")
        return

    # 2. PRE-PROCESS THE IMAGE
    # We convert to RGB to fix the 4-channel [4, 224, 224] issue
    img = Image.open(img_path).convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #After converting to a 0–1 tensor, it normalises each channel using the mean and standard deviation of the entire ImageNet dataset (the massive dataset ResNet was trained on). 
    # This makes the pixel values match what the model "expects" to see.
    ])
    
    img_tensor = preprocess(img).unsqueeze(0) # Add batch dimension, A batch is just a group of images processed together at the same time.
    #.unsqueeze(0) adds a batch dimension, changing the shape from [3, 224, 224] to [1, 3, 224, 224]. Models expect a batch of images, even if it's just one — the 1 means "a batch of 1 image".
    #1 image, with 3 colour channels(RGB), that is 224 pixels tall and 224 pixels wide

    # 3. LOAD THE BRAIN (ResNet18)
    print("Loading AI model...")
    model = models.resnet18(weights='DEFAULT')
    #resnet18 is a well-known neural network architecture. It downloads pre-trained weights (the "brain" — millions of numbers tuned by training on ImageNet).
    model.eval()
    # .eval() switches the model to inference mode — some layers like dropout behave differently during training vs. actual use, and this ensures it behaves correctly for prediction.

    # 4. RUN THE PREDICTION
    with torch.no_grad():
        output = model(img_tensor)
        #torch.no_grad() tells PyTorch not to track gradients — gradients are only needed during training to update the model. Disabling them here saves memory and makes it faster. output ends up as a tensor of 1000 scores, one for each ImageNet class.
    
# 5. TRANSLATE THE RESULT
    _, predicted_idx = torch.max(output, 1)
    #torch.max finds the highest score across all 1000 classes. The _ discards the actual score value — you only care about which class won (the index). .item() then converts it from a tensor to a plain Python integer.
    idx = predicted_idx.item()

    # ImageNet Cat Range: 281 (tabby) to 285 (Egyptian cat)
    # Also includes 287 (lynx) and others.
    cat_indices = range(281, 286) 
    #ImageNet has 1000 classes, and indices 281–285 happen to all be cat breeds (tabby, Persian, etc.). If the predicted index falls in that range, it's a cat. Simple but effective.
    
    print(f"\n--- AI ANALYSIS ---")
    print(f"Predicted Class Index: {idx}")
    
    # 6. THE VERDICT
    if idx in cat_indices:
        print("Final Verdict: CAT! 🐾")
    else:
        print("Final Verdict: NOT A CAT! 🚫")
        print("Note: This model thinks it is class index #", idx)

if __name__ == "__main__":
    run_classifier()