import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation import FCN_ResNet50_Weights

# === Step 1: Define the Pascal VOC colormap ===
PASCAL_VOC_COLORMAP = [
    [0, 0, 0],       # Background
    [128, 0, 0],     # Aeroplane
    [0, 128, 0],     # Bicycle
    [128, 128, 0],   # Bird
    [0, 0, 128],     # Boat
    [128, 0, 128],   # Bottle
    [0, 128, 128],   # Bus
    [128, 128, 128], # Car
    [64, 0, 0],      # Cat
    [192, 0, 0],     # Chair
    [64, 128, 0],    # Cow
    [192, 128, 0],   # Dining table
    [64, 0, 128],    # Dog
    [192, 0, 128],   # Horse
    [64, 128, 128],  # Motorbike
    [192, 128, 128], # Person
    [0, 64, 0],      # Potted plant
    [128, 64, 0],    # Sheep
    [0, 192, 0],     # Sofa
    [128, 192, 0],   # Train
    [0, 64, 128],    # TV monitor
]

def apply_colormap(mask):
    """
    Convert a 2D class-index mask to a color image using the PASCAL VOC colormap.
    """
    colormap = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(PASCAL_VOC_COLORMAP):
        colormap[mask == label] = color
    return colormap

# === Step 2: Load and preprocess image ===
img_path = './fcn_segmentation_trt/script/image_000.png'  # Provide the path to your image
img_pil = Image.open(img_path).convert("RGB")

# Define the transformations (normalization, resizing, etc.)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),  # Same as ImageNet normalization
])
# Apply transformations
input_tensor = transform(img_pil).unsqueeze(0)  # Shape: [1, 3, H, W]

# === Step 3: Load pretrained FCN model ===
model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
model.eval()  # Set the model to evaluation mode

# === Step 4: Run inference ===
with torch.no_grad():
    output = model(input_tensor)["out"][0]  # Shape: [21, H, W]
    predicted_mask = torch.argmax(output, dim=0).cpu().numpy()

# === Step 5: Apply colormap ===
colored_mask = apply_colormap(predicted_mask)

# === Step 6: Show results ===
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.imshow(img_pil)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 1, 2)
plt.imshow(colored_mask)
plt.title("Segmented Mask (PASCAL VOC Colors)")
plt.axis("off")

plt.tight_layout()
plt.show()
