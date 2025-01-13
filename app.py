!pip install requests streamlit torch torchvision torchaudio matplotlib Pillow pycocotools


import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image


import streamlit as st
from io import BytesIO

#BACKEND------------------------------------------
# Load Faster R-CNN with ResNet-50 backbone
def get_model(num_classes):
    # Load pre-trained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Initialize the model
num_classes = 8  # Background + chair + person + table
device = torch.device('cpu')
# Load the trained model
model = get_model(num_classes)
model.load_state_dict(torch.load("./fasterrcnn_resnet50_epoch_5.pth"))
model.to(device)
model.eval()  # Set the model to evaluation mode


def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Open image
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Convert image to tensor and add batch dimension
    return image_tensor.to(device)

def GetPred(image_path):
    # Load the unseen image
    image_tensor = prepare_image(image_path)
    
    with torch.no_grad():  # Disable gradient computation for inference
        prediction = model(image_tensor)
    
    # `prediction` contains:
    # - boxes: predicted bounding boxes
    # - labels: predicted class labels
    # - scores: predicted scores for each box (confidence level)
    COCO_CLASSES = {1 : "Other lesions", 2: "Osteophytes", 3: "Spondylolysthesis", 4: "Disc space narrowing", 5: "Foraminal stenosis", 6: "Surgical implant", 7: "Vertebral collapse", 8: "No finding"}
    
    
    def get_class_name(class_id):
        return COCO_CLASSES.get(class_id, "Unknown")
        
    # Draw bounding boxes with the correct class names and increase image size
    def draw_boxes(image, prediction, fig_size=(10, 10)):
        boxes = prediction[0]['boxes'].cpu().numpy()  # Get predicted bounding boxes
        labels = prediction[0]['labels'].cpu().numpy()  # Get predicted labels
        scores = prediction[0]['scores'].cpu().numpy()  # Get predicted scores
        
        # Set a threshold for showing boxes (e.g., score > 0.5)
        threshold = 0.5
        
        # Set up the figure size to control the image size
        figure = plt.figure(figsize=fig_size)  # Adjust the figure size here
    
        for box, label, score in zip(boxes, labels, scores):
            if score > threshold:
                x_min, y_min, x_max, y_max = box
                class_name = get_class_name(label)  # Get the class name
                plt.imshow(image)  # Display the image
                plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                                  linewidth=2, edgecolor='r', facecolor='none'))
                plt.text(x_min, y_min, f"{class_name} ({score:.2f})", color='r')
        
        plt.axis('off')  # Turn off axis
        st.pyplot(figure)
        plt.show()
    
    # Display the image with bounding boxes and correct labels
    draw_boxes(Image.open(image_path), prediction, fig_size=(12, 10))  # Example of increased size

#BACKEND----------------------------------------------

st.title("LesionNNet: ​Transforming Spinal Lesion Diagnosis with AI")
st.write("​LesionNNet is Convolutional Neural Network Model for Spinal Lesion Detection & Classification. Detecting Spinal lesions is the key to detecting many types of degenerative disc diseases and other problems related to the spine. Often these are difficult to detect in X-Rays and require a MRI or CT Scan. ")
st.write("")
st.write("Try it for yourself. Upload your X-Ray Scan below")
file = st.file_uploader("Upload File", type=["png", "jpg"])
show_file = st.empty()
if not file:
    show_file.info("Please upload a file of type: " + ", ".join(["png", "jpg"]))
elif isinstance(file, BytesIO):
    GetPred(file)
    file.close()

st.write("By Visharad Upadhyay")
st.write("[LinkedIn](%s)" % "https://www.linkedin.com/in/visharad-upadhyay/")
