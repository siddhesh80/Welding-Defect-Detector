# importing nescessary module 
import torch 
import streamlit as st
import cv2
from PIL import Image
import numpy as np

# load pretrained model best.pt
model=torch.hub.load('ultralytics/yolov5','custom',path='best.pt')


# Create Streamlit app
def main():
    st.title("WELDING DEFECT DETECTOR")

    st.subheader('Upload your welding picture to see the defect')

    uploaded_image = st.file_uploader("Choose a JPG file", type=["jpg", "jpeg"])

    if uploaded_image is not None:
        # Convert the uploaded image to PIL format
        pil_image = Image.open(uploaded_image)
        pil_image = np.array(pil_image)
        # Perform object detection
        result = model(pil_image).pandas().xyxy
        for a,b,c,d in result[0][['xmin','ymin','xmax','ymax']].values:
          a,b,c,d=int(a),int(b),int(c),int(d)
          clsname=result[0][['name']]['name'].values[0]
          cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
          rectangle=cv2.rectangle(cv_image,(a,b),(c,d),(0,255,0),3)
          image = cv2.putText(rectangle,str(clsname), (a,b), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,0,0), 2, cv2.LINE_AA)
          

        # Display the original and detected images
          st.image([pil_image,rectangle], caption=['Original Image','Defect Detected'], width=300)

# Run the Streamlit app
if __name__ == "__main__":
  main()


