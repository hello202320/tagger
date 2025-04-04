import streamlit as st
import numpy as np
import onnxruntime as rt
from PIL import Image
import pandas as pd

# Load model
MODEL_PATH = "models/model.onnx"
LABELS_PATH = "models/selected_tags.csv"

session = rt.InferenceSession(MODEL_PATH)

# Load labels
tags_df = pd.read_csv(LABELS_PATH)
tag_names = tags_df["name"].tolist()

st.title("Onxx Tagger")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    target_size = session.get_inputs()[0].shape[1]
    image = image.resize((target_size, target_size))
    img_array = np.array(image).astype(np.float32)
    img_array = img_array[:, :, ::-1]  # Convert RGB to BGR
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    preds = session.run([label_name], {input_name: img_array})[0][0]

    # Display results
    results = sorted(
        zip(tag_names, preds), key=lambda x: x[1], reverse=True
    )  # Sort by confidence
    st.subheader("Detected Tags:")
    for tag, score in results:
        if score > 0.35:
            st.write(f"{tag}: {score * 100:.1f}%")
