import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LinearRegression
import cv2
st.set_page_config(page_title="SmartSense", layout="centered")
st.title("🍱 SmartSense")
st.write("AI-based Smart Label Freshness Detection System")
st.subheader("Upload Smart Label Image")
image_file = st.file_uploader("Upload CURRENT Label Image", type=["jpg", "png"])
st.markdown("### Color Interpretation")
st.write("🟣 Purple → safe")
st.write("🔵 Blue → Medium Risk")
st.write("🟢 Green → Unsafe")
def crop_label_area(img):
    width, height = img.size
    return img.crop((width*0.6, height*0.6, width*0.9, height*0.9))
def classify_color(image):
    image = image.convert("RGB")
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([160, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    green_count = np.sum(green_mask > 0)
    blue_count = np.sum(blue_mask > 0)
    purple_count = np.sum(purple_mask > 0)
    if green_count > blue_count and green_count > purple_count:
        return "Green", green_count
    elif blue_count > purple_count:
        return "Blue", blue_count
    else:
        return "Purple", purple_count
st.sidebar.header("Storage Conditions")
food_type = st.sidebar.selectbox(
    "Food Type",
    ["Milk", "Chicken", "Fruits"]
)
temp = st.sidebar.slider("Temperature (°C)", 0, 40, 25)
time = st.sidebar.slider("Storage Time (hours)", 0, 72, 12)
data = pd.DataFrame({
    "temp": [5, 10, 20, 30, 35],
    "time": [48, 36, 24, 12, 6],
    "life": [72, 60, 36, 18, 8]
})
X = data[["temp", "time"]]
y = data["life"]
model = LinearRegression()
model.fit(X, y)
input_data = pd.DataFrame([[temp, time]], columns=["temp", "time"])
remaining_life = int(model.predict(input_data)[0])
st.sidebar.subheader("Prediction")
st.sidebar.write(f" Remaining Life: {remaining_life} hours")
if image_file:
    img = Image.open(image_file)
    st.image(img, caption="Original Image", width=300)
    label_img = crop_label_area(img)
    st.image(label_img, caption="Detected Label Area", width=200)
    st.success("✅ Analysis Completed Successfully")
    detected_color, pixel_count = classify_color(label_img)
    st.subheader(f"Detected Color: {detected_color}")
    st.caption(f"Confidence (pixel match): {pixel_count}")
    if detected_color == "Purple":
        st.error("🟣 Purple → Unsafe")
    elif detected_color == "Blue":
        st.warning("🔵 Blue → Medium Risk")
    elif detected_color == "Green":
        st.error("🟢 Green → Unsafe")
    else:
        st.info("⚪ Unable to determine")
    st.info(f"📦 Food Type: {food_type}")
    st.info(f"🌡 Temperature: {temp}°C")
    st.info(f" Estimated Remaining Life: {remaining_life} hours")
st.markdown("---")
st.caption("SmartSense uses HSV-based color detection for accurate real-world analysis.")
st.markdown("+++")