import streamlit as st
import pandas as pd
import os
import glob
from PIL import Image

# Paths
OUTPUT_DIR = "outputs"
LOG_FILE = os.path.join(OUTPUT_DIR, "occupancy_log.txt")

st.set_page_config(page_title="Student Monitoring & Occupancy Analytics", layout="wide")

st.title("📊 Student Monitoring & Occupancy Analytics")

# Read log
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()[1:]  # Skip header
    data = [line.strip().split(", ") for line in lines]
    df = pd.DataFrame(data, columns=["Timestamp", "People Count", "Occupancy"])

    # Convert types
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["People Count"] = df["People Count"].str.extract(r"(\d+)").astype(int)
    df["Occupancy"] = df["Occupancy"].str.extract(r"([\d.]+)").astype(float)

    # Show line chart
    st.subheader("📈 People Count Over Time")
    st.line_chart(df.set_index("Timestamp")["People Count"])

    st.subheader("📉 Occupancy % Over Time")
    st.line_chart(df.set_index("Timestamp")["Occupancy"])
else:
    st.warning("Log file not found. Run app.py first to generate data.")

# Show latest screenshot
st.subheader("🖼 Latest Screenshot")
image_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "frame_*.jpg")), reverse=True)

if image_files:
    image = Image.open(image_files[0])
    st.image(image, caption=os.path.basename(image_files[0]), use_column_width=True)
else:
    st.info("No screenshots found yet.")
