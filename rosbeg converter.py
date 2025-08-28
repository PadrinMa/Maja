#!/usr/bin/env python3

from mcap.reader import make_reader
import numpy as np
import cv2
import os

mcap_file = "F:/rosbag2_2025_08_04-10_44_01/rosbag2_2025_08_04-10_44_01/rosbag2_2025_08_04-10_44_01_0.mcap"
image_topic = "/front/camera_left/image"
output_dir = "./test_data/images"

os.makedirs(output_dir, exist_ok=True)

with open(mcap_file, "rb") as f:
    reader = make_reader(f)
    i = 0
    for schema, channel, message in reader.iter_messages(topics=[image_topic]):
        # Qui i dati sono un messaggio ROS serializzato -> bisogna capire se è Image o CompressedImage
        # Se è CompressedImage (più probabile):
        img_array = np.frombuffer(message.data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is not None:
            cv2.imwrite(f"{output_dir}/frame_{i:06d}.jpg", img)
            i += 1

print(f"Salvate {i} immagini in {output_dir}")
