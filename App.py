import torch 
import torch.hub
import torchvision.transforms as transforms

import os 
import csv
import json
import random
import sys 
from contextlib import redirect_stdout
import logging

import folium

import streamlit as st

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from transformers import pipeline

import numpy as np
import matplotlib.pyplot as plt

import cv2 
from PIL import Image, ImageEnhance

class SuppressOutput:
    def __enter__(self):
        self.devnull = open(os.devnull, 'w')
        self.old_stdout = sys.stdout
        sys.stdout = self.devnull
        return self.devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        self.devnull.close()

ml_model = load_model('C:\\Users\\siddu\\Desktop\\Alt\\Exports\\New_Ptz.keras')
def predict_pothole(imgpath):
    predict_image = image.load_img(imgpath, target_size = (64,64))
    predict_image = image.img_to_array(predict_image)
    predict_image = np.expand_dims(predict_image, axis=0)
    result = ml_model.predict(predict_image)
    x = result.max()

    return x

with SuppressOutput():
    torch_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
    torch_model.eval()
def torch_depth(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        image_tensor = image_tensor.type(torch.FloatTensor)
        depth_map = torch_model(image_tensor)
    depth_prediction = depth_map[0, depth_map.shape[1] // 2, depth_map.shape[2] // 2].item()
    keys = ["Severe Hazard Pothole","Moderate Risk Pothole","Minor Issue Pothole"]
    if depth_prediction>2000:
        return 2
    elif depth_prediction>1500:
        return 1
    else:
        return 0
    
def llm_depth(image_path):
    pipe = pipeline(task="depth-estimation", model="Intel/dpt-large")

    #image_path = "C:\\Users\\siddu\\Desktop\\validation\\potholes\\34.jpg"
    image = Image.open(image_path)
    image = ImageEnhance.Contrast(image).enhance(2)  
    image = ImageEnhance.Sharpness(image).enhance(2)  

    result = pipe(image)
    depth = result["depth"]


    depth_array = np.array(depth)
    depth_normalized = (255 * (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))).astype(np.uint8)
    depth_image = Image.fromarray(depth_normalized)
    depth_image.save("Depth_Map.png")

    plt.imshow(depth_normalized, cmap='gray')
    plt.colorbar()
    plt.show()

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image_cv, contours, -1, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    plt.show()

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        pothole_region = depth_array[y:y+h, x:x+w]

        min_depth = np.min(pothole_region)
        max_depth = np.max(pothole_region)
        avg_depth = np.mean(pothole_region)

        return([max_depth, min_depth, avg_depth])

    else:
        print("No pothole detected.")

def pick_random_photo():
    folder_path = "C:\\Users\\siddu\\Desktop\\Alt\\Test Data"
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    random_image = random.choice(image_files)
    return os.path.join(folder_path, random_image)

def get_routes():
    json_file = "C:\\Users\\siddu\\Desktop\\Alt\\Routes\\Combined.json"
    with open(json_file, 'r') as file:
        data = json.load(file)
        routes = data.get("routes", [])
        all_routes = {}
        for idx, route in enumerate(routes, start=1):
            coordinates = route.get("coordinates", [])
            coordinate_list = [[coord["lat"], coord["lng"]] for coord in coordinates]
            all_routes[f"Route_{idx}"] = coordinate_list
    return all_routes

def create_map_with_pothole_markers(potholes, map_file):
    base_map = folium.Map(location=[0, 0], zoom_start=2)

    intensity_colors = {
        "Severe Hazard Pothole": "red",
        "Moderate Risk Pothole": "orange",
        "Minor Issue Pothole": "green"
    }

    for lat, lon, intensity in potholes:
        color = intensity_colors.get(intensity, "blue")  
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(icon='exclamation-triangle', prefix='fa', color=color),
            popup=intensity
        ).add_to(base_map)

    base_map.save(map_file)

def rate_route(pothole_data):
    intensity_scores = {
        'Minor Issue Pothole': 1,
        'Moderate Risk Pothole': 2,
        'Severe Hazard Pothole': 3
    }
    total_score = 0
    pothole_count = {
        'Minor Issue Pothole': 0,
        'Moderate Risk Pothole': 0,
        'Severe Hazard Pothole': 0
    }
    
    for data in pothole_data:
        if len(data) >= 3:
            _, _, intensity = data
            if intensity in pothole_count:
                pothole_count[intensity] += 1
    for intensity, count in pothole_count.items():
        total_score += intensity_scores[intensity] * count
    max_possible_score = sum(intensity_scores[intensity] for intensity in pothole_count.keys()) * len(pothole_data)

    if max_possible_score > 0:
        road_condition_score = (total_score / max_possible_score) * 100
    else:
        road_condition_score = 0    
    return road_condition_score

def main():
    # Load custom CSS
    st.markdown(
        f"""
        <style>
        {open("style.css").read()}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Route Analyzer and Mapper")

    # Input locations
    start = st.text_input("Enter Start Location:")
    end = st.text_input("Enter End Location:")

    if st.button("Analyze Routes"):
        keys = ["Minor Issue Pothole", "Moderate Risk Pothole", "Severe Hazard Pothole"]
        routes = get_routes()
        Updates_routes = []

        for route_name, route_coords in routes.items():
            pothole_num = len(route_coords)
            pothole_count = 0
            updated_coords = []

            while pothole_count != pothole_num:
                path = pick_random_photo()
                prediction = 1
                if prediction == 1:
                    intensity_val = torch_depth(path)
                    intensity = keys[intensity_val]
                    current_cord = route_coords[pothole_count]
                    current_cord.append(intensity)
                    updated_coords.append(current_cord)
                    pothole_count += 1
                else:
                    pass
            
            Updates_routes.append(updated_coords)

        st.write("Analyzing All Routes...")

        count = 1
        scores = []
        st.markdown('<div class="map-header">Route Maps with Potholes</div>', unsafe_allow_html=True)
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        for route in Updates_routes:
            create_map_with_pothole_markers(route, f"Map{count}.html")
            score = rate_route(route)
            scores.append(score)
            rounded_score = round(score, 2)
            st.markdown(f"<h3>Route {count} Score: {rounded_score}%</h3>", unsafe_allow_html=True)
            st.components.v1.html(open(f"Map{count}.html").read(), width=400, height=500)
            count += 1
        st.markdown('</div>', unsafe_allow_html=True)

        with open("Map_Links.txt") as fh: 
            links = (fh.read()).split("\n")
        
        best_route_index = scores.index(min(scores))
        best_route_link = links[best_route_index]

        st.write("Grading And Selecting The Best Route...")
        st.markdown(f"<h2>Best Route: Route {best_route_index + 1}</h2>", unsafe_allow_html=True)
        st.markdown(f'<iframe src="{best_route_link}" width="600" height="450"></iframe>', unsafe_allow_html=True)
        st.markdown(f'[Click here to view the best route]({best_route_link})')

if __name__ == "__main__":
    main()