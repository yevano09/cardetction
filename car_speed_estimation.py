import cv2
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
import time

# Function to calculate centroid of a bounding box
def centroid(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

# Function to calculate distance between two points
def euclidean_dist(pt1, pt2):
    return dist.euclidean(pt1, pt2)

# Function to calculate speed of object (car) in pixels per second
def calculate_speed(previous_centroid, current_centroid, fps):
    pixels_per_meter = 20  # Adjust according to your camera's resolution and real-world distance
    meters_per_pixel = 1 / pixels_per_meter
    distance = euclidean_dist(previous_centroid, current_centroid) * meters_per_pixel
    speed = distance * fps
    return speed

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load pre-trained car detection classifier
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

# Dictionary to store previous centroid positions of detected cars
prev_centroids = OrderedDict()

# Start capturing and processing frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale for car detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, 1.1, 5)
    
    # Loop over detected cars
    for (x, y, w, h) in cars:
        # Calculate centroid of the car
        current_centroid = centroid(x, y, w, h)
        
        # Check if this is a new car or a known one
        if len(prev_centroids) == 0:
            # This is a new car, add it to the dictionary
            prev_centroids[current_centroid] = time.time()
        else:
            # Check for the closest existing centroid to this one
            closest_centroid = min(prev_centroids.keys(), key=lambda x: euclidean_dist(x, current_centroid))
            speed = calculate_speed(prev_centroids[closest_centroid], current_centroid, cap.get(cv2.CAP_PROP_FPS))
            cv2.putText(frame, f"Speed: {speed:.2f} pixels/s", current_centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            prev_centroids[current_centroid] = time.time()
            # Remove the closest centroid from the dictionary
            prev_centroids.pop(closest_centroid)
        
        # Draw rectangle around the car
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
