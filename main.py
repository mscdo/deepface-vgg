from deepface import DeepFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
import os, sys
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
from deepface.commons import functions
from deepface.commons import functions, realtime, distance as dst

#--------------------------
employees = []

for r, d, f in os.walk('./dataset'): # r=root, d=directories, f = files
    for file in f:
        if ('.jpeg' in file):
            exact_path = r + "/" + file
            employees.append(exact_path)
print(employees)

#--------------------------
representations = []
for employee in employees:
    print(employee)
    representation = DeepFace.represent(img_path = employee, model_name = "VGG-Face")[0]["embedding"]
    instance = []
    instance.append(employee)
    instance.append(representation)
    representations.append(instance)
#--------------------------
import pickle
f = open('representations.pkl', "wb")
pickle.dump(representations, f)
f.close()

	
target_path = "target.jpeg"
target_img = DeepFace.extract_faces(img_path = target_path)[0]["face"]
target_representation = DeepFace.represent(img_path = target_path, model_name = "VGG-Face")[0]["embedding"]

#load representations of faces in database
f = open('representations.pkl', 'rb')
representations = pickle.load(f)
 
distances = []
for i in range(0, len(representations)):
    source_name = representations[i][0]
    source_representation = representations[i][1]
    distance = dst.findCosineDistance(source_representation, target_representation)
    print('Source:',source_name , ' ;  Distance: ', distance)
    distances.append(distance)
#find the minimum distance index
idx = np.argmin(distances)
matched_name = representations[idx][0]
print(matched_name)