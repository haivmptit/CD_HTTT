from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import numpy as np
from os import listdir
from os.path import isdir
from numpy import savez_compressed
from face_recognize.load_face import extract_face

#load ảnh từ thư mục train/test
def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        path = directory + filename # đường dẫn ảnh
        face = extract_face(path, required_size=(224, 224)) # Lấy khuôn mặt từ thư mục ảnh mỗi người.
        faces.append(face)
    return faces

def load_dataset(directory):
    X, y = list(), list()
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        if not isdir(path):
            continue
        faces = load_faces(path) # lấy tất cả khuôn mặt từ ảnh trong mỗi path
        # gán nhãn cho mỗi khuôn mặt bằng tên thư mục chưa ảnh khuôn mặt của người đó
        labels = [subdir for _ in range(len(faces))]
        print('<3loaded %d examples for class: %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

# data = load('./gdriver//My Drive//train_model//FaceNet_keras_myself//build-tripless//faces-dataset.npz')
# load train dataset
trainX, trainy = load_dataset('dataset/train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('dataset/test/')
# Lưu các khuôn mặt với nhãn tương ứng lại.
savez_compressed('faces-dataset.npz', trainX, trainy, testX, testy)

