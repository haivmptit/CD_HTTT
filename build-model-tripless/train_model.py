import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
import tensorflow_addons as tfa
from numpy import load
import numpy as np
from sklearn.preprocessing import LabelEncoder

def _base_network(): # Xây dựng mạng cơ sở là VGG16.
  model = VGG16(include_top = True, weights = None)
  model.summary()
  dense = Dense(128)(model.layers[-4].output)
  norm2 = Lambda(lambda x: tf.math.l2_normalize(x, axis = 1))(dense)
  model = Model(inputs = [model.input], outputs = [norm2])
  return model
def convert_label(y): # chuyển nhãn từ string về dạng số : 0,1,2,3,....
    out_encoder = LabelEncoder()
    out = out_encoder.fit(y)
    trainy = out_encoder.transform(y)
    return trainy, out

def normalize(tensors): #tensors là tập các ảnh shape(tensors) = (n,224,224,3)
    new_tensor =[]
    for i in tensors: #  i là một ma trận ảnh khuôn mặt
        i = i.astype('float32')
        # standardize pixel values across channels (global)
        #mean: giá trị trung bình pixel trong ảnh
        #std: độ lệch chuẩn
        mean, std = i.mean(), i.std()
        i = (i - mean) / std # chuẩn hóa lại giá trị mỗi pixel trong ảnh
        new_tensor.append(i)
    return new_tensor

model = _base_network()
model.summary() # in ra mô hình model
model.compile(
    optimizer=tf.optimizers.Adam(0.001),
    loss=tfa.losses.TripletSemiHardLoss()) # sử dụng loss = TripletSemiHardLoss

data = load('faces-dataset.npz') # load ảnh các khuôn mặt đã được xử lí
X_train, y_train, X_test, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print(np.shape(X_train), np.shape(y_train), np.shape(X_test),np.shape(y_test))
X_train = np.asarray(normalize(X_train))

y_train, out_encoder = convert_label(y_train)
print('Loaded: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
gen_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().shuffle(1024).batch(32)
history = model.fit(gen_train,
                    steps_per_epoch =50,
                    epochs=5)

model.save("model_triplot1.h5") # Lưu model đã train