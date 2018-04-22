from keras.models import load_model
import cv2
import numpy as np

model = load_model('model.h5')

images = []
current_path = '../research/images/'
for i in range(10000,16000):
	image = cv2.imread(current_path + str(i) + '.jpg')
	images.append(image)
prediction = model.predict(np.array(images))
print(max(prediction), min(prediction), prediction.mean())
