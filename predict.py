from keras.models import load_model
import cv2
import numpy as np

model = load_model('model.h5') # insert file name of model
start = 10000 # first and last image wanted for prediction
end = 16000
images = []
current_path = '../research/images/' # path to the images to predict on
for i in range(start,end):
	image = cv2.imread(current_path + str(i) + '.jpg')
	images.append(image)
prediction = model.predict(np.array(images))
print(max(prediction), min(prediction), prediction.mean())
