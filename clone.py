from keras.models import model_from_json
import csv
import cv2
import numpy as np

#read and store lines from the log file
lines = []
with open('../lane-lines/steering_angles.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

#extract path from files
images = []
measurements = []
'''for line in lines:
	if(line == lines[0]):
		line = lines[1]
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = '../CarND-Behavioral-Cloning-P3/data/IMG/' + filename
	image = cv2.imread(current_path)
	#print(image)
	images.append(image)
	#print(line)
	measurement = float(line[3])
	measurements.append(measurement)
'''
for i in range(10000, 40000):
	current_path = '../research/images/'
	image = cv2.imread(current_path + str(i) + '.jpg')
	images.append(image)
	measurement = float(lines[i][0])
	measurements.append(measurement)
#images are the input features, measurements are the output labels
X_train = np.array(images)
y_train = np.array(measurements)
print(X_train)
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers import Cropping2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3), name='INPUT'))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1, name='OUTPUT'))

model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=100)
print(model.predict(images))
model.save('model.h5')

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        #print(output_names)
        return frozen_graph

from keras import backend as K
import tensorflow as tf
# Create, compile and train model...
print([out for out in model.outputs])
print([inp for inp in model.inputs])
frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, ".", "model.pb", as_text=False)
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
