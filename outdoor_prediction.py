import os
import sys

import cv2
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model


def make_square(image, square_size=224, inter=cv2.INTER_AREA):
	"""
	Fit image into square defined by square_size. Height/width ration is kept. Borders are
	created by black color.
	:param image: original image to be resized
	:param square_size: size of the square to fit in
	:param inter: method of interpolation of cv2
	:return: resized image into square
	"""
	(h, w) = image.shape[:2]
	ratio = float(square_size) / max(h, w)
	h_r, w_r = int(ratio * h), int(w * ratio)
	delta_w = square_size - w_r
	delta_h = square_size - h_r
	top, bottom = delta_h // 2, delta_h - (delta_h // 2)
	left, right = delta_w // 2, delta_w - (delta_w // 2)

	color = [0, 0, 0]
	resized = cv2.resize(image, (w_r, h_r), interpolation=inter)
	resized = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

	return resized


def convert_folder_into_data(input_folder):
	"""
	Read all images in the folder and resize them to square, so predictions can be made
	:param input_folder: path to input folder
	:return: loaded all picture and resized to square
	"""
	ret_data = None
	for root, dirs, files in os.walk(input_folder):
		inputs = np.array([make_square(cv2.imread(os.path.join(root, file))) for file in files])
		if len(inputs) == 0:
			continue
		if ret_data is None:
			ret_data = inputs
		else:
			ret_data = np.vstack((ret_data, inputs))
	return ret_data


def make_prediction(input_model, input_images, threshold=0.83):
	"""
	Make prediction using input model and input_images. Predictions are transferred into string
	interpretation Indoor/Outdoor. Everything above give threshold is considered as Indoor,
	otherwise it is Outdoor.
	:param input_model: model used for predictions
	:param input_images: images to be predicted
	:param threshold: threshold for splitting Indoor/Outdoor
	:return: labels for given images
	"""
	predictions = input_model.predict(input_images)
	return np.where(predictions > threshold, "Outdoor", "Indoor")


def load_prediction_model(path):
	"""
	Load model used for predictions. The base of model is VGG16 with modification
	of top layer. For model, weight saved on given path are loaded.
	:param path: path with saved weights
	:return: trained model for predictions
	"""
	base_model = VGG16(input_shape=(224, 224, 3), include_top=False, pooling='max')
	add_model = base_model.output
	add_model = Dense(1024)(add_model)
	add_model = BatchNormalization()(add_model)
	add_model = Activation("relu")(add_model)
	add_model = Dense(512)(add_model)
	add_model = BatchNormalization()(add_model)
	add_model = Activation("relu")(add_model)
	output_layer = Dense(1, activation="sigmoid")(add_model)

	model = Model(inputs=[base_model.input], outputs=[output_layer])
	model.compile(optimizer='adam', loss='categorical_crossentropy')

	model.load_weights(path)

	return model


def predict_for_files_with_model(model, input_path):
	"""
	Predict labels for all files in given input_file parameter. If input_file is only file,
	than create prediction for this file. If it is folder, than predict labels for all
	files in the folder.
	:param model: model used for prediction
	:param input_path: path with file
	:return: labels for all images on path
	"""
	if os.path.isfile(input_path):
		curr_data = np.array([make_square(cv2.imread(input_path))])
	elif os.path.isdir(input_path):
		curr_data = convert_folder_into_data(input_path)
	else:
		return
	predictions = make_prediction(model, curr_data)
	return predictions


def predict_files():
	"""
	Load model on given path and predict images for all paths, that
	were given in the arguments.
	"""
	model = load_prediction_model("./models/weights.h5")
	for arg in sys.argv[1:]:
		preds = predict_for_files_with_model(model, arg).reshape(1, -1)[0]
		print(",".join(preds))


if __name__ == "__main__":
	predict_files()
