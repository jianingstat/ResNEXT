### YOUR CODE HERE
# import tensorflow as tf
import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs
from ImageUtils import visualize


parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="train", help="train, test or predict")
parser.add_argument("--data_dir", default="./data", help="path to the data")
parser.add_argument("--save_dir", default="./saved_models", help="path to save the results")
parser.add_argument("--result_dir", default="./result/", help="path to save the predictions")
args = parser.parse_args()

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = MyModel(model_configs).to(device)

	if args.mode == 'train':
		x_train, y_train, x_test, y_test = load_data(args.data_dir)
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

		model.train(x_train, y_train, training_configs, x_valid, y_valid)
		model.evaluate(x_test, y_test)

	elif args.mode == 'test':
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)
		model.evaluate(x_test, y_test)

	elif args.mode == 'predict':
		# Loading private testing dataset
		x_test = load_testing_images(args.data_dir)
		# visualizing the first testing image to check your image shape
		visualize(x_test[0], 'test.png')
		# Predicting and storing results on private testing dataset 
		predictions = model.predict_prob(x_test)
		#np.save(args.result_dir, predictions)
		np.save(os.path.join(args.result_dir, 'predictions.npy'), predictions)


### END CODE HERE

