from Classifier import Cnn_Classifier
from Data_utils import *
import numpy as np

def build_yolo_cnn(images_np,output_np,test_images_np,test_output_np):

	input_shape = images_np.shape
	output_shape = output_np.shape
	number_of_classes = output_np.shape[3]-5
	conv_layers = [(64,3),(128,3),(128,3),(256,3),(256,3),(512,3),(512,3),(1024,1),(output_np.shape[3],1)]
	pooling_layers = [0,1,1,0,1,0,1,0,0]
	dropout_layers=[0,1,0,1,0,1,0,1,0]
	batchnorm_layers = [1,1,1,1,1,1,1,1,1]
	activation_type = "leaky relu"
	pool_type='max'
	learning_rate = 0.001
	lambda_obj = 50.
	lambda_coord=20.

	cnn = Cnn_Classifier(classifier_name = 'Yolo_Pascal',input_shape = input_shape,output_shape=output_shape,conv_layers =conv_layers,
		pooling_layers = pooling_layers,dropout_layers = dropout_layers,batchnorm_layers = batchnorm_layers, activation_type = activation_type,
		pool_type = pool_type, learning_rate = learning_rate,lambda_obj = lambda_obj,lambda_coord = lambda_coord)
	cnn.build_cnn_model()
	cnn.set_batch_size(1)
	cnn.train_classifier(input_images = images_np,output_encoded = output_np,testing_input =test_images_np,
		testing_output= test_output_np,iterations_num=300)
	quick_test = test_images_np[0:10]
	cnn.test_classifier_results(quick_test)

def get_input_and_output(from_file=1):

	if from_file == 0:
		imgs_out = load_training_xmls("../../training_data/VOCdevkit/VOC2012/Annotations/")
		imgs = load_training_images("../../training_data//VOCdevkit/VOC2012/JPEGImages/")
		yolo_output = build_yolo_output_np(imgs_out)
	else:
		imgs = load_nparray_from_file("imgs.npy")
		yolo_output = load_nparray_from_file("out.npy")
	# import ipdb;ipdb.set_trace()
	# show_real_boxes(imgs,imgs_out)
	imgs = imgs.reshape(-1,resized_shape[0],resized_shape[1],3)
	yolo_output = yolo_output[0]
	training_input = imgs[0:5]
	training_output = yolo_output[0:5]
	testing_input = imgs[5:10]
	testing_output = yolo_output[5:10]
	return training_input,training_output,testing_input,testing_output


def retrain_stored_checkpoint(images_np,output_np,test_images_np,test_output_np):
	
	input_shape = images_np.shape
	output_shape = output_np.shape
	learning_rate = 0.001
	cnn = Cnn_Classifier(classifier_name = "Yolo_Pascal",input_shape = input_shape,output_shape= output_shape,learning_rate = learning_rate,define_weights = False)
	cnn.set_batch_size(5) 
	cnn.retrain_classifier(input_images = images_np,output_classes = output_np,testing_input = test_images_np,testing_output = test_output_np
		,iterations_num = 10,checkpoint='710')


	cnn.compare_predict_with_true(images_np,output_np)
	quick_test = images_np[0:5]
	output_test = output_np[0:5]
	cnn.test_classifier_results(quick_test,output_test)



training_input,training_output,testing_input,testing_output = get_input_and_output(from_file=1)
retrain_stored_checkpoint(training_input,training_output,testing_input,testing_output)


#the current used checkpoint 95.60102343559265

#start time : 10:00 am
# time:  108 minute