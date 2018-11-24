import numpy as np
import tensorflow as tf
import os
import random
import cv2
from nms import filter_with_nms
from Data_utils import *

class Cnn_Classifier:

	def __init__(self,classifier_name,input_shape = [None,28,28,1],output_shape = [None,9,9,25],conv_layers = [(32,5),(64,5),(128,3),(256,2),(256,1)]
		,pooling_layers = [1,1,1,0,0],dropout_layers=[1,1,1,1,1], batchnorm_layers = [1,1,1,1,1],activation_type = "relu",pool_type='avg'
		,learning_rate = 0.0003,dropout=.75,define_weights = True,lambda_obj=5.,lambda_noobj=0.5):
		
		self.weights_list = []
		self.biases_list = []
		self.layers_num = len(conv_layers)

		current_channels = input_shape[3]

		self.pool_type = pool_type
		self.classifier_name = classifier_name
		self.input_shape = input_shape
		self.batch_size=32
		self.dropout = dropout
		self.learning_rate = learning_rate
		self.lambda_noobj = lambda_noobj
		self.lambda_obj = lambda_obj
		self.number_of_classes = output_shape[3]-5
		self.input_shape = input_shape
		self.output_shape = output_shape
		if define_weights:
			w_name = 'kernel_weights_'
			i=0
			for layer in conv_layers:
				current_weights = self.weights_variable(shape = [layer[1],layer[1],current_channels,layer[0]],name = w_name+str(i))
				current_bias = self.bias_variable([1,layer[0]])
				current_channels = layer[0]
				self.weights_list.append(current_weights)
				self.biases_list.append(current_bias)
				i+=1
			self.pooling_layers = pooling_layers
			self.batchnorm_layers = batchnorm_layers
			self.dropout_layers = dropout_layers
			self.activation_type = activation_type

			
			_,image_width,image_hight,channel_num = self.input_shape
			self.Input = tf.placeholder(tf.float32,[None,self.input_shape[1],self.input_shape[2],self.input_shape[3]],name = "Input")
			self.Output = tf.placeholder(tf.float32,[None,self.output_shape[1],self.output_shape[2],self.output_shape[3]],name = "Output")
			self.keep_prob = tf.placeholder(tf.float32,[],name = "keep_prob")
			tf.add_to_collection('Input', self.Input)
			tf.add_to_collection('Output',self.Output)
			tf.add_to_collection('keep_prob',self.keep_prob)

		self.sess = tf.Session()
		self.lr_decaying_dict = {.42:0,.50:0,.60:0,.70:0,.80:0,.90:0,.91:0,.92:0,.93:0,.94:0,.95:0,.99:0,.996:0,.998:0,.999:0,.9992:0,.9994:0,.9996:0,.9998:0,.9999:0}
		self.classifier_name = classifier_name
		self.batch_size = 5000
		if not os.path.exists(self.classifier_name):
			os.makedirs(self.classifier_name)


	def get_classifer_name(self):
		return self.classifier_name

	def set_batch_size(self,new_batch_size):
		self.batch_size = new_batch_size

	def weights_variable(self,shape,name):
		weights =  tf.get_variable(name = name,shape = shape,initializer=  tf.contrib.layers.xavier_initializer())
		return weights


	def bias_variable(self,shape):
		initial = tf.constant(1., shape=shape,dtype = tf.float32)
		biases =  tf.Variable(initial)
		return biases

	def restore_model(self,checkpoint='75'):

		# saver = tf.train.Saver()
		saver = tf.train.import_meta_graph(self.classifier_name+"/model_"+str(checkpoint)+'.0.meta')

		saver.restore(self.sess, self.classifier_name+"/model_"+str(checkpoint)+'.0')
		self.yolo_detection_output = tf.get_collection('yolo_detection_output')[0]
		self.optimizer = tf.get_collection('optimizer')[0]
		self.accuracy = tf.get_collection('accuracy')[0]
		self.yolo_loss = tf.get_collection('yolo_loss')[0]
		self.Input = tf.get_collection('Input')[0]
		self.Output = tf.get_collection('Output')[0]
		self.keep_prob = tf.get_collection('keep_prob')[0]
		self.predicted_labels = tf.get_collection('predicted_labels')[0]


	def shuffle_data(self,features,output):

		size = features.shape[0]

		ind_list = [i for i in range(size)]
		random.shuffle(ind_list)
		shuffled_features  = features[ind_list, :,:,:]
		shuffled_output = output[ind_list,:]

		return shuffled_features,shuffled_output

	def build_yolo_loss(self,output_layer):

		obj_idxs = tf.where(self.Output[:,:,:,4] >0)
		true_output_obj = tf.gather_nd(self.Output,obj_idxs)
		predicted_output_obj= tf.gather_nd(output_layer,obj_idxs)
 
		self.obj_x_loss= tf.reduce_sum(tf.square(tf.subtract(predicted_output_obj[:,0],true_output_obj[:,0])))
		self.obj_y_loss = tf.reduce_sum(tf.square(tf.subtract(predicted_output_obj[:,1],true_output_obj[:,1])))
		self.obj_width_loss = tf.reduce_sum(tf.square(tf.subtract(predicted_output_obj[:,2],true_output_obj[:,2])))
		self.obj_height_loss  = tf.reduce_sum(tf.square(tf.subtract(predicted_output_obj[:,3],true_output_obj[:,3])))
		self.obj_coord = tf.multiply(self.lambda_obj,tf.add(tf.add(self.obj_x_loss,self.obj_y_loss),tf.add(self.obj_height_loss,self.obj_width_loss)))

		noobj_idxs = tf.where(self.Output[:,:,:,4] < 1)
		true_output_noobj = tf.gather_nd(self.Output,noobj_idxs)
		predicted_output_noobj= tf.gather_nd(output_layer,noobj_idxs)

		self.noobj_x_loss = tf.reduce_sum(tf.square(tf.subtract(predicted_output_noobj[:,0],true_output_noobj[:,0])))
		self.noobj_y_loss = tf.reduce_sum(tf.square(tf.subtract(predicted_output_noobj[:,1],true_output_noobj[:,1])))
		self.noobj_width_loss = tf.reduce_sum(tf.square(tf.subtract(predicted_output_noobj[:,2],true_output_noobj[:,2])))
		self.noobj_height_loss  = tf.reduce_sum(tf.square(tf.subtract(predicted_output_noobj[:,3],true_output_noobj[:,3])))
		self.noobj_coord = tf.multiply(self.lambda_noobj,tf.add(tf.add(self.noobj_x_loss,self.noobj_y_loss),tf.add(self.noobj_height_loss,
			self.noobj_width_loss)))

		self.probility_loss =  tf.reduce_sum(tf.square(tf.subtract(output_layer[:,:,:,4],self.Output[:,:,:,4])))

		self.classes_loss = tf.reduce_sum(tf.square(tf.subtract(output_layer[:,:,:,5:5+self.number_of_classes],
			self.Output[:,:,:,5:5+self.number_of_classes])))

		self.total_loss = tf.add(tf.add(self.obj_coord,self.noobj_coord),tf.add(self.probility_loss,self.classes_loss))
		return self.total_loss

	def calculate_accuracy(self,output_layer):

		obj_idxs = tf.where(self.Output[:,:,:,0] >0)
		true_output_obj = tf.gather_nd(self.Output,obj_idxs)
		predicted_output_obj = tf.gather_nd(output_layer,obj_idxs)

		# noobj_idxs = tf.where(self.Output[:,:,:,0] < 1)
		# true_output_noobj = tf.gather_nd(self.Output,noobj_idxs)
		# predicted_output_noobj= tf.gather_nd(output_layer,noobj_idxs)
		
		self.predicted_labels = tf.argmax(predicted_output_obj[:,5:5+self.number_of_classes],axis = 1)
		self.true_labels = tf.argmax(true_output_obj[:,5:5+self.number_of_classes],axis = 1)
		correct = tf.equal(self.predicted_labels,self.true_labels)
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
		tf.add_to_collection('predicted_labels', self.predicted_labels)

		return accuracy



	def apply_activations_on_output(self,output_layer):

		
		x= tf.reshape(tf.nn.sigmoid(output_layer[:,:,:,0]),[-1,num_of_cells,num_of_cells,1])
		y = tf.reshape(tf.nn.sigmoid(output_layer[:,:,:,1]),[-1,num_of_cells,num_of_cells,1])
		w= tf.reshape(tf.nn.sigmoid(output_layer[:,:,:,2]),[-1,num_of_cells,num_of_cells,1])
		h = tf.reshape(tf.nn.sigmoid(output_layer[:,:,:,3]),[-1,num_of_cells,num_of_cells,1])
		p = tf.reshape(tf.nn.sigmoid(output_layer[:,:,:,4]),[-1,num_of_cells,num_of_cells,1])
		c = tf.nn.softmax(output_layer[:,:,:,5:5+self.number_of_classes],axis = 3)
		output_activation = tf.concat([x,y,w,h,p,c],3)
		return output_activation



	def build_cnn_model(self):

		input_layer = self.Input
		curr_pool_layer = None
		curr_batchnorm_layer = None
		self.all_print_layers = []
		print('')
		print('')
		print('layers stacked dimensions is as following:')
		print('')
		for i in range(self.layers_num):

			curr_conv_layer = tf.nn.conv2d(input = input_layer,filter = self.weights_list[i], strides = [1,1,1,1],padding = 'SAME',name = 'conv_'+str(i)) + self.biases_list[i] 
			tf.add_to_collection('conv_'+str(i),curr_conv_layer)



			if self.activation_type == "relu":
				curr_activation_layer = tf.nn.relu(curr_conv_layer)
			elif self.activation_type == "sigmoid":
				curr_activation_layer = tf.nn.sigmoid(curr_conv_layer)
			elif self.activation_type == "tanh":
				curr_activation_layer = tf.nn.tanh(curr_conv_layer)
			elif self.activation_type == "leaky relu":
				curr_activation_layer = tf.nn.leaky_relu(curr_conv_layer)
			else:
				curr_activation_layer = curr_conv_layer

			print(curr_activation_layer)

			if self.batchnorm_layers[i] ==1 :
				curr_batchnorm_layer = tf.contrib.layers.batch_norm(inputs = curr_activation_layer)
			else:
				curr_batchnorm_layer = curr_activation_layer
			tf.add_to_collection('batchnorm_'+str(i),curr_batchnorm_layer)

			if self.pooling_layers[i] == 1:
				if self.pool_type == 'max':
					curr_pool_layer = tf.nn.max_pool(value = curr_batchnorm_layer, ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'SAME', name = 'pool_'+str(i))
				elif self.pool_type == 'avg':
					curr_pool_layer = tf.nn.avg_pool(value = curr_batchnorm_layer, ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'SAME', name = 'pool_'+str(i))

			else:
				curr_pool_layer = curr_batchnorm_layer
			tf.add_to_collection('pool_'+str(i),curr_pool_layer)

			if self.dropout_layers[i] ==1:
				curr_dropout_layer = tf.nn.dropout(curr_pool_layer,self.keep_prob)
			else:
				curr_dropout_layer = curr_pool_layer


			print_layer = tf.Print(self.weights_list[i],[self.weights_list[i]],"The Output in the tensors is")
			self.all_print_layers.append(print_layer)
			tf.add_to_collection('dropout_'+str(i),curr_dropout_layer)
			print(curr_dropout_layer)

			input_layer = curr_dropout_layer



		self.yolo_detection_output = self.apply_activations_on_output(curr_dropout_layer)
		tf.add_to_collection('yolo_detection_output',self.yolo_detection_output)
		self.yolo_loss = self.build_yolo_loss(self.yolo_detection_output)
		self.accuracy = self.calculate_accuracy(self.yolo_detection_output)
		tf.add_to_collection('accuracy',self.accuracy)
		self.optimizer= tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.yolo_loss)

		tf.add_to_collection('yolo_loss', self.yolo_loss)
		tf.add_to_collection('optimizer',self.optimizer)

		self.calculate_accuracy(curr_dropout_layer)

		self.sess.run(tf.global_variables_initializer())

	def decaying_learning_rate(self,curr_acc,saver):
		dict_len = len(self.lr_decaying_dict.items())

		lr_list= list(self.lr_decaying_dict.items())
		for i in range(dict_len):
			key = lr_list[i][0]
			value = lr_list[i][1]
			if i+1 < dict_len:
				next_key = lr_list[i+1][0]
			if key < curr_acc and next_key > curr_acc and value <10:
				save_path = saver.save(self.sess, self.classifier_name+"/model_"+str(key*100))
				self.learning_rate = self.learning_rate/1
				self.lr_decaying_dict[key]+=1
				print("Model with accuracy higher than "+str(key)+" are saved in path: %s" % save_path)
				break



	def train_classifier(self,input_images,output_encoded,testing_input,testing_output,iterations_num):


		images_num,image_width,image_hight,channel_num = input_images.shape

		print('Start training '+self.classifier_name+'...')

		number_of_batches = images_num/self.batch_size

		if images_num % self.batch_size != 0:
			number_of_batches +=1
		
		
		for it in range(iterations_num):
			#import ipdb;ipdb.set_trace()
			avg_accuracy = 0.0
			avg_loss = 0.0

			for i in range(int(number_of_batches)):
				

				start = i*self.batch_size
				end = (i+1)*self.batch_size 

				if end > input_images.shape[0]:
					end = input_images.shape[0]
				# import ipdb;ipdb.set_trace()
				input_batch = input_images[start:end]
				output_batch  = output_encoded[start:end]
				results= self.sess.run([self.optimizer,self.accuracy,self.yolo_loss],feed_dict={self.Input:input_batch,self.Output: output_batch,self.keep_prob :self.dropout})
				# layers_prints= self.sess.run(self.all_print_layers,feed_dict={self.Input:input_batch,self.Output: output_batch,self.keep_prob :self.dropout})
				avg_accuracy += results[1]
				avg_loss += results[2]

				print('the accuracy of current batch '+str(i)+'/'+str(number_of_batches)+' with accuracy: %'+str(results[1]*100))
			
			
			avg_accuracy /=number_of_batches
			avg_loss /= number_of_batches

			saver = tf.train.Saver()
			self.decaying_learning_rate(avg_accuracy,saver)

			print('the accuracy of the epoch '+str(it+1)+' is : %'+str(avg_accuracy*100.0)+' with loss = '+str(avg_loss))
			self.test_classifier(input_images = testing_input,output_classes = testing_output)
			print("------>>>>>"+str(it+1))
			input_images,output_encoded = self.shuffle_data(input_images,output_encoded)

		print('End Training '+self.classifier_name+'.')


	def retrain_classifier(self,input_images,output_classes,testing_input,testing_output,iterations_num,checkpoint='80'):

		self.restore_model(checkpoint= checkpoint)
		self.train_classifier(input_images=input_images,output_encoded=output_classes,testing_input = testing_input,testing_output = testing_output,iterations_num=iterations_num)

	def predict_labels(self,input_image,index):

		results= self.sess.run([self.correct_labels],feed_dict={self.Input:input_image,self.keep_prob:1})
		label = self.cifar100_labels[results[0][0]]

		cv2.imshow("image "+str(index)+" predict label is "+label,input_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		
	def test_classifier(self,input_images,output_classes):

		if self.batch_size > input_images.shape[0]:
			self.batch_size = input_images.shape[0]

		images_len = input_images.shape[0]
		number_of_batches = images_len/self.batch_size
		avg_loss = 0.0
		avg_accuracy = 0.0

		if images_len%self.batch_size != 0:
			number_of_batches +=1


		for i in range(int(number_of_batches)):

			start = i*self.batch_size
			end = (i+1)*self.batch_size 

			if (i+1)*self.batch_size > input_images.shape[0]:
				end = input_images.shape[0]

			input_batch =  input_images[start:end].reshape(-1,self.input_shape[1],self.input_shape[2],self.input_shape[3])
			output_batch = output_classes[start:end]

			results= self.sess.run([self.accuracy,self.yolo_loss],feed_dict={self.Input:input_batch,self.Output: output_batch,self.keep_prob:1})
			avg_accuracy += results[0]
			avg_loss += results[1]
			# print('the accuracy of current batch: %'+str(results[0]*100))

		avg_accuracy /=number_of_batches
		avg_loss /= number_of_batches
		print('the accuracy of the test data is : %'+str(avg_accuracy*100.0)+' with loss = '+str(avg_loss))


	def compare_predict_with_true(self,input_images,true_output):
		yolo_out = self.sess.run([self.yolo_detection_output],feed_dict = {self.Input:input_images })
		for i in range(yolo_out[0].shape[0]):
			import ipdb;ipdb.set_trace()
			print(yolo_out[0][i])
			print("--------------------------------------------------")
			print(true_output[i])


	def test_classifier_results(self,input_images):


		yolo_out = self.sess.run([self.yolo_detection_output],feed_dict = {self.Input:input_images })

		cells_width = self.output_shape[1]
		for i in range(input_images.shape[0]):
			image = input_images[i].reshape(-1,input_images.shape[1],input_images.shape[2],input_images.shape[3])
			yolo_out = self.sess.run([self.yolo_detection_output],feed_dict = {self.Input:image })
			yolo_out = yolo_out[0].reshape(-1,cells_width,cells_width,25)
			yolo_out = yolo_out.reshape(-1,26)
			yolo_out = filter_with_nms(yolo_out,0.6,0.5,(input_images.shape[1],input_images.shape[2]))
			for j in range(cells_width):
				for k in range(cells_width):
					coord = yolo_out[0][j][k][0:4]
					x0 = ((coord[0]*input_images.shape[1])-((coord[2]*input_images.shape[1])/2.)).astype('int')
					y0 = ((coord[1]*input_images.shape[2])-((coord[3]*input_images.shape[2])/2.)).astype('int')
					x1 = ((coord[0]*input_images.shape[1])+((coord[2]*input_images.shape[1])/2.)).astype('int')
					y1 = ((coord[1]*input_images.shape[2])+((coord[3]*input_images.shape[2])/2.)).astype('int')
					cv2.rectangle(input_images[i], (x0, y0), (x1,y1), (255,0,0), 2)

			cv2.imshow("predicted_boxes",input_images[i])
			cv2.imwrite("images_with_drawed_bounding_boxes/img_"+str(i)+".png",input_images[i])
			cv2.waitKey(0)
			cv2.destroyAllWindows()





















		























