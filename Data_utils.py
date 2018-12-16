import numpy as np
import xml.etree.ElementTree as ET
import os
from xml.dom import minidom
import xmltodict
from scipy import sparse
import cv2

hashed_labels = {}
hashed_indeces = {}
num_of_cells = 9
resized_shape = (144,144)
imgs_names_list = []
def load_training_xmls(folder_path):
    """
    this function responsible for loading the training data from a file path 
    """
    i=0
    all_imgs = []
    for filename in os.listdir(folder_path):
       curr_img = {}
       curr_img['objects'] = []
       if not filename.endswith('.xml'): continue
       fullname = os.path.join(folder_path, filename)
       imgs_names_list.append(filename.split('.')[0])
       f = open(fullname)
       img_dict = xmltodict.parse(f.read())
       if 'annotation' not in img_dict.keys():continue
       if 'object' not in img_dict['annotation']:continue
       curr_img['width'] = float(img_dict['annotation']['size']['width'])
       curr_img['height'] = float(img_dict['annotation']['size']['height'])
       objects = img_dict['annotation']['object']
       curr_obj = {}
       if type(objects)==list:
          for obj in objects:
              curr_obj['label'] = obj['name']
              box = []
              box.append(float(obj['bndbox']['xmin']))
              box.append(float(obj['bndbox']['ymin']))
              box.append(float(obj['bndbox']['xmax']))
              box.append(float(obj['bndbox']['ymax']))
              curr_obj['box'] = box
              curr_img['objects'].append(curr_obj)
              curr_obj = {}
       else:
          curr_obj['label'] = objects['name']
          box = []
          box.append(float(objects['bndbox']['xmin']))
          box.append(float(objects['bndbox']['ymin']))
          box.append(float(objects['bndbox']['xmax']))
          box.append(float(objects['bndbox']['ymax']))
          curr_obj['box']=box
          curr_img['objects'].append(curr_obj) 
       all_imgs.append(curr_img)
       i+=1
    return all_imgs

def map_labels_to_indeces(imgs_boxes):
   
    it = 0
    max_number_of_boxes = 0
    for img_boxes in imgs_boxes:
        boxes_num=0
        for box in img_boxes['objects']:
            boxes_num+=1
            if box['label'] not in hashed_labels:
               hashed_labels[box['label']] = it
               hashed_indeces[it] = box['label']
               it+=1
        if boxes_num > max_number_of_boxes:
            max_number_of_boxes= boxes_num
    return max_number_of_boxes


def convert_np_to_sparse(out_np):

    sparse_out_list = []
    for i in range(out_np.shape[0]):
        curr_row = sparse.csr_matrix(out_np[i])
        sparse_out_list.append(curr_row)
    return np.array(sparse_out_list)
             
def load_training_images(folder_path):

    all_images = []
    i=0
    for img_name in imgs_names_list:
       filename = img_name+'.jpg' 
       fullname = os.path.join(folder_path, filename)
       image  = cv2.imread(fullname,1)
       new_image = cv2.resize(image,resized_shape)
       all_images.append(new_image)

    imgs_np = np.array(all_images)
    write_nparray_to_file(imgs_np,"imgs.npy")
    return imgs_np

def calculate_nearest_cell(box,img_size):

  center_x = (box['box'][0] + box['box'][2])/2.
  center_y = (box['box'][1] + box['box'][3])/2.
  x_cell_size = img_size[0]/num_of_cells
  y_cell_size = img_size[1]/num_of_cells
  xcoord_cell = int(center_x/x_cell_size)
  ycoord_cell = int(center_y/y_cell_size)
  return (xcoord_cell,ycoord_cell)

def set_yolo_output_param(box,img_size):

  center_x = (box['box'][0] + box['box'][2])/2.
  center_y = (box['box'][1] + box['box'][3])/2.

  center_x = center_x/img_size[0]
  center_y = center_y/img_size[1]
  width = np.abs(box['box'][0] - box['box'][2])/img_size[0]
  height = np.abs(box['box'][1] - box['box'][3])/img_size[1]
  return center_x,center_y,width,height

def build_yolo_output_np(imgs):
  
  max_boxes = map_labels_to_indeces(imgs)
  number_of_classes = len(hashed_indeces)
  ht = np.zeros(shape = (len(imgs),num_of_cells,num_of_cells))
  yolo_output_np = np.zeros(shape = (len(imgs),num_of_cells,num_of_cells,number_of_classes+5))
  i=0
  for img in imgs:
    for box in img['objects']:
      cell_coord = calculate_nearest_cell(box,(img['width'],img['height']))
      if ht[i][cell_coord[0]][cell_coord[1]] == 1:
        continue
      cx,cy,w,h = set_yolo_output_param(box,(resized_shape[0],resized_shape[1]))
      c = np.zeros(number_of_classes)
      c[hashed_labels[box['label']]]=1
      value = np.concatenate((np.array([cx,cy,w,h,1]),c),axis =0)
      yolo_output_np[i][cell_coord[0]][cell_coord[1]] = value
    i+=1
  write_nparray_to_file(yolo_output_np,"out.npy")
  return yolo_output_np


def show_real_boxes(imgs,out):

  for i in range(100):
    idx = np.random.randint(0,100,size = [1])
    for box in out[idx[0]]['objects']:
      x0 = int(imgs.shape[1]*(box['box'][0]/out[idx[0]]['width']))
      y0 = int(imgs.shape[2]*(box['box'][1]/out[idx[0]]['height']))
      x1 = int(imgs.shape[1]*(box['box'][2]/out[idx[0]]['width']))
      y1 = int(imgs.shape[2]*(box['box'][3]/out[idx[0]]['height']))
      cv2.rectangle(imgs[idx[0]], (x0,y0), (x1,y1), (255,0,0), 2)
      cv2.imshow("image with real boxes",imgs[idx[0]])
      cv2.waitKey(0)

def write_nparray_to_file(np_arr,filename = "imgs"):
  np.save(filename,[np_arr])

def load_nparray_from_file(filename = "imgs"):
  np_array = np.load(filename)
  return np_array


def apply_jitter(jitter,imgs):

  width = int((1.- jitter)*imgs.shape[1])
  height = int((1. -jitter)* imgs.shape[2])

  width_range = imgs.shape[1] - width
  height_range = imgs.shape[2] - height

  random_x = np.random.randint(0,width_range,[1])[0]
  random_y = np.random.randint(0,height_range,[1])[0]
  cropped_imgs = imgs[:,random_x:random_x+width,random_y:random_y+height,0:3]

  noised_imgs = np.random.randint(0,255,size= imgs.shape,dtype = "uint8")
 
  noised_imgs[:,random_x:random_x+width,random_y:random_y+height] = cropped_imgs

  noised_imgs = noised_imgs.astype(np.uint8)
  return noised_imgs







