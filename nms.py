import numpy as np


def filter_with_nms(net_out,overthresh,confthresh,image_shape):

	filtered_idxs = np.where(net_out[:,4] > confthresh)
	filtered_bndbox = net_out[filtered_idxs]
	filtered_bndbox = filtered_bndbox.reshape(-1,26)
	x_center = net_out[:,0]*image_shape[0]
	y_center = net_out[:,1]*image_shape[1]
	width = net_out[:,2]*image_shape[0]
	height = net_out[:,3]*image_shape[1]

	x0 = x_center - (width/2)
	x1 = x_center + (width/2)
	y0 = y_center - (height/2)
	y1 = y_center + (height/2)
	area = np.abs(x1-x0)*np.abs(y1-y0)
	idxs = np.argsort(y1)
	filtered = []
	while len(idxs) >0:
		last = len(idxs)-1
		i = idxs[-1]
		filtered.append(i)

		xmin = np.maximum(x0[last],x0[idxs[:last]])
		ymin = np.maximum(y0[last],y0[idxs[:last]])
		xmax = np.minimum(x1[last],x1[idxs[:last]])
		ymax = np.minimum(y1[last],y1[idxs[:last]])

		w = np.maximum(0,(xmax-xmin+1))
		h = np.maximum(0,(ymax-ymin+1))
		wh = w*h
		overlap = wh/area[idxs[:last]]
		import ipdb;ipdb.set_trace()
		query = np.concatenate([last],np.where(overlap > overthresh))
		idxs = np.delete(idxs,query)

	return net_out[filtered]




