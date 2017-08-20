from classifier import *
import pickle, pdb
import time
import cv2
from scipy.ndimage.measurements import label


#TODO: instead of scaling image, try to scale down window size and pixel
#per cell to see if it is the same.
def find_cars(img, cls, scaler):
	ystart = img.shape[0] // 2

	scan_img = img[ystart:, :, :]
	bboxes = []
	
	hog_params = get_hog_params()

	#sliding window approach
	#scale = [0.5, 0.8, 1.0, 2.0, 2.5, 3, 3.5]
	scale = [1., 1.5, 2.]
	window_size = 64
	cells_per_step = 2
	ppc = hog_params['pixels_per_cell'][0]
	cpb = hog_params['cells_per_block'][0]
	n_features = scaler.mean_.shape[0]

	t0 = time.time()
	fp = 0

	for s in scale:
		print("processing scale ", s)
		h,w,_ = scan_img.shape
		if s != 1:
			scaled_img = cv2.resize(scan_img, (int(w/s), int(h/s)))
		else:
			scaled_img = scan_img	

		#returns in block * block * cell * cell * orientation
		hog_array =  get_single_hog_features(scaled_img, False, hog_params)

		#how many blocks there are. 
		#note that blocks are per cell overlapping, hence there are as many
		#blocks as there are cells, minus the edge cases where there are a 
		#full block, hence -cpb + 1
		nx_blocks = (scaled_img.shape[1] // ppc) - cpb + 1
		ny_blocks = (scaled_img.shape[0] // ppc) - cpb + 1

		blocks_per_window = (window_size // ppc) - cpb + 1

		#as above, this time edge case is in terms of blocks per window
		#//cells_per_step akin to how many blocks to step.
		nx_steps = (nx_blocks - blocks_per_window) // cells_per_step
		ny_steps = (ny_blocks - blocks_per_window) // cells_per_step

		window_X = np.zeros((nx_steps * ny_steps, n_features))

		for nx in range(nx_steps):
			xidx = nx * cells_per_step

			for ny in range(ny_steps):
				yidx = ny * cells_per_step
				fv = hog_array[yidx:yidx + blocks_per_window, xidx:xidx + blocks_per_window, :, :, :].ravel()#.reshape((1, -1))
				window_X[ny * nx_steps + nx, : ] = fv

		scaled_fvs = scaler.transform(window_X)
		predictions = cls.predict(scaled_fvs)

		indices = np.nonzero(predictions)[0]
		for i in indices:
			ny = i // nx_steps
			nx = i % nx_steps

			topleft = np.array([nx * cells_per_step * ppc, 
						ny * cells_per_step * ppc])
			topleft = topleft * s
			topleft[1] = topleft[1] + ystart
			btmright = np.array([nx * cells_per_step * ppc + window_size,
						ny * cells_per_step * ppc + window_size])
			btmright = btmright * s
			btmright[1] = btmright[1] + ystart
			bboxes.append( (topleft.astype(int), btmright.astype(int)) )
			try:
				topleft = topleft.astype(int)
				btmright = btmright.astype(int)
				test_img = img[topleft[1]:btmright[1],  topleft[0]:btmright[0], :]
			except:
				pdb.set_trace()
			test_img = cv2.resize( test_img, (64, 64) )
			fv = get_single_hog_features(test_img, True, hog_params).reshape((1,-1))
			fv = scaler.transform(fv)

			if s == 1 and cls.predict(fv) == 0:
				fp += 1

				#TODO: add other feature vector
				# scaled_fv = scaler.transform(fv)
				# prediction = cls.predict(scaled_fv)

				# if prediction == 1:
				# 	#calculate bounding box
				# 	topleft = np.array([nx * cells_per_step * ppc, 
				# 		ny * cells_per_step * ppc])
				# 	topleft = topleft * s
				# 	topleft[1] = topleft[1] + ystart
				# 	btmright = np.array([nx * cells_per_step * ppc + window_size,
				# 		ny * cells_per_step * ppc + window_size
				# 		])
				# 	btmright = btmright * s
				# 	btmright[1] = btmright[1] + ystart
				# 	bboxes.append( (topleft.astype(int), btmright.astype(int)) )
	print('total ', len(bboxes), " fp" , fp)
	t1 = time.time()
	print("time taken: %.3f"%(t1-t0))
	return bboxes

def heatmap_threshold(heatmap):
	threshold = 4
	heatmap[heatmap < threshold] = 0
	return heatmap

#draw found boxes
def draw_label_bboxes(img, labels, n_features):
	box_img = img.copy()

	for i in range(1, n_features + 1):
		ys, xs = np.where(labels == i)

		topleft = (xs.min(), ys.min())
		btmright = (xs.max(), ys.max())

		cv2.rectangle(box_img, topleft, btmright, (255,0,0), 6)

	return box_img


def process_frame(img, classifier, scaler, visualize = False):
	bboxes = find_cars(img, classifier, scaler)

	heatmap = np.zeros_like(img[:,:,0])
	for box in bboxes:
		heatmap[box[0][1]: box[1][1], box[0][0]: box[1][0]] += 1

	heatmap_threshold(heatmap)
	labels, n_features = label(heatmap)
	print(n_features, "cars found!")

	new_img = draw_label_bboxes(img, labels, n_features)

	if visualize:
		raw_img = img.copy()
		for box in bboxes:
			cv2.rectangle(raw_img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (255,0,0), 6)
		return new_img, raw_img

	else:
		return new_img
	
def main(video, classifier, scaler):
	for frame in frames:
		pframe = process_frame(frame, classifier, scaler)



if __name__ == '__main__':
	dat_filename = '../models/model.dat'
	dat_dict = pickle.load(open(dat_filename, "rb"))
	scaler = dat_dict['scaler']
	classifier = dat_dict['cls']

	test_img = cv2.imread("../test_images/test5.jpg")
	frame, raw = process_frame(test_img, classifier, scaler, True)
	cv2.imshow("image", raw)
	cv2.waitKey(0)

