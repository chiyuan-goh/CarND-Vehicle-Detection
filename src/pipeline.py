from classifier import *
import pickle, pdb
from multiprocessing import Pool
import time
import cv2
from scipy.ndimage.measurements import label

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def pool_helper(args):
	return insert_features(*args)

def insert_features(img, window, hog_params):
	roi = img[window[0][1]: window[1][1], window[0][0]: window[1][0]]
	if roi.shape[0] != 64:
		roi = cv2.resize(roi, (64, 64))
	fv1 = get_single_hog_features(roi, True, hog_params)
	fv2 = color_hist(roi)
	fv3 = bin_spatial(roi)

	fv = np.concatenate((fv1, fv2, fv3))
	#fv = np.concatenate((fv1, fv2, ))
	return fv 

#TODO: instead of scaling image, try to scale down window size and pixel
#per cell to see if it is the same.
def find_cars(img, cls, scaler, process_pool):
	ystart = img.shape[0] // 2 - 50
	ystop = 600

	bboxes = []	
	hog_params = get_hog_params()

	#sliding window approach
	#scale = [0.5, 0.8, 1.0, 2.0, 2.5, 3, 3.5]
	scale = [ 1., 1.5, 2.]
	window_size = 64
	cells_per_step = 2
	ppc = hog_params['pixels_per_cell'][0]
	cpb = hog_params['cells_per_block'][0]
	n_features = scaler.mean_.shape[0]

	t0 = time.time()
	fp = 0

	all_windows = []

	for s in scale:
		wsize = int(window_size * s)
		all_windows += slide_window(img.copy(), y_start_stop = [ystart, ystop], xy_window = (wsize, wsize), xy_overlap = (.75, .75))

	fvs = np.zeros((len(all_windows), n_features))

	# for idx,window in enumerate(all_windows):
	# 	roi = img[window[0][1]: window[1][1], window[0][0]: window[1][0]]
	# 	roi = cv2.resize(roi, (64, 64))
	# 	fv1 = get_single_hog_features(roi, True, hog_params)
	# 	fv2 = color_hist(roi)
	# 	fv3 = bin_spatial(roi)

	# 	fv = np.concatenate((fv1, fv2, fv3))
	# 	# fv = np.concatenate((fv1, fv2, ))
	# 	fvs[idx, :] = fv

	job_args = [(img, w, hog_params) for w in all_windows]
	results = process_pool.map(pool_helper, job_args) 
	for i, result in enumerate(results):
		fvs[i, :] = result

	scaled_fvs = scaler.transform(fvs)
	#predictions = cls.predict(scaled_fvs)
	scores = cls.decision_function(scaled_fvs)

	#pos_indices = np.nonzero(predictions)[0]
	pos_indices = np.where(scores > 0)[0]
	pos_scores = scores[pos_indices]
	#dist = cls.decision_function(scaled_fvs)
	#m = dist.max()
	#pos_indices = np.where(dist >= m * 0.7)[0]
	for pos_idx in pos_indices:
		bboxes.append(all_windows[pos_idx])

	#print('total ', len(bboxes), " fp" , fp)
	t1 = time.time()
	print("time taken: %.3f"%(t1-t0))
	return bboxes, pos_scores

def heatmap_threshold(heatmap):
	threshold = 3
	heatmap[heatmap < threshold] = 0
	return heatmap

def draw_tracked_boxes(img, boxes):
	box_img = img.copy()

	for box in boxes:
		cv2.rectangle(box_img, box[0], box[1], (255,0,0), 6)

	return box_img

#draw found boxes
def draw_label_bboxes(img, labels, n_features):
	box_img = img.copy()

	for i in range(1, n_features + 1):
		ys, xs = np.where(labels == i)

		topleft = (xs.min(), ys.min())
		btmright = (xs.max(), ys.max())

		cv2.rectangle(box_img, topleft, btmright, (255,0,0), 6)

	return box_img

def update_centroids(centroids, labels, n_features, ttl, skipped, dist_threshold = 15):
	results = []

	#track ttl
	found = np.zeros(centroids.shape[0], dtype=bool)  #whether past tracks are linked in this frame
	indices = np.array([-1 for i in range(n_features)]) #window is assigned to which track
	tmp_centroids = []

	#each new found box
	for i in range(1, n_features + 1):
		ys, xs = np.where(labels == i)
		
		center = np.array([(xs.max() - xs.min())/2, (ys.max() - ys.min())/2])
		assigned = False #whether window is assigned to any track

		for j, centroid in enumerate(centroids):
			if np.linalg.norm(centroid - center) < dist_threshold:
				#if close enough to a track, shift track center and assign track to window 
				centroids[j, :] = (centroid + center)/ 2
				found[j] = 1
				assigned = True
				indices[i-1] = j
				print("linked!")

		#if not assigned to any track, create new track		
		if not assigned:
			tmp_centroids.append([center[0], center[1]])


	ttl[found] += 1 #increment linked tracks ttl by 1
	pos_frames = np.where(ttl >= 4)[0] #those tracks ttl >=3 is identified as positive

	for i,idx in enumerate(indices): #show those windows whose tracks are positive
		if idx in pos_frames:
			ys, xs = np.where(labels == i+1)
			topleft = (xs.min(), ys.min())
			btmright = (xs.max(), ys.max())
			results.append( (topleft, btmright) )

	print("before:" , ttl.shape[0])

	sthres = 4
	skipped[found] = 0
	skipped[~found] += 1
	centroids = centroids[(skipped < sthres)]
	ttl = ttl[skipped < sthres]
	skipped = skipped[skipped < sthres]

	if len(tmp_centroids) != 0:
		if centroids.shape[0] == 0:
			centroids = np.array(tmp_centroids).copy()
		else:	
			try:
				centroids = np.vstack((centroids, np.array(tmp_centroids))) 
			except:
				pdb.set_trace()
		ttl = np.hstack((ttl, np.ones(len(tmp_centroids))) )
		skipped = np.hstack( (skipped, np.zeros(len(tmp_centroids))))
	print("after:", ttl.shape[0])
	return results, centroids, ttl, skipped


def process_frame(img, classifier, scaler, process_pool, track_centroids, veh_ttl, skipped_tracks, visualize = False, thres = 1.):
	bboxes, scores = find_cars(img, classifier, scaler, process_pool)

	heatmap = np.zeros_like(img[:,:,0])
	for i, box in enumerate(bboxes):
		score = scores[i]
		if score > thres:
			heatmap[box[0][1]: box[1][1], box[0][0]: box[1][0]] += 1

	heatmap_threshold(heatmap)
	labels, n_features = label(heatmap)

	track_bboxes, track_centroids, veh_ttl, skipped_tracks = update_centroids(track_centroids, labels, n_features, veh_ttl, skipped_tracks)
	print(len(track_bboxes), "cars found!")

	bgr_image = cv2.cvtColor(img, cv2.COLOR_LUV2RGB)
	#new_img = draw_label_bboxes(bgr_image, labels, n_features)
	new_img = draw_tracked_boxes(bgr_image, track_bboxes)

	if visualize:
		max_score = scores.max()
		raw_img = bgr_image.copy()
		for i, box in enumerate(bboxes):
			score = scores[i]
			if score > thres:
				color = np.int(score/max_score * 255)
				cv2.rectangle(raw_img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (color,0,0), 6)

		return new_img, raw_img

	else:
		return new_img, track_centroids, veh_ttl, skipped_tracks
	
def main(video, classifier, scaler):
	process_pool = Pool(8)
	for frame in frames:
		pframe = process_frame(frame, classifier, scaler, process_pool)



if __name__ == '__main__':
	dat_filename = '../models/model_liblinear_awesome2.dat'
	dat_dict = pickle.load(open(dat_filename, "rb"))
	scaler = dat_dict['scaler']
	classifier = dat_dict['cls']

	track_centroids = np.array([])
	veh_ttl = np.array([])

	process_pool = Pool(4)

	img = prepare_img(cv2.imread("../test_images/test3.jpg"))
	test_img = img.copy()

	centroids = []
	centroid_count = []

	ttl = np.array([])
	skipped = np.array([])
	centroids = np.array([])

	frame, raw = process_frame(test_img, classifier, scaler, process_pool, ttl, skipped, centroids, True)
	cv2.imshow("image", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) )
	cv2.imwrite("../output_images/t3_results.png", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) )
	cv2.waitKey(0)



	#cv2.imshow("image", frame)
	#cv2.waitKey(0)

