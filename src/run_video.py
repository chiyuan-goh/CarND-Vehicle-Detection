from pipeline import *
from moviepy.editor import VideoFileClip
import cv2, pdb, multiprocessing

def process_video_frame(vid_frame):
	global track_centroids, veh_ttl, tracks_skipped

	bgr_image = prepare_img(vid_frame, source="RGB")
	#bgr_image = cv2.GaussianBlur(bgr_image,(5,5),0)

	#output, raw = process_frame(bgr_image, classifier, scaler, p, visualize = True)
	#pdb.set_trace()
	output, track_centroids, veh_ttl, tracks_skipped = process_frame(bgr_image, classifier, scaler, p, track_centroids, veh_ttl, tracks_skipped, visualize = False)
	return output

def run_video(video_filename, output_vid):
	clip = VideoFileClip(video_filename).subclip(10, 14)
	white_clip = clip.fl_image(process_video_frame)
	white_clip.write_videofile(output_vid, audio=False)


if __name__ == '__main__':
	dat_filename = '../models/model_liblinear_awesome2.dat'
	dat_dict = pickle.load(open(dat_filename, "rb"))
	scaler = dat_dict['scaler']
	classifier = dat_dict['cls']

	p = multiprocessing.Pool(4)
	track_centroids = np.array([])
	veh_ttl = np.array([])
	tracks_skipped = np.array([])

	video_filename = '../project_video.mp4'
	output = '../project_output.mp4'

	# video_filename = '../test_video.mp4'
	# output = '../output_test.mp4'
	run_video(video_filename, output)