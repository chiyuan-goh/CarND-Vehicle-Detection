import os
import cv2
import pdb
import numpy as np
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn import svm
import pickle
import random
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt




def get_dataset():
    pos_dir = '../data/vehicles'
    neg_dir = '../data/non-vehicles'

    neg1_dir = os.path.join(neg_dir, "Extras")
    neg2_dir = os.path.join(neg_dir, "GTI")
    neg_files1 = [os.path.join(neg1_dir, f) for f in os.listdir(neg1_dir) if f.endswith("png") or f.endswith("jpg")]
    neg_files2 = [os.path.join(neg2_dir, f) for f in os.listdir(neg2_dir) if f.endswith("png") or f.endswith("jpg")]
    neg_files = neg_files1 + neg_files2

    pos1_dir = os.path.join(pos_dir, "KITTI_extracted")
    pos2_dir = os.path.join(pos_dir, "custom")
    pos3_dir = os.path.join(pos_dir, "udacity_sample")

    pos_files1 = [os.path.join(pos1_dir, f) for f in os.listdir(pos1_dir) if f.endswith("png") or f.endswith("jpg")]
    pos_files2 = [os.path.join(pos2_dir, f) for f in os.listdir(pos2_dir) if f.endswith("png") or f.endswith("jpg")]
    pos_files3 = [os.path.join(pos3_dir, f) for f in os.listdir(pos3_dir) if f.endswith("png") or f.endswith("jpg")]
    pos_files = pos_files1 + pos_files2 + pos_files3

    print("There are %s positive and %d negative instances" %(len(pos_files),len(neg_files)) )

    sample_neg = np.random.choice(neg_files, len(pos_files), replace=False)
    neg_X = get_features(sample_neg)
    print("neg X shape ", neg_X.shape)

    pos_X = get_features(pos_files)
    print("pos X shape ", pos_X.shape)  

    X = np.vstack( (neg_X, pos_X) )
    Y = np.concatenate( (np.zeros((neg_X.shape[0],)), np.ones((pos_X.shape[0],))) )
    
    X, scaler = scale_features(X)

    return X, Y, scaler

def get_hog_params():
    return {
        'pixels_per_cell': (16,16),
        'cells_per_block': (2,2),
        'orientations': 9,
        'transform_sqrt': False
    }

def prepare_img(img, source="BGR"):
    if source == "BGR":
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        #return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif source == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        #return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def bin_spatial(img, size=(32, 32)):
    resized = cv2.resize(img, size)
    color1 = resized[:, :, 0].ravel()
    color2 = resized[:, :, 1].ravel()
    color3 = resized[:, :, 2].ravel()
    # color1 = cv2.resize(img[:,:,0], size).ravel()
    # color2 = cv2.resize(img[:,:,1], size).ravel()
    # color3 = cv2.resize(img[:,:,2], size).ravel()
    # pdb.set_trace()
    return np.hstack((color1, color2, color3))

def get_single_hog_features(img, feature_vector, hog_params):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #new_color = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    fv1 = hog(img[:,:,0], 
        feature_vector=feature_vector,
        visualise=False,
        **hog_params)
    
    fv2 = hog(img[:,:,1], 
        feature_vector=feature_vector,
        visualise=False,
        **hog_params)
    
    fv3 = hog(img[:,:,2], 
        feature_vector=feature_vector,
        visualise=False,
        **hog_params)

    fv = np.hstack( (fv1, fv2, fv3))

    return fv

def get_hog_features(imgs):
    #hog parameters
    hog_params = get_hog_params()

    hog_features = None
    for i,img in enumerate(imgs):
        feature_vector = get_single_hog_features(img, True, hog_params)
        if hog_features is None:
            hog_features = np.zeros((len(imgs), feature_vector.shape[0]))

        hog_features[i, :] = feature_vector
    return hog_features

def get_hist_features(imgs):
    hist_features = None

    for i,img in enumerate(imgs):
        fv = color_hist(img)

        if hist_features is None:        
            hist_features = np.zeros((len(imgs), fv.shape[0]))

        hist_features[i, :] = fv

    return hist_features

def get_spatial_features(imgs):
    s_features = np.zeros((len(imgs), 32 * 32 * 3))

    for i,img in enumerate(imgs):
        fv = bin_spatial(img)
        s_features[i, :] = fv

    return s_features   

#generate features for classifier
def get_features(img_files):
    features = []
    #pdb.set_trace()
    imgs = [prepare_img(cv2.imread(img)) for img in img_files]

    X1 = get_hog_features(imgs)
    X2 = get_hist_features(imgs)
    X3 = get_spatial_features(imgs)

    X = np.hstack((X1, X2, X3))
    #X = np.hstack((X1, X2))
    return X

def scale_features(X):
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return X, scaler        

def get_cls(X, y):
    #return svm.SVC() 
    return svm.LinearSVC(C=.1, dual=False)

def train_cls(X, y):
    cls = get_cls(None, None)
    return cls.fit(X, y)

def evaluate_cls():
    X, Y, scaler = get_dataset()

    X, Y = shuffle(X,Y)
    print("calculating score...")
    
    # scores = cross_val_score(get_cls(X,Y), X,Y, scoring='f1', verbose=2, n_jobs=-1)
    # print('score: ', scores)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    classifier = train_cls(X_train, Y_train)
    test_predictions = classifier.predict(X_test)

    score = f1_score(Y_test, test_predictions)
    print("f1 score:", score) 

    conf = confusion_matrix(Y_test, test_predictions)
    print("confusion matrix:\n", conf)

def do_gridsearch():
    X, Y, scaler = get_dataset()

    parameters = {'C': [0.1, 1, 10, 100], 'dual': [False, True]}
    classifier = GridSearchCV(get_cls(None, None), param_grid = parameters,n_jobs = -1, scoring = 'f1', verbose = 2)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    classifier = classifier.fit(X_train, Y_train)

    test_predictions = classifier.predict(X_test)

    score = f1_score(Y_test, test_predictions)
    print("f1 score:", score) 

    conf = confusion_matrix(Y_test, test_predictions)
    print("confusion matrix:\n", conf)

def do_learning_curve(train_sizes=np.linspace(.1, 1.0, 10), ylim=None):
    estimator = get_cls(None, None)
    X, Y, scaler = get_dataset()
    X, Y = shuffle(X,Y)

    plt.figure()
    plt.title("Learning curve of linearSVC C=0.1")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, Y, cv=None, n_jobs=-1, train_sizes=train_sizes, verbose=2)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt


def mkclassifier(filename):
    X, Y, scaler = get_dataset()
    print("training classifier")
    classifier = train_cls(X, Y)
    model_dict = {'scaler': scaler, 'cls': classifier}
    pickle.dump(model_dict, open(filename, 'wb'))
    print("saved model to ", filename)

if __name__ == '__main__':
    evaluate_cls()
    #do_gridsearch()
    mkclassifier("../models/model_liblinear_ycrcb.dat")
    #do_learning_curve()

