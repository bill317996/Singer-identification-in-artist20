import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import os,time
import model
import h5py
import itertools
import utility
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import argparse
import pickle
import librosa


def predict(song_path, gid=0, classes_num=20, model_path='./CRNN2D_elu2_model_state_dict'):

    start_time = time.time()

    #####################################

    print('Loading pretrain model ...')

    Classifier = model.CRNN2D_elu2(288,classes_num)
    Classifier.float()
    Classifier.cuda()

    Classifier.load_state_dict(torch.load(model_path, map_location={'cuda:0':'cuda:{}'.format(gid)})['Classifier_state_dict'])

    Classifier.eval()

    #####################################

    sr=16000
    n_mels=128
    n_fft=2048
    hop_length=512

    y, sr = librosa.load(song_path, sr=sr)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                            n_fft=n_fft,
                                            hop_length=hop_length)
    log_S = librosa.core.amplitude_to_db(S, ref=1.0)

    print('input_spec: ',log_S.shape)

    length = 157

    X = []

    slices = int(log_S.shape[1] / length)
    for j in range(slices - 1):
        X.append(log_S[:, length * j:length * (j + 1)])
           
    X = torch.from_numpy(np.array(X)).float()
    X = X.cuda()
    h = torch.randn(1, X.size(0), 32).cuda()

    #####################################
    
    pred_y, emb = Classifier(X, h)
    print('--- Done ----')
    print('output_pred: ',pred_y.shape)
    print('output_emb: ',emb.shape)
    return pred_y, emb

    
def parser():
    
    p = argparse.ArgumentParser()

    p.add_argument('song_path', type=str)
    p.add_argument('-gid', '--gpu_index', type=int, default=0)

    return p.parse_args()
if __name__ == '__main__':

    args = parser()

    song_path = args.song_path
    gid = args.gpu_index
    with torch.cuda.device(gid):
        pred_y, emb = predict(song_path, gid=gid)
    
        
