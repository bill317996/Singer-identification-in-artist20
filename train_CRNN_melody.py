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

class Dataset_4(Data.Dataset):
    def __init__(self, data_tensor, target_tensor1, target_tensor2, target_tensor3):
        assert data_tensor.size(0) == target_tensor1.size(0)
        self.data_tensor = data_tensor
        self.target_tensor1 = target_tensor1
        self.target_tensor2 = target_tensor2
        self.target_tensor3 = target_tensor3

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor1[index], self.target_tensor2[index], self.target_tensor3[index]

    def __len__(self):
        return self.data_tensor.size(0)

class Dataset_3(Data.Dataset):
    def __init__(self, data_tensor, target_tensor1, target_tensor2):
        assert data_tensor.size(0) == target_tensor1.size(0)
        self.data_tensor = data_tensor
        self.target_tensor1 = target_tensor1
        self.target_tensor2 = target_tensor2

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor1[index], self.target_tensor2[index]

    def __len__(self):
        return self.data_tensor.size(0)

class Dataset_2(Data.Dataset):
    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

class Dataset_1(Data.Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


# def main(gid=0, bgm_shuffle=True, early_stop=True, bs=100, learn_rate=0.001, album_split=True, random_state=0):
def main(classes_num=20, gid=0, random_state=0, \
            bs=100, learn_rate=0.0001, \
            val_num=1, stop_num=20,
            origin=True, vocal=True, remix=True,
            arc=False, focal=False):

    start_time = time.time()

    save_folder = '../save/'+str(random_state)+'/'

    if origin and vocal and remix:
        save_folder = save_folder + '/all/'
    elif origin:
        save_folder = save_folder + '/ori/'
    elif vocal:
        save_folder = save_folder + '/voc/'
    elif remix:
        save_folder = save_folder + '/remix/'

    if arc:
        focal=True
        save_folder = save_folder+'/arc/'
    elif focal:
        save_folder = save_folder+'/focal/'
    else:
        save_folder = save_folder+'/base/'

    if not os.path.exists(save_folder+'/model/'):
        os.makedirs(save_folder+'/model/')
    if not os.path.exists(save_folder+'/result/'):
        os.makedirs(save_folder+'/result/')

    epoch_num = 1000

    print('Loading pretrain model ...')

    Classifier = model.CRNN2D_melody_bgru()
    Classifier.float()
    Classifier.cuda()
    Classifier.train()

    print('Loading training data ...')

    artist_folder=f'/home/bill317996/189/homes/kevinco27/dataset/artist20_mix'
    song_folder=f'/home/bill317996/189/homes/kevinco27/ICASSP2020_meledy_extraction/music-artist-classification-crnn/song_data_mix'
    voc_folder=f'/home/bill317996/189/homes/kevinco27/ICASSP2020_meledy_extraction/music-artist-classification-crnn/song_data_open_unmix_vocal_2'
    bgm_folder = f'/home/bill317996/189/homes/kevinco27/ICASSP2020_meledy_extraction/music-artist-classification-crnn/song_data_open_unmix_kala'
    mel_folder = f'/home/bill317996/189/homes/kevinco27/ICASSP2020_meledy_extraction/music-artist-classification-crnn/song_data_open_unmix_melody'

    # random_states = [0,21,42]
    
    nb_classes = 20
    slice_length = 157


    Y_train, X_train, S_train, V_train, B_train, M_train,\
    Y_test, X_test, S_test, V_test, B_test, M_test,\
    Y_val, X_val, S_val, V_val, B_val, M_val = \
        utility.load_dataset_album_split_dam(song_folder_name=song_folder,
                                         artist_folder=artist_folder,
                                         voc_song_folder=voc_folder,
                                         bgm_song_folder=bgm_folder,
                                         mel_song_folder=mel_folder,
                                         nb_classes=nb_classes,
                                         random_state=random_state)

    print("Loaded and split dataset. Slicing songs...")

    # Create slices out of the songs
    X_train, Y_train, S_train, V_train, B_train, M_train = utility.slice_songs_dam(X_train, Y_train, S_train, V_train, B_train, M_train,
                                                    length=slice_length)
    X_val, Y_val, S_val, V_val, B_val, M_val = utility.slice_songs_dam(X_val, Y_val, S_val, V_val, B_val, M_val,
                                              length=slice_length)
    X_test, Y_test, S_test, V_test, B_test, M_test = utility.slice_songs_dam(X_test, Y_test, S_test, V_test, B_test, M_test,
                                                 length=slice_length)

    print("Training set label counts:", np.unique(Y_train, return_counts=True))

    # # Encode the target vectors into one-hot encoded vectors
    Y_train, le, enc = utility.encode_labels(Y_train)
    Y_test, le, enc = utility.encode_labels(Y_test, le, enc)
    Y_val, le, enc = utility.encode_labels(Y_val, le, enc)

    Y_train = Y_train[:,0]
    Y_test = Y_test[:,0]
    Y_val = Y_val[:,0]

    # debug

    # Y_train, X_train, S_train, V_train, B_train,\
    #     Y_test, X_test, S_test, V_test, B_test,\
    #     Y_val, X_val, S_val, V_val, B_val = \
    #     np.zeros(11437), np.zeros((11437, 128, 157)), np.zeros(11437), np.zeros((11437, 128, 157)), np.zeros((11437, 128, 157)), \
    #     np.zeros(11437), np.zeros((11437, 128, 157)), np.zeros(11437), np.zeros((11437, 128, 157)), np.zeros((11437, 128, 157)), \
    #     np.zeros(11437), np.zeros((11437, 128, 157)), np.zeros(11437), np.zeros((11437, 128, 157)), np.zeros((11437, 128, 157)) 

    print(X_train.shape, Y_train.shape, S_train.shape, V_train.shape, B_train.shape, M_train.shape)
    print(X_val.shape, Y_val.shape, S_val.shape, V_val.shape, B_val.shape, M_val.shape)
    print(X_test.shape, Y_test.shape, S_test.shape, V_test.shape, B_test.shape, M_test.shape)

    #####################################
    # numpy to tensor to data_loader
    # train

    
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).long()
    V_train = torch.from_numpy(V_train).float()
    B_train = torch.from_numpy(B_train).float()
    M_train = torch.from_numpy(M_train).float()

    if origin:
        original_set = Dataset_3(data_tensor=X_train, target_tensor1=Y_train, target_tensor2=M_train)
        original_loader = Data.DataLoader(dataset=original_set, batch_size=bs, shuffle=True)
    
    if vocal or remix:    
        vocal_set = Dataset_3(data_tensor=V_train, target_tensor1=Y_train, target_tensor2=M_train)
        vocal_loader = Data.DataLoader(dataset=vocal_set, batch_size=bs, shuffle=True)
    if remix:
        bgm_set = Dataset_1(data_tensor=B_train)
        bgm_loader = Data.DataLoader(dataset=bgm_set, batch_size=bs, shuffle=True)

    # val
    
    if vocal and not origin:
        X_val = torch.from_numpy(V_val).float()
        Y_val = torch.from_numpy(Y_val).long()
    else:
        X_val = torch.from_numpy(X_val).float()
        Y_val = torch.from_numpy(Y_val).long()

    val_set = Dataset_3(data_tensor=X_val, target_tensor1=Y_val, target_tensor2=M_val)
    val_loader = Data.DataLoader(dataset=val_set, batch_size=bs, shuffle=False)

    # Test

    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test).long()
    V_test = torch.from_numpy(V_test).float()
    M_test = torch.from_numpy(M_test).float()

    test_o_set = Dataset_4(data_tensor=X_test, target_tensor1=Y_test, target_tensor2=S_test, target_tensor3=M_test)
    test_o_loader = Data.DataLoader(dataset=test_o_set, batch_size=bs, shuffle=False)

    test_v_set = Dataset_4(data_tensor=V_test, target_tensor1=Y_test, target_tensor2=S_test, target_tensor3=M_test)
    test_v_loader = Data.DataLoader(dataset=test_v_set, batch_size=bs, shuffle=False)

    #####################################

    best_epoch = 0
    best_F1 = 0

    CELoss = nn.CrossEntropyLoss()
    FocalLoss = model.FocalLoss(gamma=2)

    opt = optim.Adam(Classifier.parameters(),lr=learn_rate)

    print('Start training ...')

    # start_time = time.time()
    early_stop_flag = False
    for epoch in range(epoch_num):
        if early_stop_flag:
            print('rs: ', random_state)
            print('Origin: ', origin, ' | Vocal: ', vocal, ' | Remix: ', remix)
            print('Focal: ', focal, ' | Arc: ', arc)
            print('     best_epoch: ', best_epoch, ' | best_val_F1: %.2f'% best_F1)
            print('     Test original | frame level: %.2f'% test_F1_frame_o, ' | songs level: %.2f'% test_F1_songs_o)
            if vocal:
                print('     Test vocal | frame level: %.2f'% test_F1_frame_v, ' | songs level: %.2f'% test_F1_songs_v)
            break
        if stop_num:
            if epoch - best_epoch >= stop_num:
                early_stop_flag = True
                print('Early Stop!')
        all_loss = 0
        Classifier.train()
        if origin:
            for step, (batch_x, batch_y, batch_m) in enumerate(original_loader):
                
                opt.zero_grad()

                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_m = batch_m.cuda()
                batch_h = torch.randn(1, batch_x.size(0), 32).cuda()
                
                pred_y, emb = Classifier(batch_x, batch_m, batch_h)
                
                loss = CELoss(pred_y, batch_y)
                loss.backward()
                opt.step()

                all_loss += loss
        if vocal:
            for step, (batch_x, batch_y, batch_m) in enumerate(vocal_loader):
                
                opt.zero_grad()

                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_m = batch_m.cuda()
                batch_h = torch.randn(1, batch_x.size(0), 32).cuda()
            
                pred_y, emb = Classifier(batch_x, batch_m, batch_h)
                
                loss = CELoss(pred_y, batch_y)
                loss.backward()
                opt.step()

                all_loss += loss
        if remix:
            for step, ((batch_x, batch_y, batch_m), batch_b) in enumerate(zip(vocal_loader,bgm_loader)):
                
                opt.zero_grad()

                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_m = batch_m.cuda()
                batch_h = torch.randn(1, batch_x.size(0), 32).cuda()
                batch_b = batch_b.cuda()

                batch_x = 10.0*torch.log10((10.0**(batch_x/10.0)) + (10.0**(batch_b/10.0)))
                
                pred_y, emb = Classifier(batch_x, batch_m, batch_h)

                loss = CELoss(pred_y, batch_y)
                loss.backward()
                opt.step()

                all_loss += loss

        print('epoch: ', epoch, ' | Loss: %.4f'% all_loss, ' | time: %.2f'% (time.time()-start_time), '(s)')
        start_time = time.time()
        if epoch % val_num == 0:

            Classifier.eval()

            frame_true = []
            frame_pred = []

            for step, (batch_x, batch_y, batch_m) in enumerate(val_loader):
                
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_m = batch_m.cuda()
                batch_h = torch.randn(1, batch_x.size(0), 32).cuda()
                
                pred_y, emb = Classifier(batch_x, batch_m, batch_h)

                pred_y = pred_y.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                for i in range(len(pred_y)):               
                    frame_true.append(batch_y[i])
                    frame_pred.append(np.argmax(pred_y[i]) )
                
            val_F1 = f1_score(frame_true, frame_pred, average='weighted')
            print('     val F1: %.2f'% val_F1)

            if best_F1 < val_F1:
                best_F1 = val_F1
                best_epoch = epoch

                print('     best_epoch: ', best_epoch, ' | best_val_F1: %.2f'% best_F1)

                torch.save({'Classifier_state_dict': Classifier.state_dict()
                            }, save_folder+'/model/CRNNM2D_elu_model_state_dict')

                frame_true = []
                frame_pred = []

                songs_true = []
                songs_pred = []

                songs_list = []

                emb_list = []

                songs_vote_dict = {}
                songs_true_dict = {}

                for step, (batch_x, batch_y, batch_song, batch_m) in enumerate(test_o_loader):
                    
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    batch_m = batch_m.cuda()
                    batch_h = torch.randn(1, batch_x.size(0), 32).cuda()

                    pred_y, emb = Classifier(batch_x, batch_m, batch_h)

                    pred_y = pred_y.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    emb = emb.detach().cpu().numpy()

                    for i in range(len(pred_y)):                
                        frame_true.append(batch_y[i])
                        frame_pred.append(np.argmax(pred_y[i]))
                        emb_list.append(emb[i])

                        onehot = np.zeros(20)
                        onehot[np.argmax(pred_y[i])] += 1

                        if batch_song[i] not in songs_list:
                            songs_list.append(batch_song[i])
                            songs_true_dict[batch_song[i]] = batch_y[i]
                            songs_vote_dict[batch_song[i]] = onehot
                        else:
                            songs_vote_dict[batch_song[i]] += onehot

                for song in songs_list:
                    songs_true.append(songs_true_dict[song])
                    songs_pred.append(np.argmax(songs_vote_dict[song]))

                # np.save(savedir+'/melody/'+ str(random_state) + '_emb.npy', np.array(emb_list))
                # np.save(savedir+'/melody/'+ str(random_state) + '_true.npy', np.array(frame_true))

                    
                test_F1_frame_o = f1_score(frame_true, frame_pred, average='weighted')
                test_F1_songs_o = f1_score(songs_true, songs_pred, average='weighted')


                frame_true = []
                frame_pred = []

                songs_true = []
                songs_pred = []

                songs_list = []

                songs_vote_dict = {}
                songs_true_dict = {}

                for step, (batch_x, batch_y, batch_song, batch_m) in enumerate(test_v_loader):
                    
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    batch_m = batch_m.cuda()
                    batch_h = torch.randn(1, batch_x.size(0), 32).cuda()

                    pred_y, emb = Classifier(batch_x, batch_m, batch_h)

                    pred_y = pred_y.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()


                    for i in range(len(pred_y)):                
                        frame_true.append(batch_y[i])
                        frame_pred.append(np.argmax(pred_y[i]))

                        onehot = np.zeros(20)
                        onehot[np.argmax(pred_y[i])] += 1
                        
                        if batch_song[i] not in songs_list:
                            songs_list.append(batch_song[i])
                            songs_true_dict[batch_song[i]] = batch_y[i]
                            songs_vote_dict[batch_song[i]] = onehot
                        else:
                            songs_vote_dict[batch_song[i]] += onehot

                for song in songs_list:
                    songs_true.append(songs_true_dict[song])
                    songs_pred.append(np.argmax(songs_vote_dict[song]))
                    
                test_F1_frame_v = f1_score(frame_true, frame_pred, average='weighted')
                test_F1_songs_v = f1_score(songs_true, songs_pred, average='weighted')

                torch.save({'Classifier_state_dict': Classifier.state_dict()
                            }, savedir+'CRNN2D_melody_bgru_model_state_dict')


            # print('CRNN2D bgru')
            # print('epoch: ', epoch, ' | val F1: %.2f'% val_F1, '| time: %.2f'% (time.time()-start_time), '(s)')
            # print('Loss: | Loss: %.4f'% all_loss)
            # print('     best_epoch: ', best_epoch, ' | best_val_F1: %.2f'% best_F1)
            print('     Test original | frame level: %.2f'% test_F1_frame_o, ' | songs level: %.2f'% test_F1_songs_o)
            print('     Test vocal    | frame level: %.2f'% test_F1_frame_v, ' | songs level: %.2f'% test_F1_songs_v)
            # print('==================')
            start_time = time.time()

def parser():
    
    p = argparse.ArgumentParser()

    p.add_argument('-class', '--classes_num', type=int, default=20)
    p.add_argument('-gid', '--gpu_index', type=int, default=0)
    p.add_argument('-bs', '--batch_size', type=int, default=100)
    p.add_argument('-lr', '--learn_rate', type=float, default=0.0001)
    p.add_argument('-val', '--val_num', type=int, default=1)
    p.add_argument('-stop', '--stop_num', type=int, default=20)

    p.add_argument('-rs', '--random_state', type=int, default=0)

    p.add_argument('--origin', dest='origin', action='store_true')
    p.add_argument('--vocal', dest='vocal', action='store_true')
    p.add_argument('--remix', dest='remix', action='store_true')
    p.add_argument('--all', dest='all', action='store_true')

    return p.parse_args()
if __name__ == '__main__':

    args = parser()

    classes_num = args.classes_num
    gid = args.gpu_index
    bs = args.batch_size
    learn_rate = args.learn_rate
    val_num = args.val_num
    stop_num = args.stop_num
    random_state = args.random_state

    origin = args.origin
    vocal = args.vocal
    remix = args.remix

    if args.all:
        origin = True
        vocal = True
        remix = True

    focal = args.Focal_loss
    arc = args.ArcMargin


    print('Singers classification with CRNN2D')
    print('Update in 20191016: artist20 ')

    print('=======================')
    print('classes_num', classes_num)
    print('gpu_index: ', gid, ' | random_state: ', random_state)
    print('bs: ',bs, ' | lr: %.5f'% learn_rate)
    print('val_num: ', val_num, ' | stop_num: ', stop_num)

    print('Origin: ', origin, ' | Vocal: ', vocal, ' | Remix: ', remix)
    
    print('Focal: ', focal, ' | Arc: ', arc)
    print('debug: ', debug)

    print('=======================')

    with torch.cuda.device(gid):
        main(classes_num=classes_num, gid=gid, random_state=random_state, \
            bs=bs, learn_rate=learn_rate, \
            val_num=val_num, stop_num=stop_num,
            origin=origin, vocal=vocal, remix=remix,
            arc=arc, focal=focal
            )

    # args = parser()

    # gid = args.gpu_index
    # bs = args.batch_size
    # learn_rate = args.learn_rate
    # random_state = args.random_state

    # album_split = args.album_split
    # bgm_shuffle=args.shuffle
    # early_stop=args.early_stop

    # print('Singers classification with CRNN2D')
    # print('Update in 20191016: artist20 ')

    # print('=======================')

    # print('gpu_index: ', gid, ' | random_state: ', random_state)
    # print('album_split: ', album_split, ' | bgm_shuffle: ', bgm_shuffle, ' | early_stop: ', early_stop)
    # print('bs: ',bs, ' | lr: %.5f'% learn_rate)


    print('=======================')

    with torch.cuda.device(gid):
        main(gid=gid, bgm_shuffle=bgm_shuffle, early_stop=early_stop,\
            bs=bs, learn_rate=learn_rate, \
            album_split=album_split, random_state=random_state)

