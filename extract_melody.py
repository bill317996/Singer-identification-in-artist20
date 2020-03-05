import os
import multiprocessing
import librosa
import numpy as np
import crepe

def get_melody_contour(melody, n_mels=None):
    # quantize
    freq_bin=librosa.core.mel_frequencies(n_mels=n_mels)
    m_contr=np.zeros((freq_bin.shape[0], len(melody))).astype('float32')
    for idx in range(len(melody)):
        p=np.where(freq_bin<=melody[idx])[0][-1]
        m_contr[p, idx]=1.0
    assert all(m_contr.sum(axis=0))
    return m_contr

def extract_melody(gpu, lists):
    import crepe
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    sr=16000
    n_mels=128
    n_fft=2048
    hop_length=512
        
    for file in tqdm(lists):
        song_path, save_path = file
        # Create mel spectrogram and convert it to the log scale
        y, sr = librosa.load(song_path, sr=sr)
        s_size = int(hop_length/sr*1000)
        _, frequency, _, _ = crepe.predict(y, sr, step_size=s_size, viterbi=True, verbose=0)
        m_contr = get_melody_contour(frequency, n_mels=n_mels)

        #Save each song
        np.save(save_path, m_contr)

if __name__ == '__main__':
    art_dir = '/home/kevinco27/nas189/home/dataset/artist20_open_unmix_vocal'
    save_dir = '../song_data_open_unmix_melody_test'
    # use_gpu = ['0','1','2', '3']
    use_gpu = ['0']
    print('art_dir:', art_dir)
    print('save_dir:', save_dir)

    os.makedirs(save_dir, exist_ok=True)
    artists = [path for path in os.listdir(art_dir) if
                os.path.isdir(art_dir+'/'+path)]

    all_list = []
    for artist in artists:
            print(artist)
            artist_path = os.path.join(art_dir, artist)
            artist_albums = os.listdir(artist_path)

            for album in artist_albums:
                album_path = os.path.join(artist_path, album)
                album_songs = os.listdir(album_path)

                for song in album_songs:
                    song_path = os.path.join(album_path, song)
                    #Save each song
                    song = song.split('.')[0]
                    save_name = artist + '_%%-%%_' + album + '_%%-%%_' + song
                    save_path = os.path.join(save_dir, save_name)
                    all_list.append([song_path, save_path])

    length = len(all_list)//len(use_gpu)
    re_len = len(all_list) % len(use_gpu)
    list1 = [all_list.pop() for _ in range(re_len)]
    all_list = np.split(np.array(all_list), len(use_gpu))
    if len(list1)>0:
        all_list[-1] = np.vstack((all_list[-1], list1))

    p_list = []
    for idx, gpu in enumerate(use_gpu):
        p = multiprocessing.Process(target=extract_melody, args=(gpu, all_list[idx]))
        p.daemon=True
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()