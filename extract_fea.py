import os
import librosa
import dill
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

def wave2spec(file_list):
    for file in tqdm(file_list):    
        sr=16000
        n_mels=128
        n_fft=2048
        hop_length=512
        
        artist_folder, artist, album, song, save_folder = file
        artist_path = os.path.join(artist_folder, artist)
        album_path = os.path.join(artist_path, album)
        song_path = os.path.join(album_path, song)
        # Create mel spectrogram and convert it to log scale
        y, sr = librosa.load(song_path, sr=sr)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                            n_fft=n_fft,
                                            hop_length=hop_length)
        log_S = librosa.core.amplitude_to_db(S, ref=1.0)
        data = (artist, log_S, song)

        # Save each song
        save_name = artist + '_%%-%%_' + album + '_%%-%%_' + song
        with open(os.path.join(save_folder, save_name), 'wb') as fp:
            dill.dump(data, fp)

def create_dataset_parellel(artist_folder='artists', save_folder='song_data',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512, num_worker=10):
    """This function creates the dataset given a folder
     with the correct structure (artist_folder/artists/albums/*.mp3)
    and saves it to a specified folder."""

    # get list of all artists
    os.makedirs(save_folder, exist_ok=True)
    artists = [path for path in os.listdir(artist_folder) if
               os.path.isdir(artist_folder+'/'+path)]

    all_list = []
    
    # iterate through all artists, albums, songs and find mel spectrogram
    for artist in tqdm(artists):
        artist_path = os.path.join(artist_folder, artist)
        artist_albums = os.listdir(artist_path)
        for album in artist_albums:
            album_path = os.path.join(artist_path, album)
            album_songs = os.listdir(album_path)
            for song in album_songs:
                song_path = os.path.join(album_path, song)
                all_list.append([artist_folder, artist, album, song, save_folder])
    
    length = len(all_list)/num_worker
    re_len = len(all_list) % num_worker


    list1 = [all_list.pop() for _ in range(re_len)]
    all_list = np.split(np.array(all_list), num_worker)
    all_list.append(np.array(list1))

    pool = mp.Pool(processes=num_worker+1)
    pool.map(wave2spec, all_list)
    pool.close()
    pool.join

if __name__ == '__main__':    
    # origin
    art_dir = '/home/kevinco27/nas189/home/dataset/artist20'
    save_dir = '../song_data_artis20_origin'
    print('art_dir:', art_dir)
    print('save_dir:', save_dir)
    create_dataset_parellel(artist_folder=art_dir, save_folder=save_dir, num_worker=10)

    # vocal
    # folder structure should follow artist20's
    art_dir = '/home/kevinco27/nas189/home/dataset/artist20_open_unmix_vocal'
    save_dir = '../song_data_artis20_vocal'
    print('art_dir:', art_dir)
    print('save_dir:', save_dir)
    create_dataset_parellel(artist_folder=art_dir, save_folder=save_dir, num_worker=10)

    # accompaniment
    # folder structure should follow artist20's
    art_dir = '/home/kevinco27/nas189/home/dataset/artist20_open_unmix_accomp'
    save_dir = '../song_data_artis20_accomp'
    print('art_dir:', art_dir)
    print('save_dir:', save_dir)
    create_dataset_parellel(artist_folder=art_dir, save_folder=save_dir, num_worker=10)