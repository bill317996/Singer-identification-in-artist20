# Singer-identification-in-artist20
The source code of "Addressing the confounds of accompaniments in singer identification"
- arxiv: https://arxiv.org/abs/2002.06817

### Dependencies

Requires following packages:

- python 3.6
- pytorch 1.3
- crepe 0.0.10
- librosa 0.7.1
- dill 0.3.1.1
- tqdm
- h5py
- sklearn

### Usage
#### extract_fea.py
Extracting melspectrograms of artist20 
1. **Origin**: the original artist20, containing both vocals and accompaniments. `art_dir`: path to artist20
    
2. **Vocal**: the vocal-only artist20, separated by [open_unmix](https://github.com/sigsep/open-unmix-pytorch).`art_dir`: path to pure vocals of artist20 (the folder structure should follow the artist20's)
    
3. **Accompaniment**: the accompaniment-only artist20 (bass+drums+other), separated by [open_unmix](https://github.com/sigsep/open-unmix-pytorch). `art_dir`: path to pure accompaniments of artis20 (the folder structure should follow the artist20's )
    
#### extract_melody.py
extract the melody of vocals using [crepe](https://github.com/marl/crepe)
#### train_CRNN.py
```
usage: train_CRNN.py [-h] [-class CLASSES_NUM] [-gid GPU_INDEX]
                     [-bs BATCH_SIZE] [-lr LEARN_RATE] [-val VAL_NUM]  
                     [-stop STOP_NUM] [-rs RANDOM_STATE] [--origin] [--vocal]
                     [--remix] [--all] [--CRNNx2] [--debug]

optional arguments:
  -class, classes number (default:20)
  -gid, gpu index (default:0)
  -bs, batch size (default:100)
  -lr, learn rate (default:0.0001)
  -val, valid per epoch (default:1)
  -stop, early stop (default:20)
  -rs random state (default:0)
  --origin, use original audio to training
  --vocal, use separated vocal audio to training
  --remix, use remix audio to training
  --all, use all of the above data to training
  --CRNNx2, use CRNNx2 model to training
  --debug, debug mode
```
#### predict_on_audio.py
```
python predict_on_audio.py your_song_path
```
