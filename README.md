# Singer-identification-in-artist20
The source code of "Addressing the confounds of accompaniments in singer identification"
- arxiv: 

### Dependencies

Requires following packages:

- python 3.6
- pytorch 1.3
- h5py
- sklearn
- librosa

### Usage
#### train_CRNN.py


optional arguments:
  -class, classes number, default:20
  -gid, gpu index, default:0
  -bs, batch size, default:100
  -lr, learn rate, default:0.0001
  -val, valid per epoch, default:1
  -stop, early stop, default:20
  -rs random state, default:0
  --origin, use original audio to training
  --vocal, use separated vocal audio to training
  --remix, use remix audio to training
  --all, use all of the above data to training
  --CRNNx2, use CRNNx2 model to training
  --debug, debug mode
```