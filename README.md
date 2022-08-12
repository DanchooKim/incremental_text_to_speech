Incremental text-to-speech
=============
A python implementation of incremental text-to-speech using fastspeech2.

How to use?
-------------
1. download pretrained tts+vocoder from https://zenodo.org/record/5498896
2. unzip the file.
3. place the unzipped files like this:
```
incremental_tts
├── exp 
|   └── stats
│	    ├── train
│	    |   ├── energy_stats.npz
│	    |   ├── energy_stats.npz
│	    |   └── energy_stats.npz
│	    └── tts
│	        ├── config.yaml
│	        └── train.total_count.ave_10best.pth
├── gan_tts.py 
├── incremental_tts.py 
└── README.py
```
                
4. install anaconda.
5. make anaconda environments.(recommanded python version -> 3.7.4)
6. install all python requirements in anaconda enviroments.
- torch (cuda version, no cpu-only version)
- numpy
- espnet2
- pyaudio
7. type and use. -> python incremental_tts.py
