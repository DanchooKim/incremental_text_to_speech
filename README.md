Incremental text-to-speech
=============
A python implementation of incremental text-to-speech using fastspeech2.

How is it different from previous-work?
-------------
 1. It uses a non-auto-regressive text-to-speech model. (tacotron, transformer-tts -> fastspeech2 )
 2. It uses a simple context discard algorithm for speed-up.

How to use?
-------------
1. Download pretrained tts+vocoder from https://zenodo.org/record/5498896
2. Unzip the file.
3. Place the unzipped files like this:
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
└── incremental_tts.py 

```
                
4. Install anaconda.
5. Make anaconda environments.(recommanded python version -> 3.7.4)
6. Install all python requirements in anaconda enviroments.
- torch (cuda version, no cpu-only version)
- numpy
- espnet2
- pyaudio
7. just type and use. -> python incremental_tts.py
