Incremental Text-to-speech
=============
efficient incremental inference algorithm for fastspeech2 in espnet2
-------------
How to use?
-------------
1. download pretrained tts+vocoder from https://zenodo.org/record/5498896
2. unzip the file.
3. place the unzipped files like this:
    incremental_tts
        L exp
            L stats 
                L train
                    L energy_stats.npz
                    L feats_stats.npz
                    L pitch_stats.npz
            L tts
                L config.yaml
                L train.total_count.ave_10best.pth
4. install anaconda.
5. make anaconda environments.(recommanded python version -> 3.7.4)
6. install all python requirements in anaconda enviroments.
- torch (cuda version, no cpu-only version)
- numpy
- espnet2
- pyaudio
7. type and use. -> python incremental_tts.py

<img src="https://img.shields.io/badge/Firebase-FFCA28?style=flat-square&logo=firebase&logoColor=white"/>
