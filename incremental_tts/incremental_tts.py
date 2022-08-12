# -*- coding: utf-8 -*-
# written by Danchu Kim
# stud088@ust.ac.kr 

from multiprocessing import dummy
import sys
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Queue, Process
import copy
from gan_tts import Synthesizer
import copy
import pyaudio
import numpy as np
from time import sleep

class _prefix_limited_incremental_tts_process(Process):
    def __init__(self, input_queue, sp_queue, prefix=7):
        super().__init__()
        self.input_queue = input_queue
        self.sound_play_queue = sp_queue
        self.dbug = False
        self.prefix = prefix + 1

        self.do_postfix = True
        self.default_postfix = 'I mean'
    
    def run(self):
        neural_tts = Synthesizer(
                    train_config="exp/tts/config.yaml",
                    model_file="exp/tts/train.total_count.ave_10best.pth",
                    use_att_constraint=True,
                )
        try:
            dummy_queue = []
            while True:            
                cur_hyps = []
                prefix_text = []
                is_last = False
                while not is_last:
                    dummy_queue.append(self.input_queue.get())
                    tok = dummy_queue.pop(0)
                    if tok == '<eos>':
                        is_last = True
                        continue
                    if self.do_postfix:
                        if len(dummy_queue)!=0:
                            postfix=[' '+dummy_queue[0]]
                        else:
                            dummy_queue.append(self.input_queue.get())
                            postfix=[' '+dummy_queue[0]]
                    else:
                        postfix=[' '+self.default_postfix]
                    if len(cur_hyps) == self.prefix:
                        cur_hyps = copy.deepcopy(cur_hyps[1::1])
                    prefix_text = copy.deepcopy(cur_hyps)
                    cur_hyps.append(' '+tok)
                    
                    arg1 = ''.join(prefix_text)
                    arg2 = ''.join(cur_hyps)
                    target = cur_hyps[-1]
                    print('prefix :' , arg1)
                    print('target :' , target)
                    arg2 = arg2+''.join(postfix)
                    arg3 = ''.join(postfix)               
                    print('postfix :' , arg3)
                    print('-----------------')
                    output_audio=neural_tts.incremental_tts_overlap(prefix_text=arg1, text=arg2, postfix_text=arg3, is_last=is_last, overlap_frame=220)
                    self.sound_play_queue.put_nowait(output_audio)
                    
                    
        except KeyboardInterrupt as e:
            sys.exit(1)

class sound_play_process(Process):
    def __init__(self, sp_queue):
        super().__init__()
        self.input_queue = sp_queue
    
    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=22050,
                        frames_per_buffer=1024,
                        output=True
                        )
        
        while True:
            samples = self.input_queue.get()
            try:
                stream.write(samples.astype(np.float32).tostring())
            except Exception as e:
                continue

class Streaming_text_generator(Process):
    def __init__(self, out_q: Queue, input_text: str):
        super().__init__()
        self.out_q = out_q
        self.text = input_text
        
    def run(self):
        text = self.text
        flag = False
        
        while True:
            new_hyps = text.split(' ')
            if '<eos>' not in new_hyps:
                new_hyps.append('<eos>')
            
            while len(new_hyps)>0:
                
                a = new_hyps.pop(0)
                if len(a) <= 3:
                    sleep(0.25)
                else:
                    sleep(0.3)
            
                while not flag:
                    try:
                        self.out_q.put(a,timeout=0.05)
                        flag = True
                    except:
                        continue
                flag = False             
            
if __name__ == '__main__':

    '''
    for real service...
    > change the Streaming_text_generator process to real text generator like Dialog generator, ASR, Translator...
    '''
    
    text_q = Queue()
    sound_q = Queue()
    sp_pr = sound_play_process(sound_q)
    sg_pr = Streaming_text_generator(out_q = text_q, input_text = 'Though wise men at their end know dark is right, Because their words had forked no lightning they Do not go gentle into that good night.')
    tts_pr = _prefix_limited_incremental_tts_process(sp_queue=sound_q, input_queue=text_q, prefix=5) # default prefix => 5
    
    try:
        sg_pr.start()
        sp_pr.start()
        tts_pr.start()
        sp_pr.join()
        tts_pr.join()
        sg_pr.join()
    except KeyboardInterrupt as e:
        sg_pr.kill()
        sp_pr.kill()
        tts_pr.kill()
        sys.exit(1)
        
        #process
