# -*- coding: utf-8 -*-
# written by Danchu Kim
# stud088@ust.ac.kr 

from typing import Union
from typing import Any
from typing import Dict
from typing import Optional
import numpy as np
import torch
import copy
import pyaudio
from multiprocessing import Process
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.bin.tts_inference import Text2Speech
from espnet2.torch_utils.device_funcs import to_device
import espnet2.text.phoneme_tokenizer as p_tokenizer
import espnet2.text.phoneme_tokenizer as p_tokenizer
import sys

class Synthesizer(Text2Speech):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.use_spembs = False
        self.device = ["cpu","cuda:0"]
        #print(self.model)
        self.model.to(self.device[0])
        self.tts_model = self.model.tts.generator['text2mel'].to(self.device[0])
        self.vocoder = self.model.tts.generator['vocoder'].to(self.device[1])
        self.use_pqmf = True
        self.g2p = p_tokenizer.G2p_en(no_space=True)
            
    def gan_tts(
        self,
        text: Union[str, torch.Tensor, np.ndarray],
        spembs: Union[torch.Tensor, np.ndarray] = None,
        decode_conf: Optional[Dict[str, Any]] = None
        ) -> Dict[str, torch.Tensor]:
        
        with torch.no_grad():
            torch.cuda.empty_cache()
            if self.use_spembs and spembs is None:
                raise RuntimeError("Missing required argument: 'spembs'")
            if isinstance(text, str):
                text = self.preprocess_fn("<dummy>", dict(text=text))["text"]
            
            batch = dict(text=text)
            print(batch)            
            
            if spembs is not None:
                batch.update(spembs=spembs)
            #torch.Tensor(batch)
            batch = to_device(batch['text'], self.device[0])
            # overwrite the decode configs if provided
            cfg = self.decode_conf
            if decode_conf is not None:
                cfg = self.decode_conf.copy()
                cfg.update(decode_conf)            
            # inference
            if self.always_fix_seed:
                set_all_random_seed(self.seed)
            output_dict = self.tts_model.inference(text=batch, **cfg)
            
            #### 
            output_dict = to_device(output_dict, 'cpu')
            input_feat = output_dict
            input_feat = to_device(input_feat, self.device[1])
            wav = self.vocoder.inference(input_feat['feat_gen'])
            '''
            if self.use_pqmf:
                wav = self.pqmf.synthesis(wav.unsqueeze(0).transpose(1,2))
                wav = wav.squeeze(0).transpose(0,1)
            '''
        
            wav = wav.cpu()
            return wav
        
    def incremental_tts_overlap(
        self,
        text: Union[str, torch.Tensor, np.ndarray],
        prefix_text : Union[str, torch.Tensor, np.ndarray]='',
        postfix_text : Union[str, torch.Tensor, np.ndarray]='', 
        is_last : Union[str, bool] = False,
        spembs: Union[torch.Tensor, np.ndarray] = None,
        decode_conf: Optional[Dict[str, Any]] = None,
        overlap_frame : int = 500,
        dbug : bool = False
        ) -> Dict[str, torch.Tensor]:
        '''
        look front(prefix) and rear(postfix) context when synthesize the speech.
        overlap_frame option allows the smooth connection of each speech chunk using fade-in fade-out effect.
        '''
        tmp_text = copy.deepcopy(text)
        with torch.no_grad():
            torch.cuda.empty_cache()
            if self.use_spembs and spembs is None:
                raise RuntimeError("Missing required argument: 'spembs'")
            if isinstance(text, str):
                text = self.preprocess_fn("<dummy>", dict(text=text))["text"]
            # print(text)
            batch = dict(text=text)
            p_tokenizer.g2p_en
            prefix_text_len = len(self.g2p(prefix_text)) 
            postfix_text_len = len(self.g2p(postfix_text))
            real_text_len = len(self.g2p(tmp_text))-postfix_text_len-prefix_text_len
            if dbug:
                print('prefix:',prefix_text)            
                print('real:',tmp_text)            
                print('postfix:',postfix_text)            
                
            if spembs is not None:
                batch.update(spembs=spembs)

            batch = to_device(batch['text'], self.device[0])
            # overwrite the decode configs if provided
            cfg = self.decode_conf
            if decode_conf is not None:
                cfg = self.decode_conf.copy()
                cfg.update(decode_conf)
            
            # inference
            if self.always_fix_seed:
                set_all_random_seed(self.seed)
    
            output_dict = self.tts_model.inference(text= batch, **cfg)
            output_dict = to_device(output_dict, 'cpu')
            
            prefix_spectrogram_duration = [0,output_dict['duration'][0:prefix_text_len].numpy().sum()]
            text_spectrogram_duration = [prefix_spectrogram_duration[1],prefix_spectrogram_duration[1]+output_dict['duration'][prefix_text_len:prefix_text_len+real_text_len].numpy().sum()]

            if is_last:
                starting_point = text_spectrogram_duration[0]
                input_feat = output_dict["feat_gen"][starting_point::1]
            else:
                starting_point = text_spectrogram_duration[0]
                ending_point = text_spectrogram_duration[1]  
                input_feat = output_dict["feat_gen"][starting_point:ending_point]
            
            input_feat = to_device(input_feat, self.device[1])
            if len(input_feat)<=0:
                return torch.Tensor([])
            wav = self.vocoder.inference(input_feat)
            wav = wav.cpu().numpy()                  
            wav = np.squeeze(wav)
            silence_len = 0
            fade_len = overlap_frame - silence_len
            silence = np.zeros((silence_len), dtype = np.float64)
            linear = np.zeros((silence_len), dtype = np.float64)
                    
            # Equal power crossfade
            t = np.linspace(-1,1,fade_len, dtype=np.float64)
            fade_in = np.sqrt(0.5 * (1 + t))
            fade_out = np.sqrt(0.5 * (1 - t))
                    
            # Concat the silence to the fades
            fade_in = np.concatenate([silence, fade_in])
            fade_out = np.concatenate([linear, fade_out])                    

            wav[:overlap_frame] *= fade_in
            wav[-overlap_frame:] *= fade_out

            return wav
        
class sound_play_process(Process):
    def __init__(self, sp_queue):
        super().__init__()
        self.input_queue = sp_queue
    
    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=21500,
                        frames_per_buffer=1024,
                        output=True
                        )

        try:
            while True:
                samples = self.input_queue.get()
                stream.write(samples.astype(np.float32).tostring())

        except KeyboardInterrupt as e:
            stream.close()
            sys.exit(1)
