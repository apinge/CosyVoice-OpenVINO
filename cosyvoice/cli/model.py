# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import numpy as np
import threading
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out
from pathlib import Path
import openvino as ov

class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.llm.fp16 = fp16
        self.flow.fp16 = fp16
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        # here we fix set flow.decoder.estimator.static_chunk_size = 0 for compatibability
        self.flow.decoder.estimator.static_chunk_size = 0
        # mel fade in out
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()
        if self.fp16 is True:
            self.llm.half()
            self.flow.half()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model):
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            self.flow.decoder.estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        if self.flow.decoder.estimator_engine is None:
            raise ValueError('failed to load trt {}'.format(flow_decoder_estimator_model))
        self.flow.decoder.estimator = self.flow.decoder.estimator_engine.create_execution_context()
    
    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        #self.convert_llm_to_ov("/home/gta/qiu/CosyVoice/ov_models")
        with self.llm_context:
            for i in self.llm.inference(text=text.to(self.device),
                                        text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_text=prompt_text.to(self.device),
                                        prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                        prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                        embedding=llm_embedding.to(self.device)):
                self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        tts_mel, flow_cache = self.flow.inference(token=token.to(self.device),
                                                  token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                  prompt_token=prompt_token.to(self.device),
                                                  prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                  prompt_feat=prompt_feat.to(self.device),
                                                  prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                  embedding=embedding.to(self.device),
                                                  flow_cache=self.flow_cache_dict[uuid])
        self.flow_cache_dict[uuid] = flow_cache
        # mel overlap fade in out
        if self.mel_overlap_dict[uuid].shape[2] != 0:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                time.sleep(0.1)
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.flow_cache_dict.pop(this_uuid)

    def vc(self, source_speech_token, flow_prompt_speech_token, prompt_speech_feat, flow_embedding, stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = source_speech_token.flatten().tolist(), True
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)


class CosyVoice2Model(CosyVoiceModel):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.llm.fp16 = fp16
        self.flow.fp16 = fp16
        self.token_hop_len = 2 * self.flow.input_frame_rate
        # here we fix flow encoder/decoder decoding_chunk_size, in the future we will send it as arguments, or use cache
        self.flow.encoder.static_chunk_size = 2 * self.flow.input_frame_rate
        self.flow.decoder.estimator.static_chunk_size = 2 * self.flow.input_frame_rate * self.flow.token_mel_ratio
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder
    def convert_flow_to_ov(self, ov_models_folder,token,
                                token_len,
                                prompt_token,
                                prompt_token_len,
                                prompt_feat,
                                prompt_feat_len,
                                embedding,
                                finalize):
        ov_models_dir = Path(ov_models_folder) / "flow.xml"
        export_model = self.flow
        example_input = {
            "token": token,
            "token_len": token_len,
            "prompt_token": prompt_token,
            "prompt_token_len": prompt_token_len,
            "prompt_feat": prompt_feat,
            "prompt_feat_len": prompt_feat_len,
            "embedding": embedding,
            "finalize": finalize
        }
        ov_model = ov.convert_model(
            export_model,
            example_input,
        )
        ov_model.save(ov_models_dir)
        print("convert flow succeed")
        pass
    def convert_flow_encoder_to_ov(self, ov_models_folder,token,
                                token_len,
                                prompt_token,
                                prompt_token_len,
                                prompt_feat,
                                prompt_feat_len,
                                embedding,
                                finalize):
        print("==========convert flow.encoder to ov==========")

        ov_models_dir = Path(ov_models_folder) / "flow.encoder.xml"
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        from cosyvoice.utils.mask import make_pad_mask
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.flow.input_embedding(torch.clamp(token, min=0)) * mask
        export_model = self.flow.encoder
        export_model.eval()
        #self.flow.forward = self.flow.encoder_forward
        example_input = {
            "xs": token,
            "xs_lens": token_len,
        }
        with torch.no_grad():
            ov_model = ov.convert_model(
                input_model = export_model,
                example_input=example_input,
                #input=[-1,-1,-1]
            )
            ov.save_model(ov_model, ov_models_dir)
        print("==========convert flow.encoder succeed==========")
        pass
    def convert_flow_decoder_to_ov(self, ov_models_folder,token,
                                token_len,
                                prompt_token,
                                prompt_token_len,
                                prompt_feat,
                                prompt_feat_len,
                                embedding,
                                finalize):
        print("==========convert flow.decoder to ov==========")
        ov_models_dir = Path(ov_models_folder) / "flow.decoder.xml"
        
        """
        make input
        """
        # if self.flow.fp16 is True:
        #     prompt_feat = prompt_feat.half()
        #     embedding = embedding.half()

        # assert token.shape[0] == 1
        # # xvec projection
        # from torch.nn import functional as F
        # embedding = F.normalize(embedding, dim=1)
        # embedding = self.flow.spk_embed_affine_layer(embedding)

        # # concat text and prompt_text
        # from cosyvoice.utils.mask import make_pad_mask
        # token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        # mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        # token = self.flow.input_embedding(torch.clamp(token, min=0)) * mask

        # # text encode
        # h, h_lengths = self.flow.encoder(token, token_len)
        # if finalize is False:
        #     h = h[:, :-self.flow.pre_lookahead_len * self.flow.token_mel_ratio]
        # mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        # h = self.flow.encoder_proj(h)

        # # get conditions
        # conds = torch.zeros([1, mel_len1 + mel_len2, self.flow.output_size], device=token.device).to(h.dtype)
        # conds[:, :mel_len1] = prompt_feat
        # conds = conds.transpose(1, 2)

        # mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        save_dir = "/home/gta/qiu/CosyVoice/tensors"
        import os
        mu_loaded = torch.load(os.path.join(save_dir, "mu.pt"))
        mask_loaded = torch.load(os.path.join(save_dir, "mask.pt"))
        spks_loaded = torch.load(os.path.join(save_dir, "spks.pt"))
        cond_loaded = torch.load(os.path.join(save_dir, "cond.pt"))
        """
        export ov model
        """
        export_model = self.flow.decoder
        export_model.eval()
      # def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        
        example_input = {
            "mu": mu_loaded.transpose(1, 2).contiguous(),
            "mask": mask_loaded.unsqueeze(1),
            "spks":spks_loaded,
            "cond":cond_loaded,
           # "n_timesteps":10
        }
        with torch.no_grad():
            ov_model = ov.convert_model(
                input_model = export_model,
                example_input=example_input,
                verbose = True
                #input=[-1,-1,-1]
            )
            ov.save_model(ov_model, ov_models_dir)
        print("==========convert flow.decoder succeed==========")
        pass
    def convert_flow_decoder_to_ov2(self, ov_models_folder,token,
                                token_len,
                                prompt_token,
                                prompt_token_len,
                                prompt_feat,
                                prompt_feat_len,
                                embedding,
                                finalize):
        print("==========convert flow.decoder to ov==========")
        ov_models_dir = Path(ov_models_folder) / "flow.decoder.xml"
        
        """
        make input
        """
        # if self.flow.fp16 is True:
        #     prompt_feat = prompt_feat.half()
        #     embedding = embedding.half()

        # assert token.shape[0] == 1
        # # xvec projection
        # from torch.nn import functional as F
        # embedding = F.normalize(embedding, dim=1)
        # embedding = self.flow.spk_embed_affine_layer(embedding)

        # # concat text and prompt_text
        # from cosyvoice.utils.mask import make_pad_mask
        # token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        # mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        # token = self.flow.input_embedding(torch.clamp(token, min=0)) * mask

        # # text encode
        # h, h_lengths = self.flow.encoder(token, token_len)
        # if finalize is False:
        #     h = h[:, :-self.flow.pre_lookahead_len * self.flow.token_mel_ratio]
        # mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        # h = self.flow.encoder_proj(h)

        # # get conditions
        # conds = torch.zeros([1, mel_len1 + mel_len2, self.flow.output_size], device=token.device).to(h.dtype)
        # conds[:, :mel_len1] = prompt_feat
        # conds = conds.transpose(1, 2)

        # mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        save_dir = "/home/qiu/CosyVoice-OpenVINO/tensors"
        import os
        mu_loaded = torch.load(os.path.join(save_dir, "mu.pt"))
        mask_loaded = torch.load(os.path.join(save_dir, "mask.pt"))
        spks_loaded = torch.load(os.path.join(save_dir, "spks.pt"))
        cond_loaded = torch.load(os.path.join(save_dir, "cond.pt"))
        """
        export ov model
        """
        from torch.export import export
        model = self.flow.decoder
        model.eval()

        example_input = {
            "mu": mu_loaded.transpose(1, 2).contiguous(),
            "mask": mask_loaded.unsqueeze(1),
            "spks": spks_loaded,
            "cond": cond_loaded,
        }
        example_args = (
            mu_loaded.transpose(1, 2).contiguous(),
            mask_loaded.unsqueeze(1),
            spks_loaded,
            cond_loaded,
        )
        
        with torch.no_grad():
            exported_model = export(model, args = example_args, kwargs=example_input)
            ov_model = ov.convert_model(
                input_model = exported_model,
                # example_input=example_input,
                # verbose = True
                #input=[-1,-1,-1]
            )
            ov.save_model(ov_model, ov_models_dir)
        print("==========convert flow.decoder succeed==========")
        pass
    
    def convert_hift_to_ov(self, ov_models_dir,speech_feat: torch.Tensor, cache_source: torch.Tensor):
        print("==========convert hift to ov==========")
        #export_model = self.hift.f0_predictor
        export_model= self.hift
        #self.hift.forward = self.hift.decode
        export_model.eval()
        example_input = {
            "speech_feat": speech_feat,
            #"cache_source": torch.tensor(cache_source)
        }
        with torch.no_grad():
            ov_model = ov.convert_model(
                input_model = export_model,
                example_input=example_input,
            )
            ov.save_model(ov_model, ov_models_dir)
        print("==========convert hift succeed==========")
        pass

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, token_offset, finalize=False, speed=1.0):
        # if not Path('/home/qiu/CosyVoice-OpenVINO/ov_models/flow/flow.decoder.xml').exists():
            # self.convert_flow_decoder_to_ov2('/home/qiu/CosyVoice-OpenVINO/ov_models/flow',token=token.to(self.device),
            #                              token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
            #                              prompt_token=prompt_token.to(self.device),
            #                              prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
            #                              prompt_feat=prompt_feat.to(self.device),
            #                              prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
            #                              embedding=embedding.to(self.device),
            #                              finalize=finalize)
        tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                         token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_token=prompt_token.to(self.device),
                                         prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_feat=prompt_feat.to(self.device),
                                         prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                         embedding=embedding.to(self.device),
                                         finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)  
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        if not Path('/home/gta/qiu/CosyVoice/ov_models/hift/hift.xml').exists():
            self.convert_hift_to_ov('/home/gta/qiu/CosyVoice/ov_models/hift/hift.xml',speech_feat=tts_mel, cache_source=hift_cache_source)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, use_ov=True, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, use_ov=True, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()
        if stream is True:
            token_offset = 0
            while True:
                time.sleep(0.1)
                if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= self.token_hop_len + self.flow.pre_lookahead_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + self.token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     token_offset=token_offset,
                                                     finalize=False)
                    token_offset += self.token_hop_len
                    yield {'tts_speech': this_tts_speech.cpu()}
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) - token_offset < self.token_hop_len + self.flow.pre_lookahead_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             token_offset=token_offset,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             token_offset=0,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
