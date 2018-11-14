#*-* coding:utf-8 *-*

import os
import time
import Queue
import signal
import select
import socket
import threading
import numpy as np

from utils import parse_musescore
from alignment import sw_alignment
from pitchDetection.mfshs import MFSHS
from onset_predict import detector_onset
from post_process import Evaluator, trans_onset_and_offset
from onset_predict import load_cnn_model, det_syllable_prob

class Server():

    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setblocking(False)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_address = ('127.0.0.1', 8125)
        self.server.bind(self.server_address)
        self.server.listen(10)
        self.inputs = [self.server]
        self.outputs = []
        self.model_joint, self.scaler_joint = load_cnn_model()

        self.input_jsons = {}
        self.wav_files = {}
        self.out_jsons = {}

        self.pitches = {}
        self.zero_amp_frame = {}
        self.obs_syllable = {}

        self.used_wavs = {}
        self.flags = {}
        self.noused_wavs = {}
        self.useids = []
        self.times = {}

        self._queue_socket = Queue.Queue()
        self.pid = os.getpid()

        t1 = threading.Thread(target=self.check_end)
        t1.start()
        print('start receive client--------')

    def run(self):
        while self.inputs:
            readable, writable, exceptional = select.select(
                self.inputs, self.outputs, self.inputs)
            if not (readable or writable or exceptional):
                break
            for s in readable:
                if s is self.server:
                    connection, client_address = s.accept()
                    connection.setblocking(0)
                    self.inputs.append(connection)
                    self._queue_socket.put(connection)
                else:
                    try:
                        cur_s = self._queue_socket.get()
                        data = cur_s.recv(1024)
                    except:
                        print 'receive data error'
                        self.inputs.remove(cur_s)
                        os.kill(self.pid,signal.SIGKILL)
                    else:
                        if data:
                            wav_file, input_json, out_json, useid, flag = data.split(
                                '+')
                            self.inputs.remove(cur_s)
                            if useid not in self.wav_files.keys():
                                self.wav_files[useid] = list()
                                self.pitches[useid] = np.array([])
                                self.used_wavs[useid] = list()
                                self.flags[useid] = list()
                                self.noused_wavs[useid] = list()
                                self.zero_amp_frame[useid] = np.array([])
                                self.obs_syllable[useid] = np.array([])
                                self.useids.append(useid)

                            
                            cur_wav_id = int(os.path.basename(
                                wav_file).split('.')[0])
                            if len(self.used_wavs[useid]) > 0:
                                last_wav_file = self.used_wavs[useid][-1]
                                last_wav_id = int(os.path.basename(
                                    last_wav_file).split('.')[0])
                            else:
                                last_wav_id = 0
                            isused = False
                            if (cur_wav_id - last_wav_id) == 1:
                                try:
                                    pitches, zero_amp_frame = self.pitch_detect(
                                        wav_file)
                                    obs_syllable = det_syllable_prob(wav_file,
                                                                     self.model_joint,
                                                                     self.scaler_joint)
                                except Exception as e:
                                    print(e)
                                    os.kill(self.pid,signal.SIGKILL)

                                self.pitches[useid] = np.concatenate(
                                    (self.pitches[useid], pitches), axis=0)
                                self.zero_amp_frame[useid] = np.concatenate(
                                    (self.zero_amp_frame[useid], zero_amp_frame), axis=0)
                                self.obs_syllable[useid] = np.concatenate(
                                    (self.obs_syllable[useid], obs_syllable), axis=0)
                                print(wav_file, self.pitches[
                                      useid].shape, self.obs_syllable[useid].shape)
                                self.used_wavs[useid].append(wav_file)
                                isused = True
                            if not isused:
                                self.noused_wavs[useid].append(wav_file)
                                nowav_ids = [int(os.path.basename(wav).split(
                                    '.')[0]) for wav in self.noused_wavs[useid]]
                                indices = np.argsort(np.array(nowav_ids))
                                self.noused_wavs[useid] = np.array(self.noused_wavs[useid])[
                                    indices].tolist()

                            self.wav_files[useid].append(wav_file)
                            if int(flag) == 1:
                                self.out_jsons[useid] = out_json
                                self.input_jsons[useid] = input_json
                                self.times[useid] = time.time()
                            self.flags[useid].append(int(flag))



    def check_end(self):
        while True:
            for useid in self.useids:
                for nowav in self.noused_wavs[useid]:
                    nowav_id = int(
                        os.path.basename(nowav).split('.')[0])
                    for wav in self.used_wavs[useid]:
                        wav_id = int(
                            os.path.basename(wav).split('.')[0])
                        if (nowav_id - wav_id == 1):
                            try:
                                pitches, zero_amp_frame = self.pitch_detect(
                                nowav)
                                obs_syllable = det_syllable_prob(nowav,
                                                             self.model_joint,
                                                             self.scaler_joint)
                            except Exception as e:
                                print(e)
                                os.kill(self.pid,signal.SIGKILL)
                            self.pitches[useid] = np.concatenate(
                                (self.pitches[useid], pitches), axis=0)
                            self.zero_amp_frame[useid] = np.concatenate(
                                (self.zero_amp_frame[useid], zero_amp_frame), axis=0)
                            self.obs_syllable[useid] = np.concatenate(
                                (self.obs_syllable[useid], obs_syllable), axis=0)
                            print(nowav, self.pitches[
                                  useid].shape, self.obs_syllable[useid].shape)
                            self.used_wavs[useid].append(nowav)
                            self.noused_wavs[useid].remove(nowav)

                end = (len(self.wav_files[useid]) == len(
                    self.used_wavs[useid]))
                if (1 in self.flags[useid]) and end:
                    self.detect(useid)
                cur_time = time.time()
                if useid in self.times.keys():
                    duration = cur_time-self.times[useid]
                    if duration>5:
                        os.kill(self.pid,signal.SIGKILL)
            time.sleep(0.1)

    def pitch_detect(self, wav_file):
        mfshs = MFSHS(wav_file)
        mfshs.pitch_detector()
        pitches = mfshs.pitches
        zero_amp_frame = mfshs.zeroAmploc
        return pitches, zero_amp_frame

    def detect(self, useid):
        start_time = time.time()
        try:
            out_json = self.out_jsons[useid]
            score_note, note_type, pauseLoc = parse_musescore(
                self.input_jsons[useid])
            onset_frame = detector_onset(self.obs_syllable[useid],
                                         self.pitches[useid],
                                         score_note)
            match_loc_info = sw_alignment(self.pitches[useid],
                                          onset_frame,
                                          score_note)
            onset_offset_pitches = trans_onset_and_offset(match_loc_info,
                                                          onset_frame,
                                                          self.pitches[useid])
            evaluator = Evaluator(out_json,
                                  onset_offset_pitches,
                                  self.zero_amp_frame[useid],
                                  score_note,
                                  pauseLoc,
                                  note_type)
        except Exception as e:
            print(e)
            os.kill(self.pid,signal.SIGKILL)
        self.clear(useid)
        print time.time() - start_time

    def clear(self, useid):
        del self.out_jsons[useid]
        del self.wav_files[useid]
        del self.input_jsons[useid]
        del self.pitches[useid]
        del self.zero_amp_frame[useid]
        del self.obs_syllable[useid]
        del self.used_wavs[useid]
        del self.flags[useid]
        del self.noused_wavs[useid]
        self.useids.remove(useid)
        del self.times[useid]

if __name__ == "__main__":
    s = Server()
    s.run()
