#*-* coding:utf-8 *-*
import select
import socket
import Queue
import time
import os
from onset_predict import load_cnn_model,det_syllable_prob
from pitchDetection.mfshs import MFSHS
import numpy as np 
from onset_predict import detector_onset
from utils import parse_musescore
from alignment import sw_alignment
from post_process import Evaluator,trans_onset_and_offset


class Server():
    def __init__(self):
        self.server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.server.setblocking(False)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_address= ('127.0.0.1',8008)
        self.server.bind(self.server_address)
        self.server.listen(10)
        self.inputs = [self.server]
        self.outputs = []
        self.model_joint,self.scaler_joint = load_cnn_model()

        self.input_jsons = {}
        self.wav_files = {}
        self.out_jsons = {}

        self._queue_socket = Queue.Queue()
        print('start receive client--------')

    def run(self):
        while self.inputs:

            readable , writable , exceptional = select.select(self.inputs, self.outputs, self.inputs)
            if not (readable or writable or exceptional) :
                break;
            for s in readable :
                if s is self.server:
                    connection, client_address = s.accept()
                    connection.setblocking(0)
                    self.inputs.append(connection)
                    self._queue_socket.put(connection)
                else:
                    try:
                        cur_s = self._queue_socket.get()
                        #data = s.recv(1024)
                        data = cur_s.recv(1024)
                    except:
                        print 'receive data error'
                        self.inputs.remove(cur_s)
                    else:
                        if data :
                            wav_file,input_json,out_json,useid = data.split('+')
                            self.inputs.remove(cur_s)
                            if useid not in self.wav_files.keys():
                                self.input_jsons[useid] = input_json
                                self.wav_files[useid] = wav_file
                                self.out_jsons[useid] = out_json
                            self.detect(useid)


    def pitch_detect(self,wav_file):
        mfshs = MFSHS(wav_file)
        mfshs.pitch_detector()
        pitches = mfshs.pitches
        zero_amp_frame = mfshs.zeroAmploc
        return pitches,zero_amp_frame


    def detect(self,useid):
        start_time = time.time()
        wav_file = self.wav_files[useid]
        out_json = self.out_jsons[useid]
        pitches,zero_amp_frame = self.pitch_detect(wav_file)
        obs_syllable = det_syllable_prob(wav_file,
                                        self.model_joint,
                                        self.scaler_joint)
        score_note,note_type,pauseLoc = parse_musescore(self.input_jsons[useid])
        onset_frame = detector_onset(obs_syllable,
                                    pitches,
                                    score_note)
        match_loc_info = sw_alignment(pitches,
                                    onset_frame,
                                    score_note)
        onset_offset_pitches = trans_onset_and_offset(match_loc_info,
                                                    onset_frame,
                                                    pitches)
        evaluator = Evaluator(out_json,
                        onset_offset_pitches,
                        zero_amp_frame,
                        score_note,
                        pauseLoc,
                        note_type)

        self.clear(useid)
        print time.time()-start_time

    def clear(self,useid):
        del self.out_jsons[useid]
        del self.wav_files[useid]
        del self.input_jsons[useid]

if __name__ == "__main__":
    s = Server()
    s.run()