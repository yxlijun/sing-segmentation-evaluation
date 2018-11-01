# -*- coding:UTF-8 -*-

import os
import socket


def SendDataToServer(wav_file,input_json,output_json,usrid):
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect(('127.0.0.1', 8008))
	data = wav_file+'+'+input_json+'+'+output_json+'+'+str(usrid)
	s.send(data)
	s.close()


if __name__ == "__main__":
	usrid = 1234
	input_json = './data/2000/1A_22.json'
	output_json = './data/2000/2000.json'
	wav_file = './data/2000/2000.mp3'
	SendDataToServer(wav_file,input_json,output_json,usrid)
    
