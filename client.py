# -*- coding:UTF-8 -*-

import os
import socket
import time

def SendDataToServer(wav_file,input_json,output_json,usrid,flag=0):
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect(('127.0.0.1', 8008))
	data = wav_file+'+'+input_json+'+'+output_json+'+'+str(usrid)+"+"+str(flag)
	s.send(data)
	s.close()


if __name__ == "__main__":
	usrid = 1234
	input_json = './data/1A_22/1A_22.json'
	output_json = './data/1A_22/2000.json'
	for i in range(17):
		wav_file = './data/1A_22/'+str(i+1)+'.mp3'
		if i==16:
			SendDataToServer(wav_file,input_json,output_json,usrid,1)
		else:
			SendDataToServer(wav_file,input_json,output_json,usrid)
		time.sleep(0.1)
    
