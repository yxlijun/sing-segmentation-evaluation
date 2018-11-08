# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os 
import json
import numpy as np 
import config as cfg
from collections import Counter

def smooth_obs(obs):
    """
    hanning window smooth the onset observation function
    :param obs: syllable/phoneme onset function
    :return:
    """
    hann = np.hanning(5)
    hann /= np.sum(hann)

    obs = np.convolve(hann, obs, mode='same')

    return obs



def offset_loc(pitches):
	'''
		find offset loc 
	'''
	pitches_ = np.array(pitches).copy()
	pitches_ = pitches_.astype(int)
	number,start_loc= 0,0
	for i,_det in enumerate(pitches_,start=1):
	    if i==len(pitches_):
	        break
	    pitch_range = sorted(pitches_[i:i+4])
	    smooth_pitch = True
	    if len(pitch_range)==4:
	        max_pitch = pitch_range[-2:]
	        if abs(max_pitch[0]-max_pitch[1])<=2 and max_pitch[0]>25:
	            smooth_pitch = False
	    diff = abs(pitches_[i]- pitches_[i-1])
	    if (_det==0 or _det<20 or (diff>2 and diff!=12 and diff!=11 and diff!=13)) and smooth_pitch:
	        if number>=8:
	            break
	        number = 0
	        start_loc = 0
	    elif diff<=2 or (diff>=11 and diff<=13) or (not smooth_pitch):
	        if number==0:
	            start_loc = i-1
	        number+=1
	flag = number+start_loc
	return flag


def modify_pitches(_pitches):
	'''
		pitch range [20,60],or make it 0
		将pitch值平滑，例如出现45,0,45情况的pitch改为45,45,45		
	'''
	if isinstance(_pitches,list):
		result_pitches = 	[]
		for pitches in _pitches:
			max_note,min_note = 70,20
			pitches = np.array(pitches)
			pitches[np.where(pitches>max_note)[0]] = 0.0
			pitches[np.where(pitches<min_note)[0]] = 0.0
			dpitches = np.copy(pitches)
			for i in range(len(dpitches)-2):
				indices = np.argsort(pitches[i:i+3])
				diff1,diff2 = abs(pitches[i+indices[0]]-pitches[i+indices[1]]),\
					abs(pitches[i+indices[1]]-pitches[i+indices[2]])
				if diff1>2 and diff2<=2:
					dpitches[i+indices[0]] = dpitches[i+indices[2]]
				elif diff1<=2 and diff2>2:
					dpitches[i+indices[2]] = dpitches[i+indices[0]]
			result_pitches.append(dpitches)
		return result_pitches
	else:
		max_note,min_note = 70,20
		pitches = np.array(_pitches)
		pitches[np.where(pitches>max_note)[0]] = 0.0
		pitches[np.where(pitches<min_note)[0]] = 0.0
		dpitches = np.copy(pitches)
		for i in range(len(dpitches)-2):
			indices = np.argsort(pitches[i:i+3])
			diff1,diff2 = abs(pitches[i+indices[0]]-pitches[i+indices[1]]),\
				abs(pitches[i+indices[1]]-pitches[i+indices[2]])
			if diff1>2 and diff2<=2:
				dpitches[i+indices[0]] = dpitches[i+indices[2]]
			elif diff1<=2 and diff2>2:
				dpitches[i+indices[2]] = dpitches[i+indices[0]]
		return dpitches


def parse_musescore(filename):
	'''
		解析json乐谱,返回乐谱中的note值和休止符位置
	'''
	with open(filename,'r') as fr:
		score_info = json.load(fr)
	linenumber = len(score_info['noteInfo'])
	pauseLoc,pitchesLoc = [],[]
	score_pitches,note_types = [],[]
	count = 0
	for number in range(linenumber):
		noteList = score_info['noteInfo'][number]['noteList']
		for note_info in noteList:
			if int(note_info['pitch'])!=0:
				score_pitches.append(int(note_info['pitch']))
				pitchesLoc.append(count)
				note_type = cfg.note_type_param[float(note_info['type'])]
				note_types.append(note_type)
			else:
				pauseLoc.append(count)
			count+=1
	for i,pause in enumerate(pauseLoc):
		index = np.where(np.array(pitchesLoc)<pause)[0]
		pauseLoc[i] = index[-1] if len(index)>0 else 0
		
	return score_pitches,note_types,pauseLoc


def smooth_pitches(cur_pitches):
	'''
		also smooth pitches

	'''
	pitches_ = cur_pitches.astype(int)
	indices = np.where(pitches_>25)[0]
	std_pitches = pitches_[indices]
	counts = np.bincount(std_pitches)
	if len(counts)>0:
		mode_pitch = np.argmax(counts)
		for i,pitch in enumerate(pitches_):
			cur_pitches[i] = mode_pitch if abs(pitch - mode_pitch)>8 and pitch>20 else cur_pitches[i]
	flag = offset_loc(cur_pitches)
	pitch = cur_pitches[0:flag]
	pitch = pitch.astype(int)

	unique_pitch = np.unique(pitch)
	maxnum_pitch = Counter(pitch).most_common(1)[0][0]
	max_indices = np.where(pitch==maxnum_pitch)[0]
	pitches = cur_pitches.copy()
	for _p in unique_pitch:
		if abs(_p - maxnum_pitch)>1:
			indices = np.where(pitch==_p)[0]
			for idx in indices:
				rand_id = np.random.permutation(len(max_indices))[0]
				cur_pitches[idx] = cur_pitches[max_indices[rand_id]]
	return cur_pitches


def filter_pitch(cur_pitches,bool_zero_loc=False):
 	'''
 		smooth pitch and add pause
	param:
		cur_pitches  音调值
		bool_zero_loc 是否需要添加休止符
	return:
		pitches
 	'''
	max_note,min_note = 70,25
	cur_pitches = np.array(cur_pitches)
	cur_pitches[np.where(cur_pitches>max_note)[0]] = 0.0
	cur_pitches[np.where(cur_pitches<min_note)[0]] = 0.0
	dpitches = np.copy(cur_pitches)
	for i in range(len(dpitches)-2):
		indices = np.argsort(cur_pitches[i:i+3])
		diff1,diff2 = abs(cur_pitches[i+indices[0]]-cur_pitches[i+indices[1]]),\
			abs(cur_pitches[i+indices[1]]-cur_pitches[i+indices[2]])
		if diff1>2 and diff2<=2:
			dpitches[i+indices[0]] = dpitches[i+indices[2]]
		elif diff1<=2 and diff2>2:
			dpitches[i+indices[2]] = dpitches[i+indices[0]]
	zero_indices = np.where(dpitches==0)[0]
	if len(zero_indices)<=15 and len(zero_indices)>0:
		dpitches[zero_indices] = dpitches[0]
	elif len(zero_indices)>15:
		dpitches[zero_indices[0]:] = 0.0
	if bool_zero_loc and len(zero_indices)<=15:
		dpitches = np.append(dpitches,np.zeros(15))
	return dpitches.tolist()

def process_pitch(pitches,onset_frame,score_note):
	result_info = []
	offset_frame = onset_frame[1:]
	offset_frame = np.append(offset_frame,len(pitches)-1)
	for idx,cur_onset_frame in enumerate(onset_frame):
		pitch_info = {}
		cur_offset_frame = offset_frame[idx]
		pitch = pitches[cur_onset_frame:cur_offset_frame]
		pitch = smooth_pitches(pitch)
		voiced_length = offset_loc(pitch)
		pitch_info['onset'] = cur_onset_frame
		pitch_info['flag'] = voiced_length
		pitch_info['pitches'] =filter_pitch(pitch,score_note)
		result_info.append(pitch_info)
	return result_info


def pitch_Note(pitches,onset_frame,score_note):
	'''
		将连续的pitch转化为note
	'''
	result_info = process_pitch(pitches,onset_frame,score_note)
	det_pitches = []
	for _info in result_info:
		loc_flag = _info['flag']
		pitches = np.round(np.array(_info['pitches'][:loc_flag])).astype(int)
		pitches = pitches[np.where(pitches>20)[0]]
		unique_pitch = np.unique(pitches)
		number_dict = {}
		for _det in unique_pitch:
			count = pitches.tolist().count(_det)
			number_dict[_det] = count
		number_values = np.array(number_dict.values())
		if len(number_values)>0:
			max_index = np.argmax(number_values)
			det_pitches.append(number_dict.keys()[max_index])
	return det_pitches



def get_wav_and_json_file(root_path,wav_files,score_json):
	'''
		获取文件夹下的音频和乐谱文件
	'''
	path_list = [os.path.join(root_path,file) for file in os.listdir(root_path)]
	for path in path_list:
		if os.path.isdir(path):
			get_wav_and_json_file(path,wav_files,score_json)
		elif os.path.isfile(path):
			if (path.endswith("wav") or path.endswith("mp3")):
				wav_files.append(path)
			elif path.endswith("est"):
				est_files.append(path)
			else:
				filename = os.path.splitext(os.path.basename(path))[0]
				if filename in cfg.score_json_name:
					score_json.append(path)



def save_files(wav_file,onset_time,pitch,det_note,score_note):
	'''
		save all files [onset,pitch,det_note,score_note]
	'''
	# save onset file
	def save_array(filename,array_):
		with open(filename,'w') as fw:
			for arr in array_:
				fw.write(str(arr)+"\n")
	onset_time = np.array(onset_time)*cfg.post_process_param['hopsize_t']
	onset_file = os.path.splitext(wav_file)[0]+"_onset.txt"
	pitch_file = os.path.splitext(wav_file)[0]+"_pitch.txt"
	det_note_file = os.path.splitext(wav_file)[0]+"_detnote.txt"
	score_note_file = os.path.splitext(wav_file)[0]+"_score.txt"
	save_array(onset_file,onset_time)
	save_array(pitch_file,pitch)
	save_array(det_note_file,det_note)
	save_array(score_note_file,score_note)

