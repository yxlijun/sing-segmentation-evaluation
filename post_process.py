#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import json
import numpy as np 
from config import post_process_param
from utils import offset_loc,smooth_pitches,filter_pitch


sample_ratio = post_process_param['sample_ratio']
hopsize_t = post_process_param['hopsize_t']


def trans_onset_and_offset(match_loc_info,onset_frame,pitches):
	'''	
		after sw alignment to modify onset
	return:
		dict include {onset_frame,offset,pitches,add zero_loc}
	'''
	modify_onset,modify_index = [],[]
	pading_zero_loc = match_loc_info['zero_loc']
	locate_info = match_loc_info['loc_info']
	for i,info in enumerate(locate_info):
		if i not in pading_zero_loc:
			modify_onset.append(onset_frame[info[0]])
			modify_index.append(i)
	modify_onset = sorted(modify_onset)
	modify_index = np.array(modify_index)
	add_onset = []
	for i in pading_zero_loc:
		if i==0:
			modify_onset.append(1)
			modify_index = np.append(modify_index,0)
		else:
			insert_index1 = np.where(modify_index>i)[0]
			insert_index2 = np.where(modify_index<i)[0]
			if len(insert_index1)>0 and len(insert_index2)>0:
				modify_onset.append((modify_onset[insert_index1[0]]+modify_onset[insert_index2[-1]])//2)
				modify_index = np.append(modify_index,i)
			elif len(insert_index1)==0:
				modify_onset.append(modify_onset[-1]+20)
		modify_onset =  sorted(modify_onset)
		modify_index = np.sort(modify_index)

	offset_frame = modify_onset[1:]
	offset_frame = np.append(offset_frame,len(pitches)-1)
	onset_frame = modify_onset

	onset_offset_pitches = {
		'onset_frame':onset_frame,
		'offset_frame':offset_frame,
		'pitches':pitches,
		'add_zero_loc':pading_zero_loc
	}
	return onset_offset_pitches



class Evaluator(object):
	"""
		save result and give score
	"""
	def __init__(self, filename,onset_offset_pitches,zeroAmpLoc,score_note,pauseLoc):
		super(Evaluator, self).__init__()
		self.filename = filename
		self.onset_frame = onset_offset_pitches['onset_frame']
		self.offset_frame = onset_offset_pitches['offset_frame']
		self.pitches = onset_offset_pitches['pitches']
		self.add_zero_loc = onset_offset_pitches['add_zero_loc']
		self.zeroAmpLoc = zeroAmpLoc
		self.score_note = score_note
		self.pauseLoc = pauseLoc

		self.result_info,self.det_Note = [],[]
		self._offset_frame = []
		self.evalutate_save_file()


	@property
	def det_note(self):
		return self.det_Note


	@property
	def offset_amp_frame(self):
		return self._offset_frame
	
	def pitches_to_result(self):
		for idx,cur_onset_frame in enumerate(self.onset_frame):
			bool_zero_loc = True if idx in self.pauseLoc else False
			pitch_info = {}
			if idx in self.add_zero_loc:
				pitch_info['onset'] = cur_onset_frame*hopsize_t*1000
				pitch = np.zeros(10)
				pitch = filter_pitch(pitch,bool_zero_loc)
				pitch_info['pitches'] = pitch
				pitch_info['flag'] = 10 if len(pitch)>10 else 0
				self.result_info.append(pitch_info)
				self.det_Note.append(0.0)
			else:
				cur_offset_frame = self.offset_frame[idx]
				pitch = self.pitches[cur_onset_frame:cur_offset_frame]
				first_zeroAmp_distance,cur_zero_amp_loc = self.find_zero_frame(cur_onset_frame,cur_offset_frame)
				pitch = smooth_pitches(pitch)
				voice_len = offset_loc(pitch)
				noise_len = (first_zeroAmp_distance - voice_len) if (first_zeroAmp_distance - voice_len)>0 else 0
				voice_len +=noise_len
				slience_len = len(pitch) - voice_len
				sample_voice_len = int(voice_len*sample_ratio)
				sample_slience_len = int(slience_len*sample_ratio)
				voice_indices = np.random.permutation(sample_voice_len)
				slience_indices = np.random.permutation(sample_slience_len)
				voice_pitch = pitch[:voice_len][voice_indices]
				slience_pitch = np.zeros(sample_slience_len)
				pitch = np.append(voice_pitch,slience_pitch).tolist()
				pitch = filter_pitch(pitch,bool_zero_loc)

				if (len(pitch)-sample_voice_len)<15:
					pitch = pitch[:sample_voice_len]
				else:
					zero_pitch = np.zeros(len(pitch)-sample_slience_len).tolist()
					pitch[sample_voice_len:] = zero_pitch
				flag = sample_voice_len
				pitch_info['onset'] = cur_onset_frame*hopsize_t*1000
				pitch_info['flag'] = flag
				pitch_info['pitches'] = pitch

				self.result_info.append(pitch_info)
				note = np.array(pitch[:flag])
				self.det_Note.append(np.round(np.mean(note)))

				if(len(cur_zero_amp_loc)>0):
					self._offset_frame.append(cur_zero_amp_loc[0])
		return 


	def find_zero_frame(self,onset,offset):
		cur_zero_amp_loc = self.zeroAmpLoc[np.where((self.zeroAmpLoc>onset) &(self.zeroAmpLoc<offset))[0]]
		first_zeroAmp_distance = (cur_zero_amp_loc[0]-onset-1) if len(cur_zero_amp_loc)>0 else 0
		return first_zeroAmp_distance,cur_zero_amp_loc


	def give_score(self):
		det_note = np.array(self.det_Note)
		score_note = np.array(self.score_note)
		diff_note = (det_note - score_note).astype(np.int)
		indices = np.where((diff_note>=-24) & (diff_note<=24))[0]
		is_octive_1 = bool((np.mean(diff_note[indices])>=10) and (np.mean(diff_note[indices])<=14))
		is_octive_2 = bool((np.mean(diff_note[indices])>=-14) and (np.mean(diff_note[indices])<=-10))
		if is_octive_1:
			_det_note = det_note-12
		elif is_octive_2:
			_det_note = det_note+12
		else:
			_det_note = det_note
		is_octive = (is_octive_1 or is_octive_2)
		count = 0
		LowOctive = list()
		for i,note in enumerate(score_note):
			if abs(note - _det_note[i])<=1.5 or \
			(note<=40 and ((_det_note[i]-note)>=10 and (_det_note[i]-note)<=14)) or \
			(note>=52 and ((note - _det_note[i])>=10 and (note - _det_note[i])<=14)):
				count+=1
			if is_octive:
				if ((det_note[i]-note)>=10 and (det_note[i]-note)<=14):
					LowOctive.append(1)
				elif ((note - det_note[i])>=10 and (note - det_note[i])<=14):
					LowOctive.append(-1)
				else:
					LowOctive.append(0)
			else:
				if note<=40 and ((det_note[i]-note)>=10 and (det_note[i]-note)<=14):
					LowOctive.append(1)
				elif note>=52 and ((note - det_note[i])>=10 and (note - det_note[i])<=14):
					LowOctive.append(-1)
				else:
					LowOctive.append(0)
		score = count *100.0 / len(score_note)
		return score,is_octive,LowOctive


	def evalutate_save_file(self):
		discardData = (len(self.score_note)-len(self.onset_frame))>0.15*len(self.score_note)
		if not discardData:
			self.pitches_to_result()
		score,is_octive = 0,False
		if len(self.det_Note):
			score,is_octive,LowOctive = self.give_score()
		for i in range(len(self.result_info)):
			self.result_info[i]['octive'] = LowOctive[i]
		results = {
			'score':score,
			'is_octive':is_octive,
			'pitches_info':self.result_info
		}
		with open(self.filename,'w') as fw:
			json.dump(results,fw)
		print('score:',score)