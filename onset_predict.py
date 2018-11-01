from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os 
import warnings
import pickle
import collections
import numpy as np 
from keras.models import load_model
from config import model_param,onset_det_param
from audio_preprocessing import feature_reshape
from audio_preprocessing import get_log_mel_madmom
from utils import offset_loc,modify_pitches,smooth_obs


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ""


fs_wav = model_param['fs_wav']
hopsize_t = model_param['hopsize_t']
model_joint_path = model_param['model_joint_path']
model_scaler_path = model_param['model_scaler_path']

min_continue_time = onset_det_param['min_continue_time']
onset_distance = onset_det_param['onset_distance']


def load_cnn_model():
	model_joint = load_model(model_joint_path)
	scaler_joint = pickle.load(open(model_scaler_path))
	return model_joint,scaler_joint

def det_syllable_prob(wav_file,model_joint,scaler_joint):
	log_mel_old = get_log_mel_madmom(wav_file,fs=fs_wav,hopsize_t=hopsize_t,channel=1)
	log_mel = scaler_joint.transform(log_mel_old)
	log_mel = feature_reshape(log_mel,nlen=7)
	log_mel = np.expand_dims(log_mel,axis=1)

	obs_syllable, obs_phoneme = model_joint.predict(log_mel, batch_size=128, verbose=2)

	obs_syllable = np.squeeze(obs_syllable)
	obs_syllable = smooth_obs(obs_syllable)
	obs_syllable[0] = 1.0
	obs_syllable[-1] = 0.0
	return obs_syllable



def syllable_prob(wav_file):
	'''
		load model and get syllable Probability
	return:
		syllable Probability [numpy 1dim]

	'''
	model_joint = load_model(model_joint_path)
	scaler_joint = pickle.load(open(model_scaler_path))
	log_mel_old = get_log_mel_madmom(wav_file,fs=fs_wav,hopsize_t=hopsize_t,channel=1)
	log_mel = scaler_joint.transform(log_mel_old)
	log_mel = feature_reshape(log_mel,nlen=7)
	log_mel = np.expand_dims(log_mel,axis=1)

	obs_syllable, obs_phoneme = model_joint.predict(log_mel, batch_size=128, verbose=2)

	obs_syllable = np.squeeze(obs_syllable)
	obs_syllable = smooth_obs(obs_syllable)
	obs_syllable[0] = 1.0
	obs_syllable[-1] = 0.0
	return obs_syllable



def detector_onset(obs_syllable,pitches,score_note):
	'''
		peak detect to get onset time

	'''
	obs_syllable = obs_syllable*100
	peak = collections.OrderedDict()
	score_length = len(score_note)
	for idx in range(1,len(obs_syllable)-1):
		if (obs_syllable[idx]-obs_syllable[idx-1]>0) \
				and (obs_syllable[idx]-obs_syllable[idx+1]>0) and (obs_syllable[idx]>1.5):
			peak[idx] = obs_syllable[idx]
	if len(peak.keys())<2:
		result_info = {'onset_frame':[],'onset_time':[]}
		return result_info
	syllable_onset = peak.keys()[0:-1]
	syllable_offset = peak.keys()[1:]

	syllable_onset.append(syllable_offset[-1])
	syllable_offset.append(len(obs_syllable)-1)
	realOnset,realCount,avg_Pitch = [],[],[]
	for x in xrange(len(syllable_onset)):
		pitch = pitches[syllable_onset[x]:syllable_offset[x]]
		count = 0
		cancidate_onset = np.empty(shape=(0),dtype=int)
		cancidate_count = np.empty(shape=(0),dtype=int)
		real_pitches = np.empty(shape=(0),dtype=np.float32)
		cancidate_pitch= []
		for i,det in enumerate(pitch,start=1):
			if i==len(pitch):
				if count>=min_continue_time:
					cancidate_onset = np.append(cancidate_onset,syllable_onset[x]+i-count)
					cancidate_count = np.append(cancidate_count,count)
					cancidate_pitch.append(real_pitches)
				break
			diff = abs(int(pitch[i])-int(pitch[i-1]))
			if int(det)==0 or diff>2 or int(det)<20:
				if count>=min_continue_time:
					cancidate_onset = np.append(cancidate_onset,syllable_onset[x]+i-count)
					cancidate_count = np.append(cancidate_count,count)
					cancidate_pitch.append(real_pitches)
				real_pitches = np.delete(real_pitches,np.arange(len(real_pitches)))
				count = 0
			elif diff<=2:
				count+=1
				real_pitches = np.append(real_pitches,pitch[i])

		cancidate_pitch = modify_pitches(cancidate_pitch)
		cancidate_pitch = np.array(cancidate_pitch)
		
		if len(cancidate_onset)>0:
			if len(cancidate_count)>1 and max(cancidate_count)>=30:
				count_low_30_index = np.where(cancidate_count<30)[0]
				cancidate_count = np.delete(cancidate_count,count_low_30_index)
				cancidate_onset = np.delete(cancidate_onset,count_low_30_index)
				cancidate_pitch = np.delete(cancidate_pitch,count_low_30_index)

			for i in range(len(cancidate_count)):
				onset = cancidate_onset[i]
				_count = cancidate_count[i]
				_pitches = cancidate_pitch[i]
				if len(realOnset)==0:
					realOnset.append(onset)
					realCount.append(_count)
					avg_Pitch.append(np.mean(_pitches))
				else:
					if (onset-realOnset[-1])>onset_distance:
						pitch_array = cancidate_pitch[i]
						equal = 1 if abs(avg_Pitch[-1] - np.mean(pitch_array))<0.4 else 0
						length = 1 if (realCount[-1]<30 or _count<30) else 0
						t_pitches = pitches[np.arange(realOnset[-1],onset)]
						_idx = np.where((t_pitches>58.) | (t_pitches[i]==.0))[0]
						conti = 0 if len(_idx)>0 else 1

						startx = 1 if len(realCount) - int(score_length/10)<=0 else len(realCount) - int(score_length/10)
						endx = len(score_note)-1 if (len(realCount) + int(score_length/10))> (len(score_note)-1) \
								else (len(realCount) + int(score_length/10))
						
						score_note = np.array(score_note)
						t_score = score_note[np.arange(startx,endx)]					
						c_score = score_note[np.arange(startx-1,endx-1)]
						idx = np.where(t_score==c_score)[0]
						note = 0 if len(idx)>0 else 1

						if (equal * conti * length * note == 0):
							realOnset.append(onset)
							realCount.append(_count)
							avg_Pitch.append(np.mean(pitch_array))

	print(len(realOnset),len(score_note))
	real_onset_frame = np.array(sorted(realOnset),dtype=np.int)
	if len(real_onset_frame)==score_length:
		onsets = real_onset_frame.copy()
		return real_onset_frame

	offset_frame = real_onset_frame[1:]
	offset_frame = np.append(offset_frame,len(pitches)-1)

	result_info = []
	for idx,cur_onset_frame in enumerate(real_onset_frame):
		pitch_info = {}
		cur_offset_frame = offset_frame[idx]
		pitch = pitches[cur_onset_frame:cur_offset_frame].tolist()
		pitch_info['pitch_length'] = len(pitch)
		pitch_info['onset'] = cur_onset_frame
		pitch_info['offset'] = offset_frame[idx]
		pitch = modify_pitches(np.array(pitch))
		flag = offset_loc(pitch)
		pitch_info['pitch_end'] = flag
		result_info.append(pitch_info)

	pitch_onset_frame = np.empty(shape=(0),dtype=int)
	pitch_on_length = np.empty(shape=(0),dtype=int)
	pitch_end_loc = np.empty(shape=(0),dtype=int)

	for info in result_info:
		pitch_on_length  = np.append(pitch_on_length,info['pitch_end'])
		pitch_onset_frame = np.append(pitch_onset_frame,info['onset'])
		pitch_end_loc = np.append(pitch_end_loc,info['pitch_end']+info['onset'])
	'''
	if len(real_onset_frame)>score_length:
		excessLength = len(real_onset_frame)-score_length
		del_onset = np.argsort(pitch_on_length)[0:excessLength]
		pitch_onset_frame = np.array(np.delete(pitch_onset_frame,del_onset))
		real_onset_frame = pitch_onset_frame
	'''
	return real_onset_frame
