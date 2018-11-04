#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os 
import time
from alignment import sw_alignment
from pitchDetection.mfshs import MFSHS
from post_process import Evaluator,trans_onset_and_offset
from utils import parse_musescore,get_wav_and_json_file,save_files
from onset_predict import detector_onset,load_cnn_model,det_syllable_prob

def main(wav_file,score_file):
	'''
		first detect pitches 
	'''
	mfshs = MFSHS(wav_file)
	mfshs.pitch_detector()
	pitches = mfshs.pitches
	zero_amp_frame = mfshs.zeroAmploc
	score_note,note_types,pauseLoc = parse_musescore(score_file)  ## parse musescore
	'''
		second detect onset 
	'''
	model_joint,scaler_joint = load_cnn_model()
	obs_syllable = det_syllable_prob(wav_file,
									model_joint,
									scaler_joint)
	print(obs_syllable.shape)
	onset_frame = detector_onset(obs_syllable,
								 pitches,
								 score_note)

	'''
		sw alignment
	'''
	match_loc_info = sw_alignment(pitches,
								  onset_frame,
								  score_note)

	'''
		post process and save result
	'''
	onset_offset_pitches = trans_onset_and_offset(match_loc_info,
												  onset_frame,
												  pitches)
	filename_json = os.path.splitext(wav_file)[0]+".json"
	evaluator = Evaluator(filename_json,
						 onset_offset_pitches,
						 zero_amp_frame,
						 score_note,
						 pauseLoc,
						 note_types)
	save_files(wav_file,
			  onset_frame,
			  pitches,
			  evaluator.det_note,
			  score_note)


if __name__ == '__main__':
	root_path = os.path.join(os.path.dirname(__file__),'data','2000')
	wav_files,score_json = [],[]
	get_wav_and_json_file(root_path,wav_files,score_json)
	start_time = time.time()
	for i in range(len(wav_files)):
		main(wav_files[i],score_json[i])
	print('cost time:',time.time()-start_time)







