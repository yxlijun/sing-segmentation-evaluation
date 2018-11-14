#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from alignment import sw_alignment
from pitchDetection.mfshs import MFSHS
from post_process import Evaluator,trans_onset_and_offset
from utils import parse_musescore,get_wav_and_json_file,save_files
from onset_predict import detector_onset,load_cnn_model,det_syllable_prob



def _main(wav_file,input_json,output_json):
	'''
		first detect pitches 
	'''
	mfshs = MFSHS(wav_file)
	mfshs.pitch_detector()
	pitches = mfshs.pitches
	zero_amp_frame = mfshs.zeroAmploc
	score_note,note_types,pauseLoc = parse_musescore(input_json)  ## parse musescore
	'''
		second detect onset 
	'''
	model_joint,scaler_joint = load_cnn_model()
	obs_syllable = det_syllable_prob(wav_file,
									model_joint,
									scaler_joint)
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
	evaluator = Evaluator(output_json,
						 onset_offset_pitches,
						 zero_amp_frame,
						 score_note,
						 pauseLoc,
						 note_types)


if __name__ == '__main__':
	wav_file = 'test.mp3'
	input_json = 'input.json'
	output_json = 'output.json'
	_main(wav_file,input_json,output_json)