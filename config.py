from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os 

note_type_param = {
	0.0625:0.5,
	0.125:1,
	0.25:2,
	0.375:3,
	0.5:4,
	0.625:5,
	0.75:6,
	0.875:7,
	1.0:8
}

pitch_det_param={
	'hopsize_t':0.01,
	'fftLength':8192,
	'windowLength':2048,
	'sampleRate':44100,
	'frameSize':2048,
	'hopSize':441,
	'H':5,
	'h':0.8
}

onset_det_param={
	'onset_distance':15,
	'min_continue_time':10
}


alignment_param = {
	'MATCH_COST':0,
	'INSERT_COST':1,
	'DELETE_COST':2
}

root_path = os.path.join(os.path.dirname(__file__))
joint_cnn_model_path = os.path.join(root_path, 'cnnModels', 'joint')

model_param={
	'model_joint_path':os.path.join(joint_cnn_model_path,'jan_joint0.h5'),
	'model_scaler_path':os.path.join(joint_cnn_model_path,'scaler_joint.pkl'),
	'hopsize_t':0.01,
	
	'fs_wav':44100,
}

post_process_param = {
	'sample_ratio':0.3,
	'hopsize_t':0.01,
}

score_json_name=['1A_35','1A_22','C_01','C_02','C_03',
				'C_04','C_05','Jingle_Bells','Poem_Chorus_final',
				'Yankee_Doodle_final','You_and_Me','G1-1']



