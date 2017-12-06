import scipy.io.wavfile as wav
import json

train_files = json.load( open( "train_files.json" ) )
for k, v in train_files.items():
	for f_name in v:
		fs, audio = wav.read( k+"/"+f_name )
		assert fs == 16000

