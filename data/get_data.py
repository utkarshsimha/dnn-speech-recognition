from scikits.audiolab import Sndfile
import json
from python_speech_features import mfcc, logfbank
import numpy as np
from sklearn.externals import joblib
import pdb
from sklearn import preprocessing
WIN_LEN=0.02
WIN_STEP=0.01

special_chars = { '!':27, '"':28, "'":29, ',':30, '-':31, ':':32, ';':33, '?':34 }
def extractData( file_names ):
        data = []
        targets = []
        for k, v in file_names.items():
                for f_name in v:
                        source_fname = k+"/"+f_name
                        target_fname = k+"/"+f_name.split(".")[0]+".TXT"
                        source_fname = "./TIMIT"+source_fname[1:]
                        target_fname = "./TIMIT"+target_fname[1:]

                        audio_file = Sndfile( source_fname, "r" )
                        sr = audio_file.samplerate
                        audio = audio_file.read_frames( audio_file.nframes )
                        datum = mfcc( audio, samplerate=sr, nfilt=64, numcep=40 )
                        #datum = logfbank( audio, samplerate=sr, nfilt=64 )
                        datum = preprocessing.scale( datum )
                        data.append( datum )
                        audio_file.close()

                        with open( target_fname, "r" ) as text_file:
                                target_txt = ' '.join( text_file.read().lower().strip().replace( ".", "" ).split()[2:] )
                                target_txt = filter( lambda x: x not in special_chars, target_txt )
                                target_txt = target_txt.replace( ' ', '  ' ).split( ' ' )
                                target = np.hstack(['<space>' if x == '' else list(x) for x in target_txt])
                                target = np.asarray( [ 0 if x == '<space>' else ord(x) - ( ord('a') - 1 )\
                                                        for x in target ] )
                                targets.append( target )
        return data, targets

train_files = json.load( open( "train_files.json" ) )
train_data, train_target = extractData( train_files )
joblib.dump( np.asarray( train_data ), "train_data.npy", compress=3 )                   
joblib.dump( np.asarray( train_target ), "train_target.npy", compress=3 )                       

test_files = json.load( open( "test_files.json" ) )
test_data, test_target = extractData( test_files )
joblib.dump( np.asarray( test_data ), "test_data.npy", compress=3 )                     
joblib.dump( np.asarray( test_target ), "test_target.npy", compress=3 )                 


