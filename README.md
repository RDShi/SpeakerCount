# Speaker Recognition and Diarization
speaker recognition and diarization system

## Content
Features are extracted from wav files, then unsupervised learning or supervised learning derived speaker recognition and a certain speaker's voice starting points, ending points

## Data preprocessing
###supervised learning:
<pre>
	train: wavs(labels) --> features(labels) --> GMM model
	predict: wavs --> features --> GMM model --> labels
</pre>
###unsupervised learning
<pre>
	wavs --> features --> kmeans(silhouette) --> num of speaker and time point
</pre>

## File structure
<pre>
model|-speaker_diarization.py: unsupervised learning function
     |-speaker_recognition.py: supervised learning function
     |-supervised_model.py: supervised learning model
     |-unsupervised_model.py: unsupervised learning detail function and model

dataset|-model.out: serialized supervised model
       |-voice_gmm: serialized no_humun_speech model
       |-test_data: test data files
       |-training_data: training data files

result

src

utils|-feature_extraction.py: feature extraction functions
     |-remove_non_human_voice.py: vocal GMM Discriminator
     |-utils.py: normalization, read audio, Gaussian distribution and other utils functions
     |-vad.py: voice discrimination functions 

test:test code
</pre>

## Requirements
<pre>
scipy==1.0.0
librosa==0.5.1
sklearn==0.18.1
numpy==1.14.0
</pre>


## TODO:

*- 边播声音边画图（未完成）-->  ./test/voice_and_members.py
*- 实时（未完成）-->
