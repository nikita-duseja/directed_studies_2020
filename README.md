Joint tool for diarization and transcription with word offsets:

Python requirements : 

Transcription 
google-cloud-speech

Diarization :
matplotlib==3.1.2
simplejson==3.16.0
scipy==1.4.1
numpy==1.18.1
hmmlearn==0.2.2
eyeD3==0.8.12
pydub==0.23.1
scikit_learn==0.21.3
tqdm==4.44.1

Installations : Install the google cloud sdk for storage (https://cloud.google.com/sdk/install)

Commands to run : After cloning this repository command to  run : 
./run_process.sh <file-path> <num-speakers> 
eg ./run_process.sh diarization/pyAudioAnalysis/data/diarizationExample.wav 4

A snapshot of the running process is



