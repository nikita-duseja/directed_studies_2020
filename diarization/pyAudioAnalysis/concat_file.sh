## upload file to google storage
file_path=$1
gsutil cp $file_path gs://tamu-nduseja-ds/
file_name=`basename $file_path`
gs_storage_path="gs://tamu-nduseja-ds/$file_name"
python transcription/transcribe_word_time_offsets.py transcription/data/commercial_mono.wav &
python3 diarization/pyAudioAnalysis/audioAnalysis.py speakerDiarization -i $file_path --num 2
