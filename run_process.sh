## upload file to google storage
file_path=$1
num_speakers=$2
rm -f speaker_windows
rm -f word-time-offsets
gsutil cp $file_path gs://tamu-nduseja-ds/
file_name=`basename $file_path`
gs_storage_path="gs://tamu-nduseja-ds/$file_name"
python transcription/transcribe_word_time_offsets.py $gs_storage_path &
python3 diarization/pyAudioAnalysis/audioAnalysis.py speakerDiarization -i $file_path --num $num_speakers &
echo "Waiting for processes(transcription, diarization) to complete"
while :
do
  if [ -f speaker_windows ] && [ -f word-time-offsets ]
  then
    break
  fi
done
python consolidate_results.py