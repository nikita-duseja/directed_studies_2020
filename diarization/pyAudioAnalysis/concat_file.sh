COUNTER=0
for file_name in `ls -rt data/wav_out/out_concat_*.wav`
do
	out_file_name="out_concat_${COUNTER}.wav"
	COUNTER=$[$COUNTER +1]
	echo "$out_file_name"
	python3 audioAnalysis.py speakerDiarization -i data/wav_out/${out_file_name} --num 0 >> log
done
