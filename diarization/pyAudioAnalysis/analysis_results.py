import csv

def preprocess_diarization_results(duration, speaker_ids):
    duration_per_sample = duration/float(len(speaker_ids))
    speaker_windows_list = []
    start_time = 0
    end_time = 0
    curr_speaker_id = speaker_ids[0]
    for id in speaker_ids:
        if curr_speaker_id == id:
            end_time += duration_per_sample
            continue
        else:
           speaker_windows_list.append((start_time, end_time, curr_speaker_id))
           start_time = end_time
           end_time = end_time + duration_per_sample
           curr_speaker_id = id
    speaker_windows_list.append((start_time, end_time, curr_speaker_id))
    # print(speaker_windows_list)
    with open("speaker_windows", "w") as the_file:
        csv.register_dialect("custom", delimiter=" ", skipinitialspace=True)
        writer = csv.writer(the_file, dialect="custom")
        for tup in speaker_windows_list:
            writer.writerow(tup)