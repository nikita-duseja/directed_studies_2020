import pandas as pd

df_words = pd.read_csv('word-time-offsets', delim_whitespace=True, header=None)
df_timeStamps = pd.read_csv('data/speaker_windows', delim_whitespace=True, header=None)
results = {}
results[0] = []
results[1] = []
for index, row in df_words.iterrows():
    word = row[0]
    start = row[1]
    end = row[2]
    for index1 in range(df_timeStamps.shape[0]):
        startwin = df_timeStamps.iloc[index1][0]
        endwin = df_timeStamps.iloc[index1][1]
        speakerId = df_timeStamps.iloc[index1][2]
        if start >= startwin and start <= endwin:
            if end <= endwin:
                results[speakerId].append((word, start, end))
            else:
                nextwindow = df_timeStamps.iloc[index1+1]
                nextSpeaker = nextwindow[2]
                duration1 = endwin - start
                duration2 = end - endwin
                if duration1 > duration2:
                    results[speakerId].append((word, start, end))
                else:
                    results[nextSpeaker].append((word, start, end))
            break

print("Words by speaker 1")
print(results[0])

print("Words by speaker 2")
print(results[1])