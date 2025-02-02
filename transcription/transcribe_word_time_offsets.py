#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample that demonstrates word time offsets.

Example usage:
    python transcribe_word_time_offsets.py resources/audio.raw
    python transcribe_word_time_offsets.py \
        gs://cloud-samples-tests/speech/vr.flac
"""

import argparse
import io
import csv


def my_callback(future):
    result = future.result()


def transcribe_gcs_with_word_time_offsets(gcs_uri):
    """Transcribe the given audio file asynchronously and output the word time
    offsets."""
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()

    audio = types.RecognitionAudio(uri=gcs_uri)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US',
        enable_word_time_offsets=True, model='video')

    operation = client.long_running_recognize(config, audio)
    result = operation.result(timeout=200)
    result_time_offsets = []
    for result in result.results:
        alternative = result.alternatives[0]
        # print(u'Transcript: {}'.format(alternative.transcript))
        # print('Confidence: {}'.format(alternative.confidence))
        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            start = start_time.seconds + start_time.nanos * 1e-9
            end_time = word_info.end_time
            end = end_time.seconds + end_time.nanos * 1e-9
            result_time_offsets.append((word,start,end))
        with open("word-time-offsets", "w") as the_file:
            csv.register_dialect("custom", delimiter=" ", skipinitialspace=True)
            writer = csv.writer(the_file, dialect="custom")
            for tup in result_time_offsets:
                writer.writerow(tup)
# [END speech_transcribe_async_word_time_offsets_gcs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'path', help='File or GCS path for audio file to be recognized')
    args = parser.parse_args()
    if args.path.startswith('gs://'):
        transcribe_gcs_with_word_time_offsets(args.path)
    else:
        transcribe_file_with_word_time_offsets(args.path)
