from pytube import YouTube
from huggingsound import SpeechRecognitionModel
import librosa
import torch
import soundfile as sf
from transformers import pipeline
import os

class YouTubeVideoSummarizer:
    def __init__(self, video_url, min_summary_length=5, max_summary_length=100):
        self.video_url = video_url
        self.min_summary_length = min_summary_length
        self.max_summary_length = max_summary_length

    def download_audio(self):
        '''downloads the audio of a youtube video using ffmpeg'''
        yt = YouTube(self.video_url)
        yt.streams.filter(only_audio=True, file_extension='mp4').first().download(filename='ytaudio.mp4')
        os.system('ffmpeg -i ytaudio.mp4 -acodec pcm_s16le -ar 16000 ytaudio.wav')

    def transcribe_audio(self):
        '''audio is transcribed using huggingface model & also chunked to preserve memory'''
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english", device=device)

        input_file = './ytaudio.wav'

        stream = librosa.stream(
            input_file,
            block_length=30,
            frame_length=16000,
            hop_length=16000
        )

        for i, speech in enumerate(stream):
            sf.write(f'{i}.wav', speech, 16000)

        audio_path = [f'./{a}.wav' for a in range(i + 1)]

        transcriptions = model.transcribe(audio_path)

        full_transcript = ' '.join([item['transcription'] for item in transcriptions])

        return full_transcript

    def summarize_text(self, text):
        '''using transformers pretrained model, choose a pipeline to summarise the text'''
        summarization = pipeline('summarization')
        num_iters = int(len(text) / 1000)
        summarized_text = []

        for i in range(0, num_iters + 1):
            start = i * 1000
            end = (i + 1) * 1000
            out = summarization(text[start:end], min_length=self.min_summary_length, max_length=self.max_summary_length)
            out = out[0]
            out = out['summary_text']
            summarized_text.append(out)

        return summarized_text

    def summarize_video(self):
        '''run all processes'''
        self.download_audio()
        full_transcript = self.transcribe_audio()
        summarized_text = self.summarize_text(full_transcript)

        return summarized_text

if __name__ == "__main__":
    VIDEO_URL = 'https://www.youtube.com/watch?v=vywRt-TFZVI&ab_channel=TheSchoolofLife' 
    min_length = 10
    max_length = 100

    summarizer = YouTubeVideoSummarizer(VIDEO_URL, min_length, max_length)
    summarized_text = summarizer.summarize_video()
    print(summarized_text)

