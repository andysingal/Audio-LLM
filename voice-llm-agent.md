```py
import assemblyai as aai
import ollama
import os
import soundfile as sf
import sounddevice as sd
import replicate
import requests
import io
import sys
from typing import Optional

# Set API Keys
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

if not ASSEMBLYAI_API_KEY:
    raise ValueError("ASSEMBLYAI_API_KEY not set.")
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN not set.")

aai.settings.api_key = ASSEMBLYAI_API_KEY

EXIT_PHRASE = "Power off."  # Define the phrase to trigger exit

class AIVoiceAgent:
    # Initializes the AI Voice Agent with necessary attributes.
    def __init__(self):
        self.replicate_token = REPLICATE_API_TOKEN
        self.transcriber = None
        self.transcript = [{"role": "system", "content": """
        You are an interviewer for a role in data science.
        Can you be proactive in asking questions to see if candidate is a good fit for the role.

        Please keep your answers concise, ideally under 300 characters.
        Please generate only text and no emojis.
        Please start by asking a welcoming question.
        Please ask only one question at a time.
        Instead of * please use numbered lists and use numbered list if there are 2 bullet points.
        """}]
        # You are a helpful coach in Python, data science and AI.
        # You are an English language learning assistant helping young learners learn the English language.


    # Starts the real-time audio transcription
    def _start_transcription(self):
        print("\n 🎙️ Listening...")
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=self._on_data,
            on_error=self._on_error,
            on_open=self._on_open,
            on_close=self._on_close,
        )
        self.transcriber.connect()
        try:
            self.transcriber.stream(aai.extras.MicrophoneStream(sample_rate=16000))
        except Exception as e:
            print(f"Mic error: {e}")
            self._close_transcriber()

    # Stops the ongoing real-time audio transcription if it's active
    def stop_transcription(self):
        if self.transcriber:
            self._close_transcriber()

    # Closes the real-time transcriber and resets the object
    def _close_transcriber(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    # Callback function called when the real-time session is opened
    def _on_open(self, session_opened: aai.RealtimeSessionOpened):
        pass

    # Callback function called when new partial or final transcriptions are received
    def _on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return
        if isinstance(transcript, aai.RealtimeFinalTranscript) and transcript.text.strip():
            print(f"👤 User: {transcript.text}")
            if transcript.text.strip().lower() == EXIT_PHRASE.lower():
                print("\n 🚪 Exiting on user command...")
                self.stop_transcription()
                sys.exit(0)
            else:
                self._generate_response(transcript)
        else:
            print(transcript.text, end="\r", flush=True)

    # Callback function called when an error occurs during real-time transcription
    def _on_error(self, error: aai.RealtimeError):
        print(f"Transcription error: {error}")
        self._close_transcriber()

    # Callback function called when the AssemblyAI real-time session is closed
    def _on_close(self, code: Optional[int] = None, reason: Optional[str] = None):
        self.transcriber = None

    # Generates speech from the given text using Replicate's TTS model and plays it
    def _play_speech(self, text: str):
        if not text.strip():
            return
        audio_url = "N/A"
        try:
            resp = replicate.run(
                "minimax/speech-02-turbo",
                input={"text": text, "pitch": 0, "speed": 1, "volume": 1,
                       "bitrate": 32000, "channel": "mono", "emotion": "happy", # "auto", "neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"
                       "voice_id": "English_Graceful_Lady", "sample_rate": 32000,
                       # English_WiseScholar, English_Graceful_Lady
                       "language_boost": "English", "english_normalization": True}
            )
            audio_url = str(resp)
            if not audio_url or not audio_url.startswith("http"):
                print(f"\n   TTS Error (invalid URL): {audio_url} for '{text}'")
                return
            audio_data = requests.get(audio_url, timeout=20)
            audio_data.raise_for_status()
            data, sr = sf.read(io.BytesIO(audio_data.content))
            sd.play(data, sr)
            sd.wait()

        # Error handling during the text-to-speech process
        except replicate.exceptions.ReplicateError as e:
            print(f"\n   TTS Replicate Error for '{text}': {e}")
        except requests.exceptions.RequestException as e:
            print(f"\n   TTS Download Error ({audio_url}) for '{text}': {e}")
        except sf.SoundFileError as e:
            print(f"\n   TTS Audio Read Error ({audio_url}) for '{text}': {e} (Ensure ffmpeg for MP3)")
        except Exception as e:
            print(f"\n   TTS Unexpected Error for '{text}': {e}")

    # Stops the transcription, adds the user's transcript to the history, generates LLM response, and plays the response
    def _generate_response(self, transcript):
        self.stop_transcription()
        self.transcript.append({"role": "user", "content": transcript.text})
        try:
            ollama_stream = ollama.chat(
                model="gemma3:1b",
                messages=self.transcript,
                stream=True,
            )
        except Exception as e:
            print(f"Ollama Error: {e}")
            self._start_transcription()
            return

        print("\n🤖 AI:", end=" ", flush=True)
        buffer = ""
        full_response = ""

        # Iterates through the streamed response chunks from Ollama
        for chunk in ollama_stream:
            content = chunk['message']['content']
            buffer += content
            print(content, end="", flush=True)

            # Generate speech if it contains a sentence-ending punctuation or exceeds 300 characters
            buffer = buffer.replace("**", "")
            if any(p in buffer for p in ['.', '?', '!', '\n']) or len(buffer) > 300:
                sentence = ""
                processed = False
                for p in reversed(['.', '?', '!', '\n']):
                    if p in buffer:
                        parts = buffer.split(p, 1)
                        sentence = parts[0] + p
                        buffer = parts[1] if len(parts) > 1 else ""
                        processed = True
                        break
                if not processed and len(buffer) > 300:
                    sentence = buffer
                    buffer = ""
                current_sentence = sentence.strip()
                if current_sentence:
                    full_response += current_sentence + " "
                    self._play_speech(current_sentence)

        # Processes and generate speech for any remaining text in the buffer after sentence processing
        remaining = buffer.strip()
        if remaining:
            print(remaining, end="\n", flush=True)
            full_response += remaining + " "
            self._play_speech(remaining)

        # Finalizes the AI response, adds it to chat history, and begins listening again
        final_response = full_response.strip()
        if final_response:
            self.transcript.append({"role": "assistant", "content": final_response})
        print("\n------------------------------------")
        self._start_transcription()

    # Starts the main loop of the AI Voice Agent
    def start(self):
        print(f"⚡ Starting AI Voice Agent... Say '{EXIT_PHRASE}' to exit.")
        self._start_transcription()
        try:
            while True:
                sd.sleep(10)
        except KeyboardInterrupt:
            print("\n 🚪 Exiting...")
            self.stop_transcription()
            print("Exited.")

# Starts the AI voice agent
if __name__ == "__main__":
    try:
        AIVoiceAgent().start()
    except ValueError as e:
        print(f"Config Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
```
