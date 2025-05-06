[dia](https://github.com/nari-labs/dia)

Dia is a 1.6B parameter text to speech model created by Nari Labs.

Dia directly generates highly realistic dialogue from a transcript. You can condition the output on audio, enabling emotion and tone control. The model can also produce nonverbal communications like laughter, coughing, clearing throat, etc.

[MLX-Audio](https://github.com/Blaizzy/mlx-audio)
A text-to-speech (TTS) and Speech-to-Speech (STS) library built on Apple's MLX framework, providing efficient speech synthesis on Apple Silicon.

```py
"uvx --from mlx-audio mlx_audio.tts.generate --model mlx-community/Dia-1.6B-6bit --text "[S1] Dia can now run on your Mac thanks to MLX. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs)""
```
