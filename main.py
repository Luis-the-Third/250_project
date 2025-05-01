import assemblyai as aai
import pyaudio
import requests
import websockets
import json
from threading import Thread

api_key = "8aeab857293d438d88f406d126bcf63e"

def on_open(ws):
    def stream_audio():
        while True:
            try:
                audio_data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                ws.send(audio_data, websocket.ABNF.OPCODE_BINARY)
            except Exception as e:
                print(f'\nError streaming audio: {e}')
                break
    audio_thread = Thread(target=stream_audio, daemon=True)
    audio_thread.start()

def on_message(ws, message):
    try:
        msg = json.loads(message)
        msg_type = msg.get('message_type')
        if msg_type == 'SessionBegins':
            session_id = msg.get('session_id')
            print("Session ID:", session_id)
            return
        text = msg.get('text', '')
        if not text:
            return
        if msg_type == 'PartialTranscript':
            print(text, end='\r')
        elif msg_type == 'FinalTranscript':
            print(text, end='\r\n')
        elif msg_type == 'error':
            print(f'\nError: {msg.get("error", "Unknown error")}')
    except Exception as e:
        print(f'\nError handling message: {e}')


def on_error(ws, error):
    print(f'\nError: {error}')

def on_close(ws, status, msg):
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print('\nDisconnected')



done = False

sample_rate = 1000


#General Server instructions:

# curl -X POST https://api.assemblyai.com/v2/realtime/token \
#      -H "Authorization: <apiKey>" \
#      -H "Content-Type: application/json" \
#      -d '{ "username": "my_username", "password": "my_password",
#   "expires_in": 60
# }'


requests = requests.post(
    'https://api.assemblyai.com/v2/realtime/token',
    headers={
        'Authorization': api_key,
        'Content-Type': 'application/json'
    },
    json={'expires_in': 60}
)
token = requests.json().get('token')


#     response = requests.post(url, headers=headers, data=json.dumps(data))
#     response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
#     token = response.json().get('token')
#     if token:
#         print("Token retrieved successfully:", token)
#     else:
#         print("Token not found in response")


ws = websocket.WebSocketApp(
    f'wss://api.assemblyai.com/v2/realtime/ws?sample_rate={SAMPLE_RATE}&token={token}&encoding=pcm_mulaw&disable_partial_transcripts=true',
    # header={'Authorization': YOUR_API_KEY},
    on_message=on_message,
    on_open=on_open,
    on_error=on_error,
    on_close=on_close
)



try:
    ws.run_forever()
except Exception as e:
    print(f'\nError: {e}')

# transcriber = aai.RealtimeTranscriber(
#     ...,
#     token=token,
#     end_utterance_silence_threshold=500,
#     disable_partial_transcripts=True        # to only take in the final transcript
# )

# transcriber.configure_end_utterance_silence_threshold(300) #To change ending time

# while(not done):
#     #transcribing some audio file

#     transcript = transcriber.transcribe("./my-local-audio-file.wav")

#     #print out the audio file
#     print(f"{transcript.text} ")



# token = requests.post(
#     'https://api.assemblyai.com/v2/realtime/token',
#     headers={
#         'Authorization': '<apiKey>',
#         'Content-Type': 'application/json'
#     },
#     json={'expires_in': 60}
# ).json()['token']