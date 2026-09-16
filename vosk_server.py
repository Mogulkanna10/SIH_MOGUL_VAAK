import asyncio
import websockets
import sys
import json
from vosk import Model, KaldiRecognizer, SpkModel

if len(sys.argv) == 2:
    model_path = sys.argv[1]
else:
    model_path = "model"

model = Model(model_path)
spk_model = SpkModel("spk_model")

async def recognize(websocket):
    print("Client connected")
    rec = None
    audio_received = False
    try:
        async for message in websocket:
            if isinstance(message, str):
                msg = json.loads(message)
                if 'config' in msg:
                    rec = KaldiRecognizer(model, msg['config']['sample_rate'])
                    rec.SetSpkModel(spk_model)
                elif 'eof' in msg:
                    if rec:
                        final_res = rec.FinalResult()
                        res_json = json.loads(final_res)
                        if "text" in res_json and res_json["text"]:
                            print(f"\n[ASR FINAL] -> {res_json['text']}")
                        await websocket.send(final_res)
                    break
            else:
                if rec:
                    if not audio_received:
                        await websocket.send(json.dumps({"ack": "first_audio_byte"}))
                        audio_received = True
                    if rec.AcceptWaveform(message):
                        res_str = rec.Result()
                        res_json = json.loads(res_str)
                        if "text" in res_json and res_json["text"]:
                            print(f"\n[ASR FINAL] -> {res_json['text']}")
                        await websocket.send(res_str)
                    else:
                        res_str = rec.PartialResult()
                        res_json = json.loads(res_str)
                        if "partial" in res_json and res_json["partial"]:
                            print(f"[ASR Stream] {res_json['partial']}")
                        await websocket.send(res_str)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(recognize, "0.0.0.0", 2700):
        print("Vosk WebSocket server started on ws://localhost:2700")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())

