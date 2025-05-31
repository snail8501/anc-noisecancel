import asyncio
import websockets
import sounddevice as sd
import numpy as np
from http.server import SimpleHTTPRequestHandler, HTTPServer
import threading

PORT = 8080
WS_PORT = 8765
SAMPLERATE = 44100
FRAMES = 512

clients = set()
loop = None  # 需要显式指定事件循环

def int16_to_bytes(data):
    return data.astype(np.int16).tobytes()

async def audio_stream():
    def callback(indata, frames, time, status):
        if status:
            print("⚠️ Input status:", status)
        data = int16_to_bytes(indata[:, 0])
        for ws in clients.copy():
            asyncio.run_coroutine_threadsafe(ws.send(data), loop)

    with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype='int16', blocksize=FRAMES, callback=callback):
        print("🎤 Audio stream started")
        while True:
            await asyncio.sleep(0.1)

async def websocket_handler(websocket):
    print("🔌 Client connected")
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)
        print("❌ Client disconnected")

def start_web_server():
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory="public", **kwargs)
    httpd = HTTPServer(("0.0.0.0", PORT), handler)
    print(f"🌐 Serving HTTP at http://localhost:{PORT}/")
    httpd.serve_forever()

async def main():
    global loop
    loop = asyncio.get_event_loop()

    # 启动 WebSocket 服务
    ws_server = await websockets.serve(websocket_handler, "0.0.0.0", WS_PORT)
    print(f"📡 WebSocket listening on ws://localhost:{WS_PORT}")

    # 启动麦克风音频流（协程内）
    await audio_stream()

if __name__ == "__main__":
    # 启动 HTTP 静态服务器线程
    threading.Thread(target=start_web_server, daemon=True).start()

    # 启动主事件循环
    asyncio.run(main())