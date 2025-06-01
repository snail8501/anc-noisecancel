import numpy as np
import sounddevice as sd
import numba
import asyncio
import logging

# === Logging 配置 ===
logging.basicConfig(level=logging.INFO)

# === 音频设置 ===
SAMPLERATE = 44100
FRAMES = 8
AMPLIFY = 5.0
CHANNELS = 1
DEVICE_ID = (1, 0)  # 固定输入设备为 1，输出设备为 0

# === Numba 加速 LMS ===
@numba.jit(nopython=True, cache=True)
def _lms_step(x, ref, w, mu):
    y = np.dot(w, ref)
    e = x - y
    w += mu * e * ref
    return e

@numba.jit(nopython=True, cache=True)
def lms_process_frame(input_frame, buffer, w, mu, order, delay, idx, filled):
    N = len(input_frame)
    output = np.zeros(N, dtype=np.float32)
    bufsize = len(buffer)

    for i in range(N):
        buffer[idx] = input_frame[i]
        idx = (idx + 1) % bufsize
        filled = min(filled + 1, bufsize)

        if filled < order + delay:
            output[i] = input_frame[i]
            continue

        ref = np.zeros(order, dtype=np.float32)
        for j in range(order):
            ref_idx = (idx - delay - j - 1) % bufsize
            ref[order - 1 - j] = buffer[ref_idx]

        output[i] = _lms_step(input_frame[i], ref, w, mu)

    return output, idx, filled

# === LMS 类 ===
class FastLMS:
    def __init__(self, order=10, mu=0.0008, delay=10):
        self.order = order
        self.mu = mu
        self.delay = delay
        self.buffer = np.zeros(order + delay + FRAMES, dtype=np.float32)
        self.w = np.zeros(order, dtype=np.float32)
        self.idx = 0
        self.filled = 0

    def process(self, x):
        return lms_process_frame(x, self.buffer, self.w, self.mu, self.order, self.delay, self.idx, self.filled)

lms = FastLMS()

# === Callback 函数 ===
def callback(indata, outdata, frames, time_info, status):
    if status:
        logging.warning(f"⚠️ Audio Callback Status: {status}")

    input_audio = indata[:, 0].copy()
    output_audio, lms.idx, lms.filled = lms.process(input_audio)
    output_audio *= AMPLIFY
    outdata[:, 0] = np.clip(output_audio, -1.0, 1.0)

# === 主程序 ===
async def main():
    logging.info("🎧 正在运行 macOS 自适应噪声消除器（固定设备 ID）...")

    try:
        with sd.Stream(
            samplerate=SAMPLERATE,
            blocksize=FRAMES,
            channels=CHANNELS,
            dtype='float32',
            latency='low',
            device=DEVICE_ID,
            callback=callback
        ):
            while True:
                await asyncio.sleep(0.1)
    except Exception as e:
        logging.error(f"❌ 出错: {e}")

# === 启动入口 ===
if __name__ == "__main__":
    asyncio.run(main())
