import numpy as np
import sounddevice as sd
import numba
import asyncio
import logging

# === Logging 配置 ===
logging.basicConfig(level=logging.INFO)  # 保持 INFO 级别，避免 DEBUG 级别日志开销

# === 音频设置 ===
SAMPLERATE = 44100
# *** 优化建议：调整 FRAMES (块大小) ***
# 8 帧可能过于激进，导致底层驱动程序内部缓冲过大，反而增加延迟。
# 尝试一个更平衡的值，如 64。您可以根据实际效果尝试 16, 32, 128。
FRAMES = 64  # 尝试从这里开始，逐步增加，直到延迟降低且稳定
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
            # 如果缓冲区未满，直接传递原始输入（Numba 内部保持 float32 类型）
            output[i] = input_frame[i]
            continue

        ref = np.zeros(order, dtype=np.float32)
        # 确保索引计算正确，并从缓冲区中提取参考向量
        for j in range(order):
            ref_idx = (idx - delay - (order - 1 - j) - 1 + bufsize) % bufsize  # 修正索引逻辑，确保 ref[0] 是最旧的
            ref[j] = buffer[ref_idx]

        output[i] = _lms_step(input_frame[i], ref, w, mu)

    return output, idx, filled


# === LMS 类 ===
class FastLMS:
    def __init__(self, order=10, mu=0.0008, delay=10):
        self.order = order
        self.mu = mu
        self.delay = delay
        # 缓冲区大小需要容纳：滤波器阶数 + 延迟 + 一个完整的帧，以确保在任何时候都有足够的数据。
        # 即使帧大小改变，缓冲区大小也要足够。
        self.buffer = np.zeros(order + delay + FRAMES, dtype=np.float32)
        self.w = np.zeros(order, dtype=np.float32)
        self.idx = 0
        self.filled = 0

    def process(self, x):
        # 注意：lms_process_frame 返回的是 (output, new_idx, new_filled)
        # 所以在回调中接收时要对应
        return lms_process_frame(x, self.buffer, self.w, self.mu, self.order, self.delay, self.idx, self.filled)


lms = FastLMS()


# === Callback 函数 ===
def callback(indata, outdata, frames, time_info, status):
    if status:
        logging.warning(f"⚠️ Audio Callback Status: {status}")

    # indata[:, 0] 通常是视图，而不是副本。如果担心原始数据被修改，可以 copy()
    # 但在此处，input_audio 会被 lms.process 处理，通常没问题。
    input_audio = indata[:, 0]  # float32 类型

    # output_audio, lms.idx, lms.filled = lms.process(input_audio)
    # FIX: 确保 lms.process 的返回值正确地解包并更新到 FastLMS 实例
    processed_output_frame, new_idx, new_filled = lms.process(input_audio)
    lms.idx = new_idx
    lms.filled = new_filled

    processed_output_frame *= AMPLIFY
    # Sounddevice 的 dtype='float32' 意味着它期望 [-1.0, 1.0] 范围的浮点数
    outdata[:, 0] = np.clip(processed_output_frame, -1.0, 1.0)


# === 主程序 ===
async def main():
    logging.info("🎧 正在运行 macOS 自适应噪声消除器（固定设备 ID）...")

    try:
        with sd.Stream(
                samplerate=SAMPLERATE,
                blocksize=FRAMES,
                channels=CHANNELS,
                dtype='float32',  # 确保数据类型为 float32
                latency='low',  # 请求低延迟
                device=DEVICE_ID,
                callback=callback
        ) as stream:  # 将 stream 对象赋值给一个变量
            logging.info(f"Sounddevice 报告的实际输入延迟: {stream.latency[0]:.4f} 秒")
            logging.info(f"Sounddevice 报告的实际输出延迟: {stream.latency[1]:.4f} 秒")
            while True:
                await asyncio.sleep(0.1)
    except Exception as e:
        logging.error(f"❌ 出错: {e}")


# === 启动入口 ===
if __name__ == "__main__":
    logging.info("🚀 启动 Numba 预热...")
    # Numba 预热
    dummy_input_frame = np.zeros(FRAMES, dtype=np.float32)
    dummy_buffer = np.zeros(lms.order + lms.delay + FRAMES, dtype=np.float32)
    dummy_w = np.zeros(lms.order, dtype=np.float32)

    # 调用 Numba 函数进行编译
    lms_process_frame(dummy_input_frame, dummy_buffer, dummy_w, lms.mu, lms.order, lms.delay, 0, 0)
    _lms_step(0.0, np.zeros(lms.order, dtype=np.float32), np.zeros(lms.order, dtype=np.float32), 0.001)

    logging.info("✅ Numba 预热完成。")
    asyncio.run(main())