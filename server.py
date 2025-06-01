import numpy as np
import sounddevice as sd
import asyncio
import logging
import numba
from scipy.signal import butter, lfilter, iirnotch
import platform  # Corrected: Used for OS detection

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Configuration ---
SAMPLERATE = 44100  # Sample rate (samples per second)
# CRITICAL for low latency: Smaller FRAMES leads to lower buffer latency.
# Test this value carefully. Too low may cause crackles/dropouts.
FRAMES = 16  # Number of audio frames processed per block.
AMPLIFY_OUTPUT = 10.0  # Output signal amplification factor. Adjust by ear to prevent clipping.


# --- SingleChannelLMS Class (for Adaptive Noise Cancellation) ---
class SingleChannelLMS:
    def __init__(self, filter_order=32, learning_rate_mu=0.002, delay_samples=50):
        """
        Initializes the LMS filter for single-channel adaptive noise cancellation.
        Parameters optimized for aircraft engine noise and low latency.
        """
        self.filter_order = filter_order  # Reduced for lower processing latency
        self.learning_rate_mu = learning_rate_mu  # Stable learning rate
        self.delay_samples = delay_samples  # Reduced for lower algorithmic latency

        self.weights = np.zeros(self.filter_order, dtype=np.float32)

        # Calculate total buffer size, accommodating filter order, delay, and one frame.
        self.buffer_size = self.filter_order + self.delay_samples + FRAMES - 1
        self.input_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_idx = 0  # Pointer for next incoming audio frame write position

        self.filled_samples = 0  # Tracks number of samples filled for initial buffer prime

        logging.info(
            f"SingleChannelLMS Initialized: Order={self.filter_order}, Learning Rate={self.learning_rate_mu}, Delay={self.delay_samples} samples, Buffer Size={self.buffer_size}")

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _process_single_sample_numba(current_input_sample, reference_vector, weights, learning_rate_mu):
        """
        Core LMS single-sample processing logic, optimized with Numba.
        """
        predicted_noise = np.dot(weights, reference_vector)
        error_signal = current_input_sample - predicted_noise
        weights += learning_rate_mu * error_signal * reference_vector
        return error_signal

    def process_frame(self, input_signal_frame):
        """
        Processes an audio frame with single-channel LMS noise cancellation.
        """
        output_frame = np.zeros_like(input_signal_frame, dtype=np.float32)
        frame_len = len(input_signal_frame)

        # Update circular buffer with the new frame
        end_idx = self.write_idx + frame_len
        if end_idx <= self.buffer_size:
            self.input_buffer[self.write_idx:end_idx] = input_signal_frame
        else:
            first_part_len = self.buffer_size - self.write_idx
            self.input_buffer[self.write_idx:self.buffer_size] = input_signal_frame[:first_part_len]
            self.input_buffer[0:frame_len - first_part_len] = input_signal_frame[first_part_len:]

        self.write_idx = end_idx % self.buffer_size
        self.filled_samples = min(self.buffer_size, self.filled_samples + frame_len)

        # Check if enough historical data is available for full processing
        required_data_for_full_processing = self.filter_order + self.delay_samples
        if self.filled_samples < required_data_for_full_processing:
            logging.debug(
                f"Buffer not yet full ({self.filled_samples}/{required_data_for_full_processing}), skipping LMS processing.")
            return input_signal_frame

        # Process each sample in the current frame
        for i in range(frame_len):
            current_input_sample = input_signal_frame[i]
            current_sample_buffer_idx = (self.write_idx - frame_len + i + self.buffer_size) % self.buffer_size
            reference_start_offset = self.delay_samples + self.filter_order - 1
            linear_start_idx = current_sample_buffer_idx - reference_start_offset

            if linear_start_idx >= 0:
                reference_vector = self.input_buffer[linear_start_idx: linear_start_idx + self.filter_order]
            else:
                part1 = self.input_buffer[linear_start_idx + self.buffer_size: self.buffer_size]
                part2 = self.input_buffer[0: self.filter_order - len(part1)]
                reference_vector = np.concatenate((part1, part2))

            if len(reference_vector) != self.filter_order:
                # Should not happen if filled_samples check is robust
                output_frame[i] = current_input_sample
                continue

            error_signal_sample = SingleChannelLMS._process_single_sample_numba(
                current_input_sample,
                reference_vector,
                self.weights,
                self.learning_rate_mu
            )
            output_frame[i] = error_signal_sample

        # Stability checks
        if np.any(np.isnan(output_frame)) or np.max(np.abs(output_frame)) > 1e9:
            logging.error("❌ LMS output contains NaN or exploded! Learning rate (Mu) might be too high.")
            self.weights = np.zeros(self.filter_order, dtype=np.float32)
            logging.info("LMS weights reset, system recovered.")
            return input_signal_frame
        elif np.any(np.isnan(self.weights)):
            logging.error("❌ LMS weights contain NaN! Learning rate (Mu) might be too high.")
            self.weights = np.zeros(self.filter_order, dtype=np.float32)
            logging.info("LMS weights reset.")
            return input_signal_frame
        return output_frame


# --- StaticFilters Class (for Hiss, Wind, Fixed-Frequency Hums) ---
class StaticFilters:
    def __init__(self, samplerate):
        self.samplerate = samplerate

        self.lp_b = None
        self.lp_a = None
        self.zi_lp = None

        self.notch_b = None
        self.notch_a = None
        self.zi_notch = None

        # Low-pass filter for high-frequency hiss/air sounds in aircraft noise.
        self._configure_lowpass_filter(cutoff_freq=280, order=4)

        # Notch filter for specific hums (e.g., 50Hz electrical hum).
        self._configure_notch_filter(freq=50.0, Q_factor=70.0)
        # self._configure_notch_filter(freq=60.0, Q_factor=70.0) # For 60Hz regions
        # self._configure_notch_filter(freq=120.0, Q_factor=50.0) # Harmonics

    def _configure_lowpass_filter(self, cutoff_freq, order):
        nyquist = 0.5 * self.samplerate
        normal_cutoff = cutoff_freq / nyquist
        self.lp_b, self.lp_a = butter(order, normal_cutoff, btype='lowpass', analog=False)
        self.zi_lp = np.zeros(max(len(self.lp_b), len(self.lp_a)) - 1)

    def _configure_notch_filter(self, freq, Q_factor):
        nyquist = 0.5 * self.samplerate
        w0 = freq / nyquist
        self.notch_b, self.notch_a = iirnotch(w0, Q_factor)
        self.zi_notch = np.zeros(max(len(self.notch_b), len(self.notch_a)) - 1)

    def apply_filters(self, audio_frame):
        processed_frame = audio_frame.copy()
        if self.lp_b is not None and self.lp_a is not None:
            processed_frame, self.zi_lp = lfilter(self.lp_b, self.lp_a, processed_frame, zi=self.zi_lp)
        if self.notch_b is not None and self.notch_a is not None:
            processed_frame, self.zi_notch = lfilter(self.notch_b, self.notch_a, processed_frame, zi=self.zi_notch)
        return processed_frame


# --- Global Filter Instances ---
lms_filter = SingleChannelLMS(
    filter_order=32,  # Further reduced for lower processing latency
    learning_rate_mu=0.002,
    delay_samples=50  # Further reduced for lower algorithmic latency
)
static_filters = StaticFilters(SAMPLERATE)


# --- Audio Callback Function ---
def callback(indata, outdata, frames, time_info, status):
    if status:
        logging.warning("Stream status warning: %s", status)

    if indata.shape[1] > 1:
        logging.warning("⚠️ Multi-channel input detected, processing only the first channel.")

    # Convert int16 to float32 normalized to [-1.0, 1.0]
    input_mono = indata[:, 0].astype(np.float32) / 32768.0

    # Apply static filters first for high-frequency and fixed-frequency noise
    processed_audio = static_filters.apply_filters(input_mono)

    # Then apply LMS filter for residual periodic noise (engine hum)
    processed_audio = lms_filter.process_frame(processed_audio)

    # Amplify and convert to output format (int16)
    # Clip to prevent audio distortion from exceeding int16 range
    output_data = np.clip(processed_audio * AMPLIFY_OUTPUT * 32768.0, -32768, 32767).astype(np.int16)

    # Ensure output is single channel
    outdata[:, 0] = output_data

    logging.debug("🎧 Input Max Abs: %f, 🔊 Output Max Abs: %d",
                  np.max(np.abs(input_mono)), np.max(np.abs(output_data)))


# --- Main Asynchronous Loop ---
async def main():
    logging.info("Available audio devices:\n%s", sd.query_devices())
    try:
        # --- CRITICAL LATENCY OPTIMIZATION: Choose lowest latency audio API and correct device IDs ---
        # 1. Run `import sounddevice as sd; print(sd.query_devices())` to find your device IDs and supported hostapis.
        # 2. For Windows, prioritize 'ASIO' (if ASIO driver installed) or 'WASAPI' (exclusive mode).
        # 3. For macOS, 'CoreAudio' is usually default and best.
        # 4. For Linux, 'ALSA' is usually best.

        # Example: Using default input/output devices from sounddevice
        input_device_id = sd.default.device[0]
        output_device_id = sd.default.device[1]

        # Attempt to find and use ASIO (if available)
        # Note: ASIO typically only supports specific hardware and requires corresponding drivers.
        # If no ASIO or error, comment out hostapi line or change to another backend (e.g., 'WASAPI').
        hostapi_index = None
        for i, api in enumerate(sd.query_hostapis()):
            # Using platform.system() for OS detection
            if platform.system() == 'Windows' and 'ASIO' in api['name']:
                hostapi_index = i
                logging.info(f"Detected ASIO hostapi, ID: {hostapi_index}.")
                break

        # If ASIO not found, try WASAPI (Exclusive Mode) on Windows
        if hostapi_index is None and platform.system() == 'Windows':
            for i, api in enumerate(sd.query_hostapis()):
                if 'WASAPI' in api['name'] and 'Exclusive' in api['name']:
                    hostapi_index = i
                    logging.info(f"Detected WASAPI (Exclusive) hostapi, ID: {hostapi_index}.")
                    break
            if hostapi_index is None:
                logging.warning(
                    "Could not find ASIO or WASAPI (Exclusive) hostapi. Using system default. Latency may be higher.")

        # Ensure input/output devices are compatible with selected hostapi
        # You might need to manually specify device IDs based on `sd.query_devices()` output.
        # Example: device=(1, 0)
        # The 'hostapi' parameter is only needed if manually specifying device or overriding default.

        stream_params = {
            'samplerate': SAMPLERATE,
            'blocksize': FRAMES,  # Reduced for lower latency
            'channels': 1,  # Force single-channel input
            'dtype': 'int16',
            'callback': callback,
            'device': (input_device_id, output_device_id),
        }
        if hostapi_index is not None:
            stream_params['hostapi'] = hostapi_index

        with sd.Stream(**stream_params):
            logging.info(
                "🎤 Starting Noise Cancellation System (primarily for aircraft engine hum). Press Ctrl+C to stop.")
            logging.info(f"Current FRAMES (block size) set to {FRAMES} for latency optimization.")
            logging.info(
                f"Current LMS filter_order set to {lms_filter.filter_order} and delay_samples to {lms_filter.delay_samples} for low processing/algorithmic latency.")
            while True:
                await asyncio.sleep(1)  # Keep stream running
    except Exception as e:
        logging.error(f"Error starting audio stream: {e.__class__.__name__}: {e}")
        logging.info("Please check your device IDs and hostapi settings, and ensure devices are connected and working.")
        logging.info("Consider increasing FRAMES (block size) or changing hostapi if experiencing dropouts.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("👋 Exiting gracefully.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e.__class__.__name__}: {e}")