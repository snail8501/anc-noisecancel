# anc-noisecancel, Real-Time Aircraft Sound Extraction

This project focuses on extracting clear aircraft sounds in real time from environmental audio by leveraging AI and audio processing techniques.

## Core Approach

1. **Real-Time Audio Capture**  
   Use libraries such as `sounddevice` or similar to capture audio streams from the microphone in real time.

2. **Real-Time Feature Extraction & AI Classification**  
   Apply models like YAMNet or others to classify audio frames in real time, detecting the presence and timing of aircraft sounds.  
   YAMNet is an out-of-the-box environmental sound classification model that can estimate the probability of "aircraft" sounds.

3. **Real-Time Source Enhancement / Separation (Noise Reduction)**  
   Use deep learning models or DSP (Digital Signal Processing) methods for source separation or enhancement, suppressing non-aircraft noises and isolating aircraft sounds.  
   Mature open-source models include Demucs, Conv-TasNet, and VoiceFilter. However, most are designed for speech and may require fine-tuning to specialize in aircraft sound separation.

4. **Output Enhanced Audio**  
   Play back or save the enhanced audio containing the isolated aircraft sounds.
