import unittest
import numpy as np
import soundfile as sf
from io import BytesIO
from fir import create_fir_filter, apply_filters

class TestFIRFilters(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 44100
        self.audio_data = np.random.randn(self.sample_rate * 2)  # 2 seconds of random audio data

    def test_create_fir_filter(self):
        bass_gain = 1.0
        bass_band = [20, 250]
        filter_coefs = create_fir_filter(bass_gain, bass_band)
        self.assertEqual(len(filter_coefs), 101)
        self.assertTrue(np.all(filter_coefs >= 0))

    def test_apply_filters(self):
        bass_gain = 1.0
        mid_gain = 1.0
        treble_gain = 1.0
        filtered_audio = apply_filters(self.audio_data, bass_gain, mid_gain, treble_gain)
        self.assertEqual(filtered_audio.shape, self.audio_data.shape)

class TestStreamlitApp(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 44100
        self.audio_data = np.random.randn(self.sample_rate * 2)  # 2 seconds of random audio data
        self.uploaded_file = BytesIO()
        sf.write(self.uploaded_file, self.audio_data, self.sample_rate, format="wav", subtype="FLOAT")
        self.uploaded_file.seek(0)

    def test_file_upload(self):
        # Simulate file upload
        audio_data, sample_rate = sf.read(self.uploaded_file)
        self.assertEqual(sample_rate, self.sample_rate)
        self.assertTrue(np.allclose(audio_data, self.audio_data))

    def test_apply_equalizer(self):
        bass_gain = 1.0
        mid_gain = 1.0
        treble_gain = 1.0
        filtered_audio = apply_filters(self.audio_data, bass_gain, mid_gain, treble_gain)
        self.assertEqual(filtered_audio.shape, self.audio_data.shape)

if __name__ == '__main__':
    unittest.main()
