import unittest
import numpy as np
from scipy.signal import firwin
import fir  # Assuming your module is named `fir.py`

class TestFIRFilterModule(unittest.TestCase):
    
    def setUp(self):
        """
        Set up test environment, e.g., sample audio data and default parameters.
        Case: Initialize test environment
        Type: Setup
        Input: None
        Expected Output: None
        Output: None
        """
        # Create sample mono and stereo audio data
        self.mono_audio = np.random.randn(44100)  # 1 second of random noise (mono)
        self.stereo_audio = np.random.randn(44100, 2)  # 1 second of random noise (stereo)
        
        # Default frequency bands and gains
        self.bass_band = [20, 250]
        self.mid_band = [250, 4000]
        self.treble_band = [4000, 20000]
        self.bass_gain = 1.0
        self.mid_gain = 1.0
        self.treble_gain = 1.0

    def test_create_fir_filter_lowpass(self):
        """
        Test the creation of a lowpass filter.
        Case: Lowpass filter creation
        Type: Functionality
        Input: bass_gain, bass_band, pass_zero='lowpass'
        Expected Output: Filter coefficients of length 101, symmetric coefficients
        Output: Filter coefficients
        """
        filter_coefs = fir.create_fir_filter(self.bass_gain, self.bass_band, pass_zero='lowpass')
        self.assertEqual(len(filter_coefs), 101)  # Default numtaps is 101
        self.assertTrue(np.allclose(filter_coefs[0], filter_coefs[-1]))  # Symmetry of FIR filter
        
    def test_create_fir_filter_bandpass(self):
        """
        Test the creation of a bandpass filter.
        Case: Bandpass filter creation
        Type: Functionality
        Input: mid_gain, mid_band, pass_zero='bandpass'
        Expected Output: Filter coefficients of length 101, non-symmetric coefficients
        Output: Filter coefficients
        """
        filter_coefs = fir.create_fir_filter(self.mid_gain, self.mid_band, pass_zero='bandpass')
        self.assertEqual(len(filter_coefs), 101)  # Default numtaps is 101
        self.assertEqual(filter_coefs[0], filter_coefs[-1])  # Bandpass filters should not be symmetric

    def test_apply_filters_mono(self):
        """
        Test applying filters to mono audio data.
        Case: Apply filters to mono audio
        Type: Functionality
        Input: mono_audio, bass_gain, mid_gain, treble_gain
        Expected Output: Filtered audio and individual band components of the same length as input audio, non-zero output
        Output: Filtered audio, bass_filtered, mid_filtered, treble_filtered
        """
        filtered_audio, bass_filtered, mid_filtered, treble_filtered = fir.apply_filters(
            self.mono_audio, self.bass_gain, self.mid_gain, self.treble_gain
        )
        
        # Check that filtered_audio is the same length as input audio
        self.assertEqual(len(filtered_audio), len(self.mono_audio))
        
        # Check that the bass, mid, and treble components are also the same length
        self.assertEqual(len(bass_filtered), len(self.mono_audio))
        self.assertEqual(len(mid_filtered), len(self.mono_audio))
        self.assertEqual(len(treble_filtered), len(self.mono_audio))
        
        # Ensure the output is not just zeros (this checks if filtering is occurring)
        self.assertTrue(np.any(filtered_audio != 0))

    def test_apply_filters_stereo(self):
        """
        Test applying filters to stereo audio data.
        Case: Apply filters to stereo audio
        Type: Functionality
        Input: stereo_audio, bass_gain, mid_gain, treble_gain
        Expected Output: Filtered audio and individual band components of the same shape as input audio, non-zero output
        Output: Filtered audio, bass_filtered, mid_filtered, treble_filtered
        """
        filtered_audio, bass_filtered, mid_filtered, treble_filtered = fir.apply_filters(
            self.stereo_audio, self.bass_gain, self.mid_gain, self.treble_gain
        )
        
        # Check that filtered_audio is the same shape as input stereo audio
        self.assertEqual(filtered_audio.shape, self.stereo_audio.shape)
        
        # Check that the bass, mid, and treble components are also the same length as input audio
        self.assertEqual(bass_filtered.shape, (self.stereo_audio.shape[0],))
        self.assertEqual(mid_filtered.shape, (self.stereo_audio.shape[0],))
        self.assertEqual(treble_filtered.shape, (self.stereo_audio.shape[0],))
        
        # Ensure the output is not just zeros (this checks if filtering is occurring)
        self.assertTrue(np.any(filtered_audio != 0))

    def test_invalid_audio_data(self):
        """
        Test for invalid audio data (e.g., 3D arrays or invalid input types).
        Case: Invalid audio data
        Type: Error Handling
        Input: 3D array audio data
        Expected Output: Raise ValueError
        Output: ValueError
        """
        with self.assertRaises(ValueError):
            fir.apply_filters(np.random.randn(44100, 2, 2), self.bass_gain, self.mid_gain, self.treble_gain)
        
    def test_gain_effect(self):
        """
        Test that applying different gains to filters affects the output.
        Case: Gain effect on filtered output
        Type: Functionality
        Input: mono_audio, different bass gains (0.5 and 2.0), mid_gain, treble_gain
        Expected Output: Higher gain results in larger amplitude in filtered output
        Output: Filtered audio with different gains
        """
        # Apply filters with different gains
        filtered_audio_low_gain, _, _, _ = fir.apply_filters(self.mono_audio, 0.5, self.mid_gain, self.treble_gain)
        filtered_audio_high_gain, _, _, _ = fir.apply_filters(self.mono_audio, 2.0, self.mid_gain, self.treble_gain)
        
        # Ensure that the higher gain results in a larger amplitude in the filtered output
        self.assertGreater(np.max(np.abs(filtered_audio_high_gain)), np.max(np.abs(filtered_audio_low_gain)))
        
    def test_create_fir_filter_min_gain(self):
        """
        Test the creation of a filter with minimum gain (zero).
        Case: Minimum gain filter creation
        Type: Functionality
        Input: gain=0, mid_band, pass_zero='bandpass'
        Expected Output: Filter coefficients of length 101, all coefficients should be zero
        Output: Filter coefficients
        """
        filter_coefs = fir.create_fir_filter(0, self.mid_band, pass_zero='bandpass')
        self.assertEqual(len(filter_coefs), 101)  # Ensure the number of taps is unchanged
        self.assertTrue(np.allclose(filter_coefs, 0))  # All coefficients should be zero

    def test_create_fir_filter_max_gain(self):
        """
        Test the creation of a filter with maximum gain.
        Case: Maximum gain filter creation
        Type: Functionality
        Input: max_gain=1e6, mid_band, pass_zero='bandpass'
        Expected Output: Filter coefficients of length 101, coefficients magnitude scaled by max_gain
        Output: Filter coefficients
        """
        max_gain = 1e6
        filter_coefs = fir.create_fir_filter(max_gain, self.mid_band, pass_zero='bandpass')
        self.assertEqual(len(filter_coefs), 101)  # Ensure the number of taps is unchanged
        self.assertTrue(np.allclose(filter_coefs[0] * max_gain, filter_coefs[-1] * max_gain))  # Check coefficients magnitude

    def test_create_fir_filter_invalid_band(self):
        """
        Test the creation of a filter with an invalid frequency range (lowcut > highcut).
        Case: Invalid frequency range
        Type: Error Handling
        Input: mid_gain, invalid band [4000, 250], pass_zero='bandpass'
        Expected Output: Raise ValueError
        Output: ValueError
        """
        with self.assertRaises(ValueError):
            fir.create_fir_filter(self.mid_gain, [4000, 250], pass_zero='bandpass')

    def test_create_fir_filter_invalid_filter_type(self):
        """
        Test the creation of a filter with an invalid filter type.
        Case: Invalid filter type
        Type: Error Handling
        Input: mid_gain, mid_band, pass_zero='invalidtype'
        Expected Output: Raise ValueError
        Output: ValueError
        """
        with self.assertRaises(ValueError):
            fir.create_fir_filter(self.mid_gain, self.mid_band, pass_zero='invalidtype')

    def test_create_fir_filter_narrow_band(self):
        """
        Test the creation of a very narrow bandpass filter (1000Hz to 1001Hz).
        Case: Narrow bandpass filter creation
        Type: Functionality
        Input: mid_gain, narrow_band [1000, 1001], pass_zero='bandpass'
        Expected Output: Filter coefficients of length 101, non-trivially symmetric coefficients
        Output: Filter coefficients
        """
        narrow_band = [1000, 1001]
        filter_coefs = fir.create_fir_filter(self.mid_gain, narrow_band, pass_zero='bandpass')
        self.assertEqual(len(filter_coefs), 101)  # Ensure the number of taps is unchanged
        self.assertEqual(filter_coefs[0], filter_coefs[-1])  # Check that filter is not trivially symmetric

    def test_apply_filters_stereo_audio(self):
        """
        Test applying FIR filters on stereo audio input.
        Case: Apply filters to stereo audio
        Type: Functionality
        Input: stereo audio data, bass_gain, mid_gain, treble_gain
        Expected Output: Filtered audio and individual band components of the same shape as input audio
        Output: Filtered audio, bass_filtered, mid_filtered, treble_filtered
        """
        audio_data = np.random.randn(44100, 2)  # Simulate stereo audio (1 second of random noise)
        bass_gain = 1.0
        mid_gain = 1.0
        treble_gain = 1.0

        filtered_audio, bass_filtered, mid_filtered, treble_filtered = fir.apply_filters(audio_data, bass_gain, mid_gain, treble_gain)

        self.assertEqual(filtered_audio.shape, audio_data.shape)  # Ensure the output has the same shape as the input
        self.assertEqual(bass_filtered.shape, audio_data[:, 0].shape)  # Check individual channel filtering
        self.assertEqual(mid_filtered.shape, audio_data[:, 0].shape)
        self.assertEqual(treble_filtered.shape, audio_data[:, 0].shape)

    def test_apply_filters_unsupported_audio_dim(self):
        """
        Test the application of FIR filters on unsupported audio data dimensionality.
        Case: Unsupported audio data dimensionality
        Type: Error Handling
        Input: 3D array audio data
        Expected Output: Raise ValueError
        Output: ValueError
        """
        audio_data = np.random.randn(44100, 2, 2)  # 3D array, which is not supported
        with self.assertRaises(ValueError):
            fir.apply_filters(audio_data, 1.0, 1.0, 1.0)

    def test_apply_filters_single_sample(self):
        """
        Test the application of FIR filters on audio data with a single sample.
        Case: Single sample audio data
        Type: Functionality
        Input: single sample audio data, bass_gain, mid_gain, treble_gain
        Expected Output: Filtered audio and individual band components of the same shape as input audio
        Output: Filtered audio, bass_filtered, mid_filtered, treble_filtered
        """
        audio_data = np.random.randn(1)  # Single sample of audio
        bass_gain = 1.0
        mid_gain = 1.0
        treble_gain = 1.0

        filtered_audio, bass_filtered, mid_filtered, treble_filtered = fir.apply_filters(audio_data, bass_gain, mid_gain, treble_gain)

        self.assertEqual(filtered_audio.shape, (1,))  # The output should still be a single sample
        self.assertEqual(bass_filtered.shape, (1,))
        self.assertEqual(mid_filtered.shape, (1,))
        self.assertEqual(treble_filtered.shape, (1,))

    def test_apply_filters_zero_length_audio(self):
        """
        Test the application of FIR filters on zero-length audio input.
        Case: Zero-length audio input
        Type: Functionality
        Input: zero-length audio data, bass_gain, mid_gain, treble_gain
        Expected Output: Filtered audio and individual band components of the same shape as input audio
        Output: Filtered audio, bass_filtered, mid_filtered, treble_filtered
        """
        audio_data = np.array([])  # Empty audio input
        bass_gain = 1.0
        mid_gain = 1.0
        treble_gain = 1.0

        filtered_audio, bass_filtered, mid_filtered, treble_filtered = fir.apply_filters(audio_data, bass_gain, mid_gain, treble_gain)

        self.assertEqual(filtered_audio.shape, (0,))  # The output should still be empty
        self.assertEqual(bass_filtered.shape, (0,))
        self.assertEqual(mid_filtered.shape, (0,))
        self.assertEqual(treble_filtered.shape, (0,))

    def test_create_fir_filter_high_sample_rate(self):
        """
        Test the creation of a filter with a high sampling rate (192kHz).
        Case: High sampling rate filter creation
        Type: Functionality
        Input: mid_gain, mid_band, sample_rate=192000, pass_zero='bandpass'
        Expected Output: Filter coefficients of length 101
        Output: Filter coefficients
        """
        high_sample_rate = 192000
        filter_coefs = fir.create_fir_filter(self.mid_gain, self.mid_band, sample_rate=high_sample_rate, pass_zero='bandpass')
        self.assertEqual(len(filter_coefs), 101)  # Ensure the number of taps is unchanged

if __name__ == "__main__":
    unittest.main()
