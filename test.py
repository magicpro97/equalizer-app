import unittest
import time
import cProfile
import memory_profiler
from unittest import skip
import numpy as np
from fir import apply_bandpass_filter, apply_filters
from LabUtils import gen_continous_signal, sampling, min_max_norm, z_score_norm, rms_norm, max_abs_norm, normalizing_signal
import psutil
from concurrent.futures import ThreadPoolExecutor
import threading

class TestFIR(unittest.TestCase):
    def test_create_fir_filter(self):
        # Test case: FIR filter creation with standard parameters
        # Type: Unit Test
        # Input: Random audio data (44100 samples), band=[20, 250], gain=1.0
        # Expected Output: Filter coefficients matching input length, all values finite
        filter_coefs = apply_bandpass_filter(np.random.randn(44100), 20, 250, 1.0)
        self.assertEqual(len(filter_coefs), 44100)
        self.assertTrue(np.all(np.isfinite(filter_coefs)))
    
    def test_apply_filters(self):
        # Test case: Filter application with unity gains
        # Type: Unit Test
        # Input: Random audio data (44100 samples), all gains=1.0
        # Expected Output: Three filtered signals matching input length
        audio_data = np.random.randn(44100)
        bass_gain, mid_gain, treble_gain = 1.0, 1.0, 1.0
        _, bass_filtered, mid_filtered, treble_filtered = apply_filters(audio_data, bass_gain, mid_gain, treble_gain)
        self.assertEqual(len(bass_filtered), len(audio_data))
        self.assertEqual(len(mid_filtered), len(audio_data))
        self.assertEqual(len(treble_filtered), len(audio_data))

    def test_create_fir_filter_large_numtaps(self):
        # Test case: FIR filter creation with increased complexity
        # Type: Boundary Test
        # Input: Random audio data, band=[20, 250], gain=1.0, numtaps=1000
        # Expected Output: Filter coefficients matching input length, stable response
        filter_coefs = apply_bandpass_filter(np.random.randn(44100), 20, 250, 1.0, numtaps=1000)
        self.assertEqual(len(filter_coefs), 44100)
        self.assertTrue(np.all(np.isfinite(filter_coefs)))

    def test_apply_filters_large_gain(self):
        # Test case: High-gain filter application
        # Type: Boundary Test
        # Input: Random audio data, all gains=100.0
        # Expected Output: Linearly scaled outputs, preserved signal integrity
        audio_data = np.random.randn(44100)
        
        # Test with normal gain first (baseline)
        _, bass_normal, mid_normal, treble_normal = apply_filters(audio_data, 1.0, 1.0, 1.0)
        
        # Test with large gain
        bass_gain, mid_gain, treble_gain = 100.0, 100.0, 100.0
        _, bass_filtered, mid_filtered, treble_filtered = apply_filters(audio_data, bass_gain, mid_gain, treble_gain)
        
        # Check outputs are finite
        self.assertTrue(np.all(np.isfinite(bass_filtered)))
        self.assertTrue(np.all(np.isfinite(mid_filtered)))
        self.assertTrue(np.all(np.isfinite(treble_filtered)))
        
        # Check scaling is approximately proportional (allowing for some numerical error)
        self.assertTrue(np.allclose(bass_filtered, bass_normal * bass_gain, rtol=1e-5))
        self.assertTrue(np.allclose(mid_filtered, mid_normal * mid_gain, rtol=1e-5))
        self.assertTrue(np.allclose(treble_filtered, treble_normal * treble_gain, rtol=1e-5))
        
        # Check signals aren't completely saturated (should have some variation)
        self.assertGreater(np.std(bass_filtered), 0)
        self.assertGreater(np.std(mid_filtered), 0)
        self.assertGreater(np.std(treble_filtered), 0)

    def test_apply_filters_edge_case(self):
        # Test case: Complete signal attenuation
        # Type: Edge Case
        # Input: Random audio data, all gains=0.0
        # Expected Output: Zero-valued outputs across all bands
        audio_data = np.random.randn(44100)
        bass_gain, mid_gain, treble_gain = 0.0, 0.0, 0.0
        output, bass_filtered, mid_filtered, treble_filtered = apply_filters(audio_data, bass_gain, mid_gain, treble_gain)
        
        # Check that lengths are preserved
        self.assertEqual(len(bass_filtered), len(audio_data))
        self.assertEqual(len(mid_filtered), len(audio_data))
        self.assertEqual(len(treble_filtered), len(audio_data))
        
        # Check that the signals are significantly attenuated
        self.assertTrue(np.allclose(bass_filtered, 0, atol=1e-10))
        self.assertTrue(np.allclose(mid_filtered, 0, atol=1e-10))
        self.assertTrue(np.allclose(treble_filtered, 0, atol=1e-10))
        
        # Check that the final output is also attenuated
        self.assertTrue(np.allclose(output, 0, atol=1e-10))

    def test_apply_filters_empty_audio(self):
        # Test case: Processing empty audio stream
        # Type: Edge Case
        # Input: Empty numpy array, unity gains
        # Expected Output: Empty arrays for all filter outputs
        audio_data = np.array([])
        bass_gain, mid_gain, treble_gain = 1.0, 1.0, 1.0
        _, bass_filtered, mid_filtered, treble_filtered = apply_filters(audio_data, bass_gain, mid_gain, treble_gain)
        self.assertEqual(len(bass_filtered), 0)
        self.assertEqual(len(mid_filtered), 0)
        self.assertEqual(len(treble_filtered), 0)

    def test_apply_filters_negative_gain(self):
        # Test case: Phase inversion verification
        # Type: Edge Case
        # Input: Random audio data, all gains=-1.0
        # Expected Output: Phase-inverted signals matching positive gain outputs
        audio_data = np.random.randn(44100)
        
        # Get outputs with positive gain
        _, bass_pos, mid_pos, treble_pos = apply_filters(audio_data, 1.0, 1.0, 1.0)
        
        # Get outputs with negative gain
        _, bass_neg, mid_neg, treble_neg = apply_filters(audio_data, -1.0, -1.0, -1.0)
        
        # Check that outputs are finite
        self.assertTrue(np.all(np.isfinite(bass_neg)))
        self.assertTrue(np.all(np.isfinite(mid_neg)))
        self.assertTrue(np.all(np.isfinite(treble_neg)))
        
        # Check that negative gain results in phase inversion (approximately equal to positive gain * -1)
        self.assertTrue(np.allclose(bass_neg, -bass_pos, rtol=1e-10))
        self.assertTrue(np.allclose(mid_neg, -mid_pos, rtol=1e-10))
        self.assertTrue(np.allclose(treble_neg, -treble_pos, rtol=1e-10))
    
    def test_create_fir_filter_zero_gain(self):
        # Test case: Zero gain
        # Type: Edge Case
        # Input: gain=0.0, band=[20, 250]
        # Expected Output: Filter coefficients all zero
        filter_coefs = apply_bandpass_filter(np.random.randn(44100), 20, 250, 0.0)
        self.assertTrue(np.all(filter_coefs == 0))

    def test_create_fir_filter_invalid_band(self):
        # Test case: Invalid band range
        # Type: Error Handling
        # Input: gain=1.0, band=[250, 20]
        # Expected Output: Raise ValueError
        with self.assertRaises(ValueError):
            apply_bandpass_filter(np.random.randn(44100), 250, 20, 1.0)

    def test_create_fir_filter_null_band(self):
        # Test case: Null band
        # Type: Error Handling
        # Input: gain=1.0, band=None
        # Expected Output: Raise TypeError
        with self.assertRaises(TypeError):
            apply_bandpass_filter(np.random.randn(44100), None, None, 1.0)

    def test_create_fir_filter_negative_numtaps(self):
        # Test case: Negative number of taps
        # Type: Error Handling
        # Input: gain=1.0, band=[20, 250], numtaps=-10
        # Expected Output: Raise ValueError
        with self.assertRaises(ValueError):
            apply_bandpass_filter(np.random.randn(44100), 20, 250, 1.0, numtaps=-10)

    def test_apply_filters_null_audio(self):
        # Test case: Null audio data
        # Type: Error Handling
        # Input: None, gains=1.0 for all bands
        # Expected Output: Raise TypeError
        with self.assertRaises(TypeError):
            apply_filters(None, 1.0, 1.0, 1.0)

    def test_apply_filters_mismatched_audio_dimensions(self):
        # Test case: Mismatched audio dimensions
        # Type: Error Handling
        # Expected Output: Raise ValueError for non-1D arrays
        
        # Test 2D stereo data
        stereo_data = np.random.randn(2, 44100)
        with self.assertRaises(ValueError):
            apply_filters(stereo_data, 1.0, 1.0, 1.0)
        
        # Test 3D data
        three_d_data = np.random.randn(2, 44100, 2)
        with self.assertRaises(ValueError):
            apply_filters(three_d_data, 1.0, 1.0, 1.0)

    def test_apply_filters_audio_dimensions(self):
        # Test case: Audio dimension handling
        # Type: Error Handling
        # Input: Various audio data formats (mono, stereo, invalid dimensions)
        # Expected Output: 
        #   - Accept mono (1D) and stereo (2D with samples as rows)
        #   - Reject wrong stereo orientation and 3D data
        bass_gain, mid_gain, treble_gain = 1.0, 1.0, 1.0
        
        # Test invalid cases
        # Wrong orientation (channels as rows instead of columns)
        wrong_stereo = np.random.randn(2, 44100)  # channels should be columns, not rows
        with self.assertRaises(ValueError):
            apply_filters(wrong_stereo, bass_gain, mid_gain, treble_gain)
        
        # 3D data is not allowed
        three_d_data = np.random.randn(44100, 2, 2)
        with self.assertRaises(ValueError):
            apply_filters(three_d_data, bass_gain, mid_gain, treble_gain)
        
        # Valid cases should not raise exceptions
        mono_data = np.random.randn(44100)  # 1D mono
        correct_stereo = np.random.randn(44100, 2)  # 2D stereo with channels as columns
        
        try:
            apply_filters(mono_data, bass_gain, mid_gain, treble_gain)
            apply_filters(correct_stereo, bass_gain, mid_gain, treble_gain)
        except ValueError:
            self.fail("apply_filters raised ValueError unexpectedly for valid input dimensions")
    
    def test_apply_filters_invalid_gains(self):
        # Test case: Invalid gain values
        # Type: Error Handling
        # Input: Audio data with infinite or NaN gain values
        # Expected Output: Raise ValueError
        audio_data = np.random.randn(44100)
        
        # Test with infinite gain
        with self.assertRaises(ValueError):
            apply_filters(audio_data, np.inf, 1.0, 1.0)
        
        # Test with NaN gain
        with self.assertRaises(ValueError):
            apply_filters(audio_data, 1.0, np.nan, 1.0)
        
        # Test with negative infinity
        with self.assertRaises(ValueError):
            apply_filters(audio_data, 1.0, 1.0, -np.inf)

class TestPerformance(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.audio_data = np.random.randn(44100)
        self.large_audio_data = np.random.randn(441000)  # 10x larger
        self.max_time = 0.1

    def test_apply_filters_execution_time(self):
        # Test case: Measure execution time
        # Type: Performance
        # Input: Random audio data (44100 samples, ~1 second at 44.1kHz)
        # Expected Output: Function executes within 100ms threshold
        
        # Run multiple times and take average to account for system variations
        num_runs = 5
        times = []
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _, bass, mid, treble = apply_filters(self.audio_data, 1.0, 1.0, 1.0)
            times.append(time.perf_counter() - start_time)
        
        avg_time = sum(times) / len(times)
        self.assertLess(avg_time, self.max_time, 
            f"Average execution time {avg_time:.3f}s exceeds threshold of {self.max_time}s")

    def test_apply_filters_memory_usage(self):
        # Test case: Memory consumption measurement
        # Type: Performance
        # Input: Large random audio data (441000 samples, ~10 seconds at 44.1kHz)
        # Expected Output: Memory increase less than 100MB from baseline
        baseline_mem = memory_profiler.memory_usage()[0]
        
        @memory_profiler.profile
        def measure_memory():
            _, bass, mid, treble = apply_filters(self.large_audio_data, 1.0, 1.0, 1.0)
            peak_mem = max(memory_profiler.memory_usage())
            return peak_mem - baseline_mem
        
        mem_increase = measure_memory()
        self.assertLess(mem_increase, 100)  # MB

    def test_apply_filters_scaling(self):
        # Test case: Performance scaling with input size
        # Type: Performance
        # Input: Two test signals:
        #   - Small: 44100 samples (~1 second at 44.1kHz)
        #   - Large: 441000 samples (~10 seconds at 44.1kHz)
        # Expected Output: Processing time scales linearly (scaling factor < 15x)
        
        # Time with small input
        start_time = time.perf_counter()
        apply_filters(self.audio_data, 1.0, 1.0, 1.0)
        small_time = time.perf_counter() - start_time
        
        # Time with large input (10x)
        start_time = time.perf_counter()
        apply_filters(self.large_audio_data, 1.0, 1.0, 1.0)
        large_time = time.perf_counter() - start_time
        
        scaling_factor = large_time / small_time
        self.assertLess(scaling_factor, 15, 
            f"Performance scaling factor {scaling_factor:.2f} exceeds linear growth")

    def test_apply_filters_cpu_load(self):
        # Test case: CPU utilization measurement
        # Type: Performance
        # Input: Large random audio data (441000 samples, ~10 seconds at 44.1kHz)
        # Expected Output: CPU usage increase less than 80% from baseline
        baseline_cpu = psutil.cpu_percent(interval=0.1)
        _, bass, mid, treble = apply_filters(self.large_audio_data, 1.0, 1.0, 1.0)
        peak_cpu = psutil.cpu_percent(interval=0.1)
        
        self.assertLess(peak_cpu - baseline_cpu, 80)  # Should not use more than 80% CPU

    def test_apply_filters_concurrent(self):
        # Test case: Concurrent execution stability
        # Type: Performance
        # Input: 4 concurrent processes processing random audio data (44100 samples each)
        # Expected Output: All processes complete successfully without errors
        def concurrent_process():
            return apply_filters(self.audio_data, 1.0, 1.0, 1.0)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(concurrent_process) for _ in range(4)]
            results = [f.result() for f in futures]

    def test_apply_filters_worst_case(self):
        # Test case: Worst-case performance scenario
        # Type: Performance
        # Input: Rapidly oscillating signal (1000π frequency, 44100 samples)
        # Expected Output: Processing time less than 2x normal threshold
        worst_case_signal = np.sin(np.linspace(0, 1000*np.pi, 44100))
        
        start_time = time.perf_counter()
        _, bass, mid, treble = apply_filters(worst_case_signal, 1.0, 1.0, 1.0)
        execution_time = time.perf_counter() - start_time
        
        self.assertLess(execution_time, self.max_time * 2)  # Allow double time for worst case

    def test_apply_filters_profiling(self):
        # Test case: Detailed performance profiling
        # Type: Performance
        # Input: Random audio data (44100 samples) processed 100 times
        # Expected Output: Generated profile data for performance analysis
        profiler = cProfile.Profile()
        profiler.enable()
        
        for _ in range(100):
            apply_filters(self.audio_data, 1.0, 1.0, 1.0)
        
        profiler.disable()
        # profiler.print_stats(sort='cumulative')

if __name__ == '__main__':
    unittest.main()
