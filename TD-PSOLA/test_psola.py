"""
Simple test script for TD-PSOLA implementation.
Tests basic functionality without requiring audio files.
"""

import numpy as np
from td_psola import TDPSOLA
import soundfile as sf


def generate_synthetic_speech(duration=3.0, sr=16000):
    """
    Generate a synthetic speech-like signal for testing.
    
    Args:
        duration: Duration in seconds
        sr: Sampling rate
        
    Returns:
        signal: Synthetic speech signal
    """
    t = np.linspace(0, duration, int(sr * duration))
    
    # Generate varying F0 (fundamental frequency)
    f0 = 150 + 30 * np.sin(2 * np.pi * 0.5 * t)  # F0 varies between 120-180 Hz
    
    # Generate harmonic signal
    signal = np.zeros_like(t)
    for i in range(1, 8):  # 7 harmonics
        amplitude = 1.0 / i  # Decreasing amplitude for higher harmonics
        signal += amplitude * np.sin(2 * np.pi * i * f0 * t)
    
    # Add formant-like resonances using simple filtering
    from scipy.signal import lfilter, butter
    
    # Bandpass filter to simulate formants
    b, a = butter(4, [300, 3400], btype='band', fs=sr)
    signal = lfilter(b, a, signal)
    
    # Add some amplitude modulation
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    signal = signal * envelope
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.5
    
    return signal


def test_pitch_modification():
    """Test pitch modification functionality."""
    print("=" * 60)
    print("TEST 1: Pitch Modification")
    print("=" * 60)
    
    # Generate test signal
    sr = 16000
    signal = generate_synthetic_speech(duration=2.0, sr=sr)
    
    # Initialize PSOLA
    psola = TDPSOLA(sr=sr)
    
    # Test different pitch shifts
    test_cases = [0.8, 1.0, 1.2, 1.5]
    
    for pitch_shift in test_cases:
        print(f"\nTesting pitch shift: {pitch_shift}x ({(pitch_shift-1)*100:+.1f}%)")
        
        try:
            # Apply pitch modification
            modified = psola.modify_pitch(signal, pitch_shift=pitch_shift)
            
            # Check output
            if len(modified) > 0:
                print(f"✓ Success! Output length: {len(modified)} samples")
                print(f"  Output duration: {len(modified)/sr:.2f}s")
                
                # Save output
                output_file = f"test_pitch_{pitch_shift:.1f}x.wav"
                sf.write(output_file, modified, sr)
                print(f"  Saved to: {output_file}")
            else:
                print("✗ Failed: Empty output")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)


def test_duration_modification():
    """Test duration modification functionality."""
    print("=" * 60)
    print("TEST 2: Duration Modification")
    print("=" * 60)
    
    # Generate test signal
    sr = 16000
    signal = generate_synthetic_speech(duration=2.0, sr=sr)
    original_duration = len(signal) / sr
    
    # Initialize PSOLA
    psola = TDPSOLA(sr=sr)
    
    # Test different time stretches
    test_cases = [0.5, 0.7, 1.0, 1.5, 2.0]
    
    for time_stretch in test_cases:
        print(f"\nTesting time stretch: {time_stretch}x ({(time_stretch-1)*100:+.1f}%)")
        
        try:
            # Apply duration modification
            modified = psola.modify_duration(signal, time_stretch=time_stretch)
            
            # Check output
            if len(modified) > 0:
                modified_duration = len(modified) / sr
                actual_stretch = modified_duration / original_duration
                
                print(f"✓ Success! Output length: {len(modified)} samples")
                print(f"  Original duration: {original_duration:.2f}s")
                print(f"  Modified duration: {modified_duration:.2f}s")
                print(f"  Actual stretch factor: {actual_stretch:.2f}x")
                
                # Save output
                output_file = f"test_duration_{time_stretch:.1f}x.wav"
                sf.write(output_file, modified, sr)
                print(f"  Saved to: {output_file}")
            else:
                print("✗ Failed: Empty output")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)


def test_f0_estimation():
    """Test F0 estimation functionality."""
    print("=" * 60)
    print("TEST 3: F0 Estimation")
    print("=" * 60)
    
    # Generate test signal
    sr = 16000
    signal = generate_synthetic_speech(duration=2.0, sr=sr)
    
    # Initialize PSOLA
    psola = TDPSOLA(sr=sr)
    
    print("\nEstimating F0...")
    
    try:
        # Estimate F0
        times, f0 = psola.estimate_f0(signal)
        
        # Check results
        print(f"✓ Success! Estimated {len(f0)} F0 values")
        print(f"  Time range: {times[0]:.2f}s to {times[-1]:.2f}s")
        print(f"  F0 range: {np.nanmin(f0):.1f} Hz to {np.nanmax(f0):.1f} Hz")
        print(f"  Mean F0: {np.nanmean(f0):.1f} Hz")
        print(f"  (Expected ~150 Hz for this synthetic signal)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)


def test_pitch_mark_detection():
    """Test pitch mark detection functionality."""
    print("=" * 60)
    print("TEST 4: Pitch Mark Detection")
    print("=" * 60)
    
    # Generate test signal
    sr = 16000
    signal = generate_synthetic_speech(duration=2.0, sr=sr)
    
    # Initialize PSOLA
    psola = TDPSOLA(sr=sr)
    
    print("\nDetecting pitch marks...")
    
    try:
        # Estimate F0 first
        f0_time, f0 = psola.estimate_f0(signal)
        
        # Detect pitch marks
        pitch_marks = psola.detect_pitch_marks(signal, f0_time, f0)
        
        # Check results
        print(f"✓ Success! Detected {len(pitch_marks)} pitch marks")
        
        if len(pitch_marks) > 1:
            # Calculate periods
            periods = np.diff(pitch_marks)
            period_time = periods / sr * 1000  # in milliseconds
            
            print(f"  Period range: {np.min(period_time):.2f} ms to {np.max(period_time):.2f} ms")
            print(f"  Mean period: {np.mean(period_time):.2f} ms")
            
            # Convert to F0
            f0_from_marks = 1000 / period_time
            print(f"  Implied F0 range: {np.min(f0_from_marks):.1f} Hz to {np.max(f0_from_marks):.1f} Hz")
            print(f"  Mean F0 from marks: {np.mean(f0_from_marks):.1f} Hz")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TD-PSOLA UNIT TESTS")
    print("=" * 60)
    print("\nThis script tests the TD-PSOLA implementation")
    print("using synthetic speech signals.\n")
    
    # Run tests
    test_f0_estimation()
    print("\nPress Enter to continue...")
    input()
    
    test_pitch_mark_detection()
    print("\nPress Enter to continue...")
    input()
    
    test_pitch_modification()
    print("\nPress Enter to continue...")
    input()
    
    test_duration_modification()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print("\nGenerated test files:")
    import os
    for file in os.listdir('.'):
        if file.startswith('test_') and file.endswith('.wav'):
            print(f"  - {file}")
    print("\nYou can listen to these files to verify the modifications.")


if __name__ == "__main__":
    main()
