"""
TD-PSOLA Demonstration Script
This script demonstrates the TD-PSOLA implementation with:
- Pitch modification
- Duration modification
- Spectrograms visualization
- F0 curves plotting
- Performance measurement
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import time
from td_psola import TDPSOLA
import os


def plot_spectrogram(signal, sr, title="Spectrogram", ax=None):
    """
    Plot spectrogram of a signal.
    
    Args:
        signal: Audio signal
        sr: Sampling rate
        title: Plot title
        ax: Matplotlib axis (if None, creates new figure)
    """
    if ax is None:
        plt.figure(figsize=(10, 4))
        ax = plt.gca()
    
    # Compute spectrogram
    D = librosa.stft(signal)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Plot
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title(title)
    ax.set_ylim([0, 4000])  # Focus on speech range
    
    return img


def plot_f0_curve(times, f0, title="F0 Curve", ax=None, color='blue', label=None):
    """
    Plot F0 (fundamental frequency) curve.
    
    Args:
        times: Time points
        f0: F0 values in Hz
        title: Plot title
        ax: Matplotlib axis
        color: Line color
        label: Line label for legend
    """
    if ax is None:
        plt.figure(figsize=(10, 4))
        ax = plt.gca()
    
    ax.plot(times, f0, color=color, linewidth=2, label=label)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('F0 (Hz)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if label:
        ax.legend()


def demonstrate_pitch_modification(input_file, output_dir="output"):
    """
    Demonstrate pitch modification with visualization.
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save outputs
    """
    print("=" * 60)
    print("PITCH MODIFICATION DEMONSTRATION")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    print(f"\nLoading audio file: {input_file}")
    signal, sr = librosa.load(input_file, sr=16000)
    duration = len(signal) / sr
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sampling rate: {sr} Hz")
    
    # Initialize PSOLA
    psola = TDPSOLA(sr=sr)
    
    # Estimate original F0
    print("\nEstimating original F0...")
    f0_time_orig, f0_orig = psola.estimate_f0(signal)
    mean_f0_orig = np.nanmean(f0_orig)
    print(f"Mean F0: {mean_f0_orig:.2f} Hz")
    
    # Pitch shift factor (1.2 = 20% higher pitch)
    pitch_shift = 1.2
    print(f"\nApplying pitch shift: {pitch_shift}x ({(pitch_shift-1)*100:+.1f}%)")
    
    # Measure processing time
    start_time = time.time()
    modified_signal = psola.modify_pitch(signal, pitch_shift=pitch_shift)
    processing_time = time.time() - start_time
    
    print(f"Processing time: {processing_time:.3f} seconds")
    print(f"Real-time factor: {processing_time/duration:.3f}x")
    
    if processing_time < duration:
        print("✓ Faster than real-time")
    else:
        print("✗ Slower than real-time")
    
    # Estimate modified F0
    print("\nEstimating modified F0...")
    f0_time_mod, f0_mod = psola.estimate_f0(modified_signal)
    mean_f0_mod = np.nanmean(f0_mod)
    print(f"Mean F0: {mean_f0_mod:.2f} Hz")
    print(f"Actual pitch shift: {mean_f0_mod/mean_f0_orig:.3f}x")
    
    # Save output audio
    output_file = os.path.join(output_dir, "pitch_shifted.wav")
    sf.write(output_file, modified_signal, sr)
    print(f"\nOutput saved to: {output_file}")
    
    # Create visualization figure
    print("\nGenerating visualizations...")
    fig = plt.figure(figsize=(15, 10))
    
    # Spectrograms
    ax1 = plt.subplot(2, 2, 1)
    plot_spectrogram(signal, sr, "Original Signal Spectrogram", ax=ax1)
    
    ax2 = plt.subplot(2, 2, 2)
    plot_spectrogram(modified_signal, sr, f"Pitch Shifted ({pitch_shift}x) Spectrogram", ax=ax2)
    
    # F0 curves
    ax3 = plt.subplot(2, 2, 3)
    plot_f0_curve(f0_time_orig, f0_orig, "F0 Curves Comparison", ax=ax3, 
                  color='blue', label='Original')
    plot_f0_curve(f0_time_mod, f0_mod, ax=ax3, color='red', label='Modified')
    
    # Waveforms
    ax4 = plt.subplot(2, 2, 4)
    time_orig = np.arange(len(signal)) / sr
    time_mod = np.arange(len(modified_signal)) / sr
    ax4.plot(time_orig[:min(8000, len(signal))], signal[:min(8000, len(signal))], 
             label='Original', alpha=0.7)
    ax4.plot(time_mod[:min(8000, len(modified_signal))], 
             modified_signal[:min(8000, len(modified_signal))], 
             label='Modified', alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Waveform Comparison (first 0.5s)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_file = os.path.join(output_dir, "pitch_modification_analysis.png")
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {fig_file}")
    plt.close()
    
    print("\n" + "=" * 60)


def demonstrate_duration_modification(input_file, output_dir="output"):
    """
    Demonstrate duration modification with visualization.
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save outputs
    """
    print("=" * 60)
    print("DURATION MODIFICATION DEMONSTRATION")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    print(f"\nLoading audio file: {input_file}")
    signal, sr = librosa.load(input_file, sr=16000)
    duration = len(signal) / sr
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sampling rate: {sr} Hz")
    
    # Initialize PSOLA
    psola = TDPSOLA(sr=sr)
    
    # Estimate original F0
    print("\nEstimating original F0...")
    f0_time_orig, f0_orig = psola.estimate_f0(signal)
    mean_f0_orig = np.nanmean(f0_orig)
    print(f"Mean F0: {mean_f0_orig:.2f} Hz")
    
    # Time stretch factor (1.5 = 50% longer)
    time_stretch = 1.5
    print(f"\nApplying time stretch: {time_stretch}x ({(time_stretch-1)*100:+.1f}%)")
    
    # Measure processing time
    start_time = time.time()
    modified_signal = psola.modify_duration(signal, time_stretch=time_stretch)
    processing_time = time.time() - start_time
    
    print(f"Processing time: {processing_time:.3f} seconds")
    print(f"Real-time factor: {processing_time/duration:.3f}x")
    
    if processing_time < duration:
        print("✓ Faster than real-time")
    else:
        print("✗ Slower than real-time")
    
    # Check duration change
    modified_duration = len(modified_signal) / sr
    print(f"\nOriginal duration: {duration:.2f} seconds")
    print(f"Modified duration: {modified_duration:.2f} seconds")
    print(f"Actual time stretch: {modified_duration/duration:.3f}x")
    
    # Estimate modified F0
    print("\nEstimating modified F0...")
    f0_time_mod, f0_mod = psola.estimate_f0(modified_signal)
    mean_f0_mod = np.nanmean(f0_mod)
    print(f"Mean F0: {mean_f0_mod:.2f} Hz")
    print(f"F0 preservation: {abs(mean_f0_mod - mean_f0_orig):.2f} Hz difference")
    
    # Save output audio
    output_file = os.path.join(output_dir, "time_stretched.wav")
    sf.write(output_file, modified_signal, sr)
    print(f"\nOutput saved to: {output_file}")
    
    # Create visualization figure
    print("\nGenerating visualizations...")
    fig = plt.figure(figsize=(15, 10))
    
    # Spectrograms
    ax1 = plt.subplot(2, 2, 1)
    plot_spectrogram(signal, sr, "Original Signal Spectrogram", ax=ax1)
    
    ax2 = plt.subplot(2, 2, 2)
    plot_spectrogram(modified_signal, sr, f"Time Stretched ({time_stretch}x) Spectrogram", ax=ax2)
    
    # F0 curves
    ax3 = plt.subplot(2, 2, 3)
    plot_f0_curve(f0_time_orig, f0_orig, "F0 Curves Comparison", ax=ax3, 
                  color='blue', label='Original')
    plot_f0_curve(f0_time_mod, f0_mod, ax=ax3, color='green', label='Modified')
    
    # Waveforms
    ax4 = plt.subplot(2, 2, 4)
    time_orig = np.arange(len(signal)) / sr
    time_mod = np.arange(len(modified_signal)) / sr
    max_samples = min(16000, len(signal))  # Show first 1 second
    ax4.plot(time_orig[:max_samples], signal[:max_samples], 
             label='Original', alpha=0.7)
    max_samples_mod = min(int(16000 * time_stretch), len(modified_signal))
    ax4.plot(time_mod[:max_samples_mod], modified_signal[:max_samples_mod], 
             label='Modified', alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Waveform Comparison (first 1s)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_file = os.path.join(output_dir, "duration_modification_analysis.png")
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {fig_file}")
    plt.close()
    
    print("\n" + "=" * 60)


def measure_performance_10s(input_file, output_dir="output"):
    """
    Measure processing time for a 10-second audio file.
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save outputs
    """
    print("=" * 60)
    print("PERFORMANCE MEASUREMENT (10-SECOND AUDIO)")
    print("=" * 60)
    
    # Load audio
    print(f"\nLoading audio file: {input_file}")
    signal, sr = librosa.load(input_file, sr=16000)
    duration = len(signal) / sr
    
    # If audio is shorter than 10 seconds, repeat it
    if duration < 10:
        print(f"Audio is {duration:.2f}s, repeating to reach ~10 seconds...")
        repeats = int(np.ceil(10 / duration))
        signal = np.tile(signal, repeats)
        duration = len(signal) / sr
    
    # If audio is longer than 10 seconds, trim it
    if duration > 10:
        signal = signal[:int(10 * sr)]
        duration = 10.0
    
    print(f"Processing audio of {duration:.2f} seconds")
    
    # Initialize PSOLA
    psola = TDPSOLA(sr=sr)
    
    # Test 1: Pitch modification
    print("\n" + "-" * 60)
    print("Test 1: Pitch Modification (1.3x)")
    print("-" * 60)
    
    start_time = time.time()
    pitch_modified = psola.modify_pitch(signal, pitch_shift=1.3)
    pitch_time = time.time() - start_time
    
    print(f"Processing time: {pitch_time:.3f} seconds")
    print(f"Real-time factor: {pitch_time/duration:.3f}x")
    
    if pitch_time < duration:
        print("✓ Performance: FASTER than real-time")
        print(f"  Can process {duration/pitch_time:.2f}x real-time speed")
    elif pitch_time < duration * 2:
        print("⚠ Performance: Acceptable for offline processing")
    else:
        print("✗ Performance: SLOW, needs optimization")
    
    # Save output
    output_file = os.path.join(output_dir, "10s_pitch_modified.wav")
    sf.write(output_file, pitch_modified, sr)
    
    # Test 2: Duration modification
    print("\n" + "-" * 60)
    print("Test 2: Duration Modification (0.7x)")
    print("-" * 60)
    
    start_time = time.time()
    duration_modified = psola.modify_duration(signal, time_stretch=0.7)
    duration_time = time.time() - start_time
    
    print(f"Processing time: {duration_time:.3f} seconds")
    print(f"Real-time factor: {duration_time/duration:.3f}x")
    
    if duration_time < duration:
        print("✓ Performance: FASTER than real-time")
        print(f"  Can process {duration/duration_time:.2f}x real-time speed")
    elif duration_time < duration * 2:
        print("⚠ Performance: Acceptable for offline processing")
    else:
        print("✗ Performance: SLOW, needs optimization")
    
    # Save output
    output_file = os.path.join(output_dir, "10s_duration_modified.wav")
    sf.write(output_file, duration_modified, sr)
    
    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Pitch modification: {pitch_time:.3f}s ({pitch_time/duration:.3f}x real-time)")
    print(f"Duration modification: {duration_time:.3f}s ({duration_time/duration:.3f}x real-time)")
    print(f"\nAverage processing time: {(pitch_time + duration_time)/2:.3f}s")
    print(f"Average real-time factor: {((pitch_time + duration_time)/2)/duration:.3f}x")
    
    print("\n" + "=" * 60)


def main():
    """
    Main demonstration function.
    """
    print("\n" + "=" * 60)
    print("TD-PSOLA SPEECH MANIPULATION SYSTEM")
    print("=" * 60)
    print("\nThis demonstration requires an input audio file.")
    print("Place your audio file (16kHz, mono) in the same directory")
    print("or provide the path below.")
    
    # Check for sample audio files
    possible_files = ["input.wav", "speech.wav", "test.wav", "sample.wav"]
    input_file = None
    
    for filename in possible_files:
        if os.path.exists(filename):
            input_file = filename
            print(f"\n✓ Found audio file: {filename}")
            break
    
    if input_file is None:
        print("\n⚠ No audio file found. Please provide a path:")
        input_file = input("Enter path to audio file: ").strip()
        
        if not os.path.exists(input_file):
            print(f"\n✗ Error: File not found: {input_file}")
            print("\nCreating a synthetic speech-like signal for demonstration...")
            
            # Create synthetic signal
            sr = 16000
            duration = 3
            t = np.linspace(0, duration, int(sr * duration))
            
            # Generate speech-like signal with varying F0
            f0 = 150 + 30 * np.sin(2 * np.pi * 0.5 * t)  # Varying F0
            signal = np.zeros_like(t)
            
            for i in range(1, 6):  # Harmonics
                signal += (1/i) * np.sin(2 * np.pi * i * f0 * t)
            
            # Add formant-like filtering
            from scipy.signal import lfilter, butter
            b, a = butter(4, [300, 3400], btype='band', fs=sr)
            signal = lfilter(b, a, signal)
            
            # Normalize
            signal = signal / np.max(np.abs(signal)) * 0.5
            
            # Save synthetic signal
            input_file = "synthetic_speech.wav"
            sf.write(input_file, signal, sr)
            print(f"✓ Saved synthetic signal to: {input_file}")
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run demonstrations
    try:
        # 1. Pitch modification
        demonstrate_pitch_modification(input_file, output_dir)
        
        print("\nPress Enter to continue to duration modification...")
        input()
        
        # 2. Duration modification
        demonstrate_duration_modification(input_file, output_dir)
        
        print("\nPress Enter to continue to performance measurement...")
        input()
        
        # 3. Performance measurement
        measure_performance_10s(input_file, output_dir)
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print(f"\nAll outputs saved to: {output_dir}/")
        print("\nGenerated files:")
        for file in os.listdir(output_dir):
            print(f"  - {file}")
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
