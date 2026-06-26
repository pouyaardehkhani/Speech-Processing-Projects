"""
Generate Transformed Speech Samples
This script generates pitch-up, pitch-down, and time-stretch outputs
from the input audio file.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import time
from td_psola import TDPSOLA
import os


def generate_all_samples(input_file, output_dir="output"):
    """
    Generate all three transformed samples: pitch-up, pitch-down, and time-stretch.
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save outputs
    """
    print("=" * 70)
    print("GENERATING TRANSFORMED SPEECH SAMPLES")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    print(f"\nLoading audio file:")
    print(f"  {input_file}")
    
    if not os.path.exists(input_file):
        print(f"\n✗ Error: File not found: {input_file}")
        return
    
    signal, sr = librosa.load(input_file, sr=16000)
    duration = len(signal) / sr
    print(f"\n✓ Audio loaded successfully")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Sampling rate: {sr} Hz")
    print(f"  Samples: {len(signal)}")
    
    # Initialize PSOLA
    psola = TDPSOLA(sr=sr)
    
    # Estimate original F0
    print("\nAnalyzing original audio...")
    f0_time, f0 = psola.estimate_f0(signal)
    mean_f0 = np.nanmean(f0)
    print(f"  Mean F0: {mean_f0:.2f} Hz")
    
    # ========================================================================
    # 1. PITCH-UP (1.3x = 30% higher pitch)
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRANSFORMATION 1: PITCH-UP (1.3x higher)")
    print("=" * 70)
    
    pitch_up_factor = 1.3
    print(f"\nApplying pitch shift: {pitch_up_factor}x ({(pitch_up_factor-1)*100:+.1f}%)")
    
    start_time = time.time()
    pitch_up_signal = psola.modify_pitch(signal, pitch_shift=pitch_up_factor)
    processing_time = time.time() - start_time
    
    print(f"✓ Processing complete")
    print(f"  Processing time: {processing_time:.3f} seconds")
    print(f"  Real-time factor: {processing_time/duration:.3f}x")
    
    # Estimate F0 of modified signal
    f0_time_up, f0_up = psola.estimate_f0(pitch_up_signal)
    mean_f0_up = np.nanmean(f0_up)
    print(f"  Original mean F0: {mean_f0:.2f} Hz")
    print(f"  Modified mean F0: {mean_f0_up:.2f} Hz")
    print(f"  Actual pitch shift: {mean_f0_up/mean_f0:.3f}x")
    
    # Save output
    output_file_up = os.path.join(output_dir, "pitch_up.wav")
    sf.write(output_file_up, pitch_up_signal, sr)
    print(f"\n✓ Saved to: {output_file_up}")
    
    # ========================================================================
    # 2. PITCH-DOWN (0.75x = 25% lower pitch)
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRANSFORMATION 2: PITCH-DOWN (0.75x lower)")
    print("=" * 70)
    
    pitch_down_factor = 0.75
    print(f"\nApplying pitch shift: {pitch_down_factor}x ({(pitch_down_factor-1)*100:+.1f}%)")
    
    start_time = time.time()
    pitch_down_signal = psola.modify_pitch(signal, pitch_shift=pitch_down_factor)
    processing_time = time.time() - start_time
    
    print(f"✓ Processing complete")
    print(f"  Processing time: {processing_time:.3f} seconds")
    print(f"  Real-time factor: {processing_time/duration:.3f}x")
    
    # Estimate F0 of modified signal
    f0_time_down, f0_down = psola.estimate_f0(pitch_down_signal)
    mean_f0_down = np.nanmean(f0_down)
    print(f"  Original mean F0: {mean_f0:.2f} Hz")
    print(f"  Modified mean F0: {mean_f0_down:.2f} Hz")
    print(f"  Actual pitch shift: {mean_f0_down/mean_f0:.3f}x")
    
    # Save output
    output_file_down = os.path.join(output_dir, "pitch_down.wav")
    sf.write(output_file_down, pitch_down_signal, sr)
    print(f"\n✓ Saved to: {output_file_down}")
    
    # ========================================================================
    # 3. TIME-STRETCH (1.4x = 40% longer duration)
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRANSFORMATION 3: TIME-STRETCH (1.4x longer)")
    print("=" * 70)
    
    time_stretch_factor = 1.4
    print(f"\nApplying time stretch: {time_stretch_factor}x ({(time_stretch_factor-1)*100:+.1f}%)")
    
    start_time = time.time()
    time_stretched_signal = psola.modify_duration(signal, time_stretch=time_stretch_factor)
    processing_time = time.time() - start_time
    
    print(f"✓ Processing complete")
    print(f"  Processing time: {processing_time:.3f} seconds")
    print(f"  Real-time factor: {processing_time/duration:.3f}x")
    
    # Check duration change
    stretched_duration = len(time_stretched_signal) / sr
    print(f"  Original duration: {duration:.2f} seconds")
    print(f"  Modified duration: {stretched_duration:.2f} seconds")
    print(f"  Actual time stretch: {stretched_duration/duration:.3f}x")
    
    # Estimate F0 to verify pitch preservation
    f0_time_stretch, f0_stretch = psola.estimate_f0(time_stretched_signal)
    mean_f0_stretch = np.nanmean(f0_stretch)
    print(f"  Original mean F0: {mean_f0:.2f} Hz")
    print(f"  Modified mean F0: {mean_f0_stretch:.2f} Hz")
    print(f"  F0 preservation: {abs(mean_f0_stretch - mean_f0):.2f} Hz difference")
    
    # Save output
    output_file_stretch = os.path.join(output_dir, "time_stretched.wav")
    sf.write(output_file_stretch, time_stretched_signal, sr)
    print(f"\n✓ Saved to: {output_file_stretch}")
    
    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    print("\nCreating comparison plots...")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Spectrograms
    print("  - Spectrograms...")
    ax1 = plt.subplot(3, 4, 1)
    plot_spectrogram(signal, sr, "Original", ax1)
    
    ax2 = plt.subplot(3, 4, 2)
    plot_spectrogram(pitch_up_signal, sr, f"Pitch-Up ({pitch_up_factor}x)", ax2)
    
    ax3 = plt.subplot(3, 4, 3)
    plot_spectrogram(pitch_down_signal, sr, f"Pitch-Down ({pitch_down_factor}x)", ax3)
    
    ax4 = plt.subplot(3, 4, 4)
    plot_spectrogram(time_stretched_signal, sr, f"Time-Stretch ({time_stretch_factor}x)", ax4)
    
    # Row 2: F0 curves
    print("  - F0 curves...")
    ax5 = plt.subplot(3, 4, 5)
    ax5.plot(f0_time, f0, 'b-', linewidth=2, label='Original')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('F0 (Hz)')
    ax5.set_title('Original F0')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    ax6 = plt.subplot(3, 4, 6)
    ax6.plot(f0_time, f0, 'b-', linewidth=1, alpha=0.5, label='Original')
    ax6.plot(f0_time_up, f0_up, 'r-', linewidth=2, label='Pitch-Up')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('F0 (Hz)')
    ax6.set_title('Pitch-Up F0 Comparison')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    ax7 = plt.subplot(3, 4, 7)
    ax7.plot(f0_time, f0, 'b-', linewidth=1, alpha=0.5, label='Original')
    ax7.plot(f0_time_down, f0_down, 'g-', linewidth=2, label='Pitch-Down')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('F0 (Hz)')
    ax7.set_title('Pitch-Down F0 Comparison')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(f0_time, f0, 'b-', linewidth=1, alpha=0.5, label='Original')
    ax8.plot(f0_time_stretch, f0_stretch, 'm-', linewidth=2, label='Time-Stretch')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('F0 (Hz)')
    ax8.set_title('Time-Stretch F0 Comparison')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    # Row 3: Waveforms (zoom in on first 1 second)
    print("  - Waveforms...")
    plot_samples = min(16000, len(signal))
    time_orig = np.arange(plot_samples) / sr
    
    ax9 = plt.subplot(3, 4, 9)
    ax9.plot(time_orig, signal[:plot_samples], 'b-', linewidth=0.5)
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Amplitude')
    ax9.set_title('Original Waveform (first 1s)')
    ax9.grid(True, alpha=0.3)
    
    ax10 = plt.subplot(3, 4, 10)
    plot_samples_up = min(16000, len(pitch_up_signal))
    time_up = np.arange(plot_samples_up) / sr
    ax10.plot(time_up, pitch_up_signal[:plot_samples_up], 'r-', linewidth=0.5)
    ax10.set_xlabel('Time (s)')
    ax10.set_ylabel('Amplitude')
    ax10.set_title('Pitch-Up Waveform (first 1s)')
    ax10.grid(True, alpha=0.3)
    
    ax11 = plt.subplot(3, 4, 11)
    plot_samples_down = min(16000, len(pitch_down_signal))
    time_down = np.arange(plot_samples_down) / sr
    ax11.plot(time_down, pitch_down_signal[:plot_samples_down], 'g-', linewidth=0.5)
    ax11.set_xlabel('Time (s)')
    ax11.set_ylabel('Amplitude')
    ax11.set_title('Pitch-Down Waveform (first 1s)')
    ax11.grid(True, alpha=0.3)
    
    ax12 = plt.subplot(3, 4, 12)
    plot_samples_stretch = min(int(16000 * time_stretch_factor), len(time_stretched_signal))
    time_stretch_plot = np.arange(plot_samples_stretch) / sr
    ax12.plot(time_stretch_plot, time_stretched_signal[:plot_samples_stretch], 'm-', linewidth=0.5)
    ax12.set_xlabel('Time (s)')
    ax12.set_ylabel('Amplitude')
    ax12.set_title('Time-Stretch Waveform (first ~1s)')
    ax12.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_file = os.path.join(output_dir, "all_transformations_comparison.png")
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {fig_file}")
    plt.close()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nInput file: {os.path.basename(input_file)}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Mean F0: {mean_f0:.2f} Hz")
    
    print("\nGenerated outputs:")
    print(f"\n1. PITCH-UP ({pitch_up_factor}x):")
    print(f"   File: {output_file_up}")
    print(f"   Mean F0: {mean_f0_up:.2f} Hz (target: {mean_f0*pitch_up_factor:.2f} Hz)")
    print(f"   Duration: {len(pitch_up_signal)/sr:.2f}s (unchanged)")
    
    print(f"\n2. PITCH-DOWN ({pitch_down_factor}x):")
    print(f"   File: {output_file_down}")
    print(f"   Mean F0: {mean_f0_down:.2f} Hz (target: {mean_f0*pitch_down_factor:.2f} Hz)")
    print(f"   Duration: {len(pitch_down_signal)/sr:.2f}s (unchanged)")
    
    print(f"\n3. TIME-STRETCH ({time_stretch_factor}x):")
    print(f"   File: {output_file_stretch}")
    print(f"   Mean F0: {mean_f0_stretch:.2f} Hz (should be ~{mean_f0:.2f} Hz)")
    print(f"   Duration: {stretched_duration:.2f}s (target: {duration*time_stretch_factor:.2f}s)")
    
    print(f"\n4. VISUALIZATION:")
    print(f"   File: {fig_file}")
    print(f"   Contains: Spectrograms, F0 curves, and waveforms for all transformations")
    
    print("\n" + "=" * 70)
    print("ALL TRANSFORMATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


def plot_spectrogram(signal, sr, title, ax):
    """
    Plot spectrogram of a signal.
    
    Args:
        signal: Audio signal
        sr: Sampling rate
        title: Plot title
        ax: Matplotlib axis
    """
    D = librosa.stft(signal)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title(title)
    ax.set_ylim([0, 4000])  # Focus on speech range


def main():
    """
    Main function to generate all transformed samples.
    """
    # Input file
    input_file = "High jumps require focus, balance, and precise timing, allowing the athlete to convert speed and strength into smooth, controlled motion over the bar..wav"
    
    # Generate all samples
    generate_all_samples(input_file, output_dir="output")


if __name__ == "__main__":
    main()
