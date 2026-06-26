"""
Formant Tracking Analysis using LPC
Analyzes F1 and F2 formant frequencies before and after PSOLA modifications
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.signal.windows import hamming
from scipy.fft import fft
import soundfile as sf


def lpc_analysis(signal, order=12, sr=16000):
    """
    Perform LPC analysis to estimate formant frequencies.
    Uses librosa.lpc (allowed) and root-finding method.
    
    Args:
        signal: Audio signal segment
        order: LPC order (typically 2 + sr/1000)
        sr: Sampling rate
        
    Returns:
        formants: List of formant frequencies in Hz
        lpc_spectrum: LPC spectrum for visualization
        freqs: Frequency axis for spectrum
    """
    # Ensure signal is not too short
    if len(signal) < order * 3:
        # Pad signal if too short
        signal = np.pad(signal, (0, order * 3 - len(signal)), mode='constant')
    
    # 1. Pre-emphasis to enhance high frequencies
    y_pre = librosa.effects.preemphasis(signal)
    
    # 2. Apply Hamming window
    y_windowed = y_pre * hamming(len(y_pre))
    
    # 3. Compute LPC coefficients using librosa (allowed by assignment)
    a = librosa.lpc(y_windowed, order=order)
    
    # 4. Find roots of the LPC polynomial
    roots = np.roots(a)
    
    # 5. Keep only roots in upper half of unit circle (imaginary part > 0)
    roots = [r for r in roots if np.imag(r) > 0]
    
    # 6. Convert roots to formant frequencies
    formants = []
    for r in roots:
        # Frequency from angle of root
        freq = np.angle(r) * (sr / (2 * np.pi))
        # Bandwidth from magnitude of root
        bandwidth = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(r))
        
        # Filter for typical formant characteristics
        # Freq > 90Hz (skip DC/hum), bandwidth < 400Hz (reject noise)
        if freq > 90 and bandwidth < 400:
            formants.append(freq)
    
    # Sort by frequency
    formants = sorted(formants)
    formants = np.array(formants)
    
    # Also compute spectrum for visualization
    nfft = 2048
    freqs = np.linspace(0, sr/2, nfft//2)
    
    # LPC spectrum (inverse of prediction error filter)
    w = np.exp(-2j * np.pi * np.arange(nfft) / nfft)
    lpc_fft = np.polyval(a, w)
    
    # Avoid division by zero
    lpc_fft = np.where(np.abs(lpc_fft) < 1e-10, 1e-10, lpc_fft)
    
    lpc_spectrum = 20 * np.log10(np.abs(1.0 / lpc_fft[:nfft//2]))
    
    return formants, lpc_spectrum, freqs


def extract_vowel_segment(signal, sr=16000, segment_duration=0.05):
    """
    Extract a vowel segment from the signal.
    Finds the point of maximum amplitude (usually vowel center) and extracts around it.
    
    Args:
        signal: Audio signal
        sr: Sampling rate
        segment_duration: Duration of segment to extract in seconds
        
    Returns:
        vowel_segment: Extracted vowel segment
        start_idx: Start index of segment
    """
    # Find the point of maximum amplitude (usually the vowel center)
    center_sample = np.argmax(np.abs(signal))
    
    # Take a window around it
    half_window = int(segment_duration * sr / 2)
    start_idx = max(0, center_sample - half_window)
    end_idx = min(len(signal), center_sample + half_window)
    
    vowel_segment = signal[start_idx:end_idx]
    
    return vowel_segment, start_idx


def analyze_formants(input_file, output_dir="output"):
    """
    Analyze formant frequencies for original and modified signals.
    
    Args:
        input_file: Path to original audio file
        output_dir: Directory containing modified audio files
    """
    print("=" * 70)
    print("FORMANT TRACKING ANALYSIS USING LPC")
    print("=" * 70)
    
    sr = 16000
    
    # Load original signal
    print(f"\nLoading original signal: {input_file}")
    signal_orig, _ = librosa.load(input_file, sr=sr)
    
    # Extract vowel segment from original
    print("\nExtracting vowel segment...")
    vowel_orig, start_idx = extract_vowel_segment(signal_orig, sr=sr, segment_duration=0.05)
    print(f"Vowel segment: {len(vowel_orig)} samples ({len(vowel_orig)/sr:.3f}s)")
    print(f"Position: {start_idx/sr:.3f}s to {(start_idx+len(vowel_orig))/sr:.3f}s")
    
    # LPC order (rule of thumb: 2 + sr/1000)
    lpc_order = 2 + int(sr / 1000)
    print(f"LPC order: {lpc_order}")
    
    # Analyze original formants
    print("\n" + "-" * 70)
    print("ORIGINAL SIGNAL FORMANTS")
    print("-" * 70)
    formants_orig, spectrum_orig, freqs = lpc_analysis(vowel_orig, order=lpc_order, sr=sr)
    
    if len(formants_orig) >= 2:
        print(f"F1 = {formants_orig[0]:.1f} Hz")
        print(f"F2 = {formants_orig[1]:.1f} Hz")
        if len(formants_orig) >= 3:
            print(f"F3 = {formants_orig[2]:.1f} Hz")
        if len(formants_orig) >= 4:
            print(f"F4 = {formants_orig[3]:.1f} Hz")
    else:
        print("Warning: Could not detect sufficient formants")
    
    # Analyze modified signals
    modifications = [
        ("pitch_up", "Pitch Up (1.3x)"),
        ("pitch_down", "Pitch Down (0.75x)"),
        ("time_stretched", "Time Stretch (1.5x)")
    ]
    
    results = {"original": {"formants": formants_orig, "spectrum": spectrum_orig}}
    
    for mod_name, mod_label in modifications:
        mod_file = f"{output_dir}/{mod_name}.wav"
        
        print("\n" + "-" * 70)
        print(f"{mod_label.upper()}")
        print("-" * 70)
        
        try:
            # Load modified signal
            signal_mod, _ = librosa.load(mod_file, sr=sr)
            
            # Extract vowel segment from modified signal at same relative position
            # Find maximum amplitude point in the modified signal
            vowel_mod, start_idx_mod = extract_vowel_segment(signal_mod, sr=sr, segment_duration=0.05)
            
            print(f"Analyzing segment: {len(vowel_mod)} samples ({len(vowel_mod)/sr:.3f}s)")
            
            # Analyze formants
            formants_mod, spectrum_mod, _ = lpc_analysis(vowel_mod, order=lpc_order, sr=sr)
            
            print(f"Detected {len(formants_mod)} formants: {formants_mod if len(formants_mod) > 0 else 'none'}")
            
            if len(formants_mod) >= 2:
                print(f"F1 = {formants_mod[0]:.1f} Hz")
                print(f"F2 = {formants_mod[1]:.1f} Hz")
                if len(formants_mod) >= 3:
                    print(f"F3 = {formants_mod[2]:.1f} Hz")
                
                # Calculate displacement
                if len(formants_orig) >= 2:
                    f1_diff = formants_mod[0] - formants_orig[0]
                    f2_diff = formants_mod[1] - formants_orig[1]
                    
                    print(f"\nFormant Displacement:")
                    print(f"ΔF1 = {f1_diff:+.1f} Hz ({f1_diff/formants_orig[0]*100:+.1f}%)")
                    print(f"ΔF2 = {f2_diff:+.1f} Hz ({f2_diff/formants_orig[1]*100:+.1f}%)")
                    
                    results[mod_name] = {
                        "formants": formants_mod,
                        "spectrum": spectrum_mod,
                        "f1_diff": f1_diff,
                        "f2_diff": f2_diff
                    }
            else:
                print("Warning: Could not detect sufficient formants")
                
        except Exception as e:
            print(f"Error analyzing {mod_name}: {e}")
    
    # Create visualization
    print("\n" + "=" * 70)
    print("Generating formant analysis plots...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Original waveform with vowel segment highlighted
    ax1 = plt.subplot(3, 3, 1)
    time_orig = np.arange(len(signal_orig)) / sr
    ax1.plot(time_orig, signal_orig, alpha=0.5, linewidth=0.5)
    segment_time = start_idx / sr
    ax1.axvspan(segment_time, segment_time + len(vowel_orig)/sr, 
                alpha=0.3, color='red', label='Analyzed segment')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Original Signal - Vowel Segment Location')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Vowel segment waveform
    ax2 = plt.subplot(3, 3, 2)
    vowel_time = np.arange(len(vowel_orig)) / sr * 1000  # in ms
    ax2.plot(vowel_time, vowel_orig)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Extracted Vowel Segment')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Original LPC spectrum
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(freqs, spectrum_orig, linewidth=2, label='LPC Spectrum')
    if len(formants_orig) >= 2:
        for i, f in enumerate(formants_orig[:4]):
            ax3.axvline(f, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax3.text(f, ax3.get_ylim()[1]*0.95, f'F{i+1}', ha='center', fontsize=9)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude (dB)')
    ax3.set_title('Original - LPC Spectrum & Formants')
    ax3.set_xlim([0, 4000])
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plots 4-6: Pitch-up
    if "pitch-up" in results:
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(freqs, results["pitch-up"]["spectrum"], linewidth=2, label='LPC Spectrum', color='orange')
        formants_mod = results["pitch-up"]["formants"]
        if len(formants_mod) >= 2:
            for i, f in enumerate(formants_mod[:4]):
                ax4.axvline(f, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                ax4.text(f, ax4.get_ylim()[1]*0.95, f'F{i+1}', ha='center', fontsize=9)
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Magnitude (dB)')
        ax4.set_title('Pitch-Up (1.3x) - LPC Spectrum & Formants')
        ax4.set_xlim([0, 4000])
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # Plots 7-9: Pitch-down
    if "pitch-down" in results:
        ax5 = plt.subplot(3, 3, 7)
        ax5.plot(freqs, results["pitch-down"]["spectrum"], linewidth=2, label='LPC Spectrum', color='green')
        formants_mod = results["pitch-down"]["formants"]
        if len(formants_mod) >= 2:
            for i, f in enumerate(formants_mod[:4]):
                ax5.axvline(f, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                ax5.text(f, ax5.get_ylim()[1]*0.95, f'F{i+1}', ha='center', fontsize=9)
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('Magnitude (dB)')
        ax5.set_title('Pitch-Down (0.75x) - LPC Spectrum & Formants')
        ax5.set_xlim([0, 4000])
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    
    # Comparison plots
    ax6 = plt.subplot(3, 3, 5)
    ax6.plot(freqs, spectrum_orig, linewidth=2, label='Original', alpha=0.7)
    if "pitch-up" in results:
        ax6.plot(freqs, results["pitch-up"]["spectrum"], linewidth=2, 
                label='Pitch-Up', alpha=0.7)
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Magnitude (dB)')
    ax6.set_title('Spectrum Comparison: Original vs Pitch-Up')
    ax6.set_xlim([0, 4000])
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    ax7 = plt.subplot(3, 3, 8)
    ax7.plot(freqs, spectrum_orig, linewidth=2, label='Original', alpha=0.7)
    if "pitch-down" in results:
        ax7.plot(freqs, results["pitch-down"]["spectrum"], linewidth=2, 
                label='Pitch-Down', alpha=0.7)
    ax7.set_xlabel('Frequency (Hz)')
    ax7.set_ylabel('Magnitude (dB)')
    ax7.set_title('Spectrum Comparison: Original vs Pitch-Down')
    ax7.set_xlim([0, 4000])
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Formant displacement bar chart
    ax8 = plt.subplot(3, 3, 6)
    mod_names = []
    f1_displacements = []
    f2_displacements = []
    
    for mod_name, mod_label in modifications:
        if mod_name in results and "f1_diff" in results[mod_name]:
            mod_names.append(mod_label.split(' (')[0])
            f1_displacements.append(results[mod_name]["f1_diff"])
            f2_displacements.append(results[mod_name]["f2_diff"])
    
    if mod_names:
        x = np.arange(len(mod_names))
        width = 0.35
        ax8.bar(x - width/2, f1_displacements, width, label='ΔF1', alpha=0.8)
        ax8.bar(x + width/2, f2_displacements, width, label='ΔF2', alpha=0.8)
        ax8.set_xlabel('Modification Type')
        ax8.set_ylabel('Frequency Displacement (Hz)')
        ax8.set_title('Formant Displacement (F1 & F2)')
        ax8.set_xticks(x)
        ax8.set_xticklabels(mod_names, rotation=15, ha='right')
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')
        ax8.axhline(0, color='black', linewidth=0.8)
    
    # Formant values comparison table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    table_data = [['Modification', 'F1 (Hz)', 'F2 (Hz)', 'ΔF1 (Hz)', 'ΔF2 (Hz)']]
    
    if len(formants_orig) >= 2:
        table_data.append(['Original', f'{formants_orig[0]:.0f}', f'{formants_orig[1]:.0f}', '-', '-'])
    
    for mod_name, mod_label in modifications:
        if mod_name in results and "formants" in results[mod_name]:
            f_mod = results[mod_name]["formants"]
            if len(f_mod) >= 2:
                f1_str = f'{f_mod[0]:.0f}'
                f2_str = f'{f_mod[1]:.0f}'
                
                if "f1_diff" in results[mod_name]:
                    df1_str = f'{results[mod_name]["f1_diff"]:+.0f}'
                    df2_str = f'{results[mod_name]["f2_diff"]:+.0f}'
                else:
                    df1_str = '-'
                    df2_str = '-'
                
                table_data.append([mod_label.split(' (')[0], f1_str, f2_str, df1_str, df2_str])
    
    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax9.set_title('Formant Frequency Summary', pad=20, fontsize=12, weight='bold')
    
    plt.tight_layout()
    
    # Save figure
    fig_file = f"{output_dir}/formant_analysis.png"
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {fig_file}")
    plt.close()
    
    # Print summary report
    print("\n" + "=" * 70)
    print("FORMANT DISPLACEMENT SUMMARY")
    print("=" * 70)
    
    if len(formants_orig) >= 2:
        print(f"\nOriginal Formants:")
        print(f"  F1 = {formants_orig[0]:.1f} Hz")
        print(f"  F2 = {formants_orig[1]:.1f} Hz")
    
    for mod_name, mod_label in modifications:
        if mod_name in results and "f1_diff" in results[mod_name]:
            print(f"\n{mod_label}:")
            f_mod = results[mod_name]["formants"]
            print(f"  F1 = {f_mod[0]:.1f} Hz  (ΔF1 = {results[mod_name]['f1_diff']:+.1f} Hz)")
            print(f"  F2 = {f_mod[1]:.1f} Hz  (ΔF2 = {results[mod_name]['f2_diff']:+.1f} Hz)")
            print(f"  Relative change: ΔF1 = {results[mod_name]['f1_diff']/formants_orig[0]*100:+.1f}%, "
                  f"ΔF2 = {results[mod_name]['f2_diff']/formants_orig[1]*100:+.1f}%")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("\nTD-PSOLA aims to preserve formants during pitch modification.")
    print("Small formant displacements (<5-10%) indicate good preservation.")
    print("Larger displacements suggest formant tracking errors or artifacts.")
    print("\nFor time-stretch operations, formants should be well preserved")
    print("since pitch is maintained.")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    input_file = "he.wav"
    output_dir = "output"
    
    analyze_formants(input_file, output_dir)
