"""
Time-Domain Pitch-Synchronous Overlap-Add (TD-PSOLA) Implementation

This module implements the TD-PSOLA algorithm for speech manipulation including:
- Pitch modification (±50%)
- Duration modification (0.5× to 2×)
- Preservation of timbre and formants
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
from typing import Tuple, Optional


class TDPSOLA:
    """
    Time-Domain Pitch-Synchronous Overlap-Add (TD-PSOLA) implementation.
    
    This class implements the core TD-PSOLA algorithm for speech manipulation
    without using ready-made PSOLA implementations.
    """
    
    def __init__(self, sr: int = 16000):
        """
        Initialize TD-PSOLA processor.
        
        Args:
            sr: Sampling rate (default: 16000 Hz)
        """
        self.sr = sr
        
    def estimate_f0(self, signal: np.ndarray, 
                   fmin: float = 75.0, 
                   fmax: float = 400.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate fundamental frequency (F0) using autocorrelation method.
        This can use librosa or other libraries as allowed.
        
        Args:
            signal: Input audio signal
            fmin: Minimum F0 in Hz
            fmax: Maximum F0 in Hz
            
        Returns:
            times: Time points for F0 estimates
            f0: F0 values in Hz
        """
        # Using librosa's pyin for F0 estimation (allowed by assignment)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            signal,
            fmin=fmin,
            fmax=fmax,
            sr=self.sr,
            frame_length=2048
        )
        
        # Create time array
        hop_length = 512
        times = librosa.frames_to_time(
            np.arange(len(f0)),
            sr=self.sr,
            hop_length=hop_length
        )
        
        # Interpolate NaN values
        mask = np.isnan(f0)
        if not mask.all():
            f0[mask] = np.interp(
                np.flatnonzero(mask),
                np.flatnonzero(~mask),
                f0[~mask]
            )
        else:
            # If all NaN, use median frequency
            f0[:] = (fmin + fmax) / 2
            
        return times, f0
    
    def detect_pitch_marks(self, signal: np.ndarray, 
                          f0_time: np.ndarray, 
                          f0: np.ndarray) -> np.ndarray:
        """
        Detect pitch-synchronous points (pitch marks) in the signal.
        
        This is a core component that must be implemented manually.
        Uses local peak detection combined with F0 estimates.
        
        Args:
            signal: Input audio signal
            f0_time: Time points for F0 estimates
            f0: F0 values in Hz
            
        Returns:
            pitch_marks: Array of sample indices for pitch marks
        """
        # Interpolate F0 to signal sample rate
        signal_times = np.arange(len(signal)) / self.sr
        f0_interp = np.interp(signal_times, f0_time, f0)
        
        # Calculate expected period at each sample
        period_samples = self.sr / f0_interp
        
        # Detect all peaks in the signal
        peaks, _ = find_peaks(signal, distance=int(self.sr * 0.002))  # Min 2ms apart
        
        if len(peaks) == 0:
            # Fallback: create uniform marks based on median F0
            median_period = int(self.sr / np.median(f0))
            return np.arange(0, len(signal), median_period)
        
        # Select peaks that align with expected pitch periods
        pitch_marks = []
        current_sample = peaks[0]
        pitch_marks.append(current_sample)
        
        i = 0
        while current_sample < len(signal):
            # Get expected period at current position
            expected_period = period_samples[min(current_sample, len(period_samples) - 1)]
            
            # Look for next peak in expected range
            search_start = current_sample + int(expected_period * 0.7)
            search_end = current_sample + int(expected_period * 1.3)
            
            # Find peaks in search window
            candidates = peaks[(peaks > search_start) & (peaks < search_end)]
            
            if len(candidates) > 0:
                # Choose peak with maximum amplitude
                best_peak = candidates[np.argmax(np.abs(signal[candidates]))]
                pitch_marks.append(best_peak)
                current_sample = best_peak
            else:
                # No peak found, step by expected period
                current_sample = current_sample + int(expected_period)
                if current_sample < len(signal):
                    pitch_marks.append(current_sample)
        
        return np.array(pitch_marks)
    
    def apply_hanning_window(self, length: int) -> np.ndarray:
        """
        Create a Hanning window for segmentation.
        
        Args:
            length: Window length in samples
            
        Returns:
            window: Hanning window array
        """
        return np.hanning(length)
    
    def extract_grains(self, signal: np.ndarray, 
                       pitch_marks: np.ndarray,
                       max_period: Optional[int] = None) -> list:
        """
        Extract and window signal segments (grains) around pitch marks.
        
        This is a core component that must be implemented manually.
        
        Args:
            signal: Input audio signal
            pitch_marks: Array of pitch mark sample indices
            max_period: Maximum period length (if None, computed from marks)
            
        Returns:
            grains: List of windowed signal grains with their center positions
        """
        grains = []
        
        # Calculate periods between consecutive marks
        if len(pitch_marks) < 2:
            return grains
        
        periods = np.diff(pitch_marks)
        
        if max_period is None:
            max_period = int(np.max(periods) * 1.5)
        
        for i, mark in enumerate(pitch_marks):
            # Determine window size based on local period
            if i < len(periods):
                period = periods[i]
            elif i > 0:
                period = periods[i - 1]
            else:
                period = max_period
            
            # Window extends to ±1 period around the mark
            window_size = int(period * 2)
            half_window = window_size // 2
            
            # Extract grain boundaries
            start = max(0, mark - half_window)
            end = min(len(signal), mark + half_window)
            
            # Extract grain
            grain = signal[start:end].copy()
            
            # Apply Hanning window
            if len(grain) > 0:
                window = self.apply_hanning_window(len(grain))
                grain = grain * window
                
                # Store grain with its center position
                grains.append({
                    'samples': grain,
                    'center': mark,
                    'start': start,
                    'end': end
                })
        
        return grains
    
    def synthesize_psola(self, grains: list, 
                        pitch_marks: np.ndarray,
                        pitch_shift: float = 1.0,
                        time_stretch: float = 1.0,
                        output_length: Optional[int] = None) -> np.ndarray:
        """
        Synthesize output signal using overlap-add of grains.
        
        This is a core component that must be implemented manually.
        
        Args:
            grains: List of windowed signal grains
            pitch_marks: Original pitch mark positions
            pitch_shift: Pitch modification factor (1.0 = no change, 2.0 = one octave up)
            time_stretch: Duration modification factor (1.0 = no change, 2.0 = twice as long)
            output_length: Desired output length (if None, computed automatically)
            
        Returns:
            output: Synthesized signal
        """
        if len(grains) == 0:
            return np.array([])
        
        # Calculate new pitch mark positions
        # For pitch shift: marks are closer/farther apart
        # For time stretch: marks are proportionally spaced
        
        periods = np.diff(pitch_marks)
        if len(periods) == 0:
            periods = np.array([512])  # Default period
        
        # New periods: scale by time_stretch and inverse of pitch_shift
        # (higher pitch = shorter period, longer duration = longer spacing)
        new_periods = periods * time_stretch / pitch_shift
        
        # Generate new pitch mark positions
        new_marks = [0]
        for period in new_periods:
            new_marks.append(new_marks[-1] + int(period))
        new_marks = np.array(new_marks)
        
        # Determine output length
        if output_length is None:
            last_grain_end = new_marks[-1] + len(grains[-1]['samples']) // 2
            output_length = int(last_grain_end * 1.1)  # Add 10% buffer
        
        # Initialize output buffer
        output = np.zeros(output_length, dtype=np.float64)
        
        # Overlap-add grains at new positions
        for i, grain in enumerate(grains):
            if i >= len(new_marks):
                break
            
            new_center = new_marks[i]
            grain_samples = grain['samples']
            half_len = len(grain_samples) // 2
            
            # Calculate placement window
            start_pos = new_center - half_len
            end_pos = start_pos + len(grain_samples)
            
            # Handle boundary conditions
            grain_start = 0
            grain_end = len(grain_samples)
            
            if start_pos < 0:
                grain_start = -start_pos
                start_pos = 0
            
            if end_pos > output_length:
                grain_end = grain_end - (end_pos - output_length)
                end_pos = output_length
            
            if start_pos < output_length and grain_start < len(grain_samples):
                # Overlap-add
                output[start_pos:end_pos] += grain_samples[grain_start:grain_end]
        
        return output
    
    def modify_pitch(self, signal: np.ndarray, 
                    pitch_shift: float = 1.0) -> np.ndarray:
        """
        Modify pitch of the signal while preserving duration.
        
        Args:
            signal: Input audio signal
            pitch_shift: Pitch shift factor (1.0 = no change, 2.0 = one octave up,
                        0.5 = one octave down). Range approximately 0.5 to 1.5.
                        
        Returns:
            output: Pitch-modified signal
        """
        # Estimate F0
        f0_time, f0 = self.estimate_f0(signal)
        
        # Detect pitch marks
        pitch_marks = self.detect_pitch_marks(signal, f0_time, f0)
        
        # Extract grains
        grains = self.extract_grains(signal, pitch_marks)
        
        # Synthesize with pitch shift and compensating time stretch
        # to maintain original duration
        output = self.synthesize_psola(
            grains,
            pitch_marks,
            pitch_shift=pitch_shift,
            time_stretch=pitch_shift,  # Compensate for pitch shift
            output_length=len(signal)
        )
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * np.max(np.abs(signal))
        
        return output
    
    def modify_duration(self, signal: np.ndarray,
                       time_stretch: float = 1.0) -> np.ndarray:
        """
        Modify duration of the signal while preserving pitch.
        
        Args:
            signal: Input audio signal
            time_stretch: Time stretch factor (1.0 = no change, 2.0 = twice as long,
                         0.5 = half as long). Range approximately 0.5 to 2.0.
                         
        Returns:
            output: Duration-modified signal
        """
        # Estimate F0
        f0_time, f0 = self.estimate_f0(signal)
        
        # Detect pitch marks
        pitch_marks = self.detect_pitch_marks(signal, f0_time, f0)
        
        # Extract grains
        grains = self.extract_grains(signal, pitch_marks)
        
        # Synthesize with time stretch only
        output = self.synthesize_psola(
            grains,
            pitch_marks,
            pitch_shift=1.0,  # No pitch change
            time_stretch=time_stretch,
            output_length=None
        )
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * np.max(np.abs(signal))
        
        return output
    
    def modify_pitch_and_duration(self, signal: np.ndarray,
                                 pitch_shift: float = 1.0,
                                 time_stretch: float = 1.0) -> np.ndarray:
        """
        Modify both pitch and duration independently.
        
        Args:
            signal: Input audio signal
            pitch_shift: Pitch shift factor
            time_stretch: Time stretch factor
            
        Returns:
            output: Modified signal
        """
        # Estimate F0
        f0_time, f0 = self.estimate_f0(signal)
        
        # Detect pitch marks
        pitch_marks = self.detect_pitch_marks(signal, f0_time, f0)
        
        # Extract grains
        grains = self.extract_grains(signal, pitch_marks)
        
        # Synthesize with both modifications
        output = self.synthesize_psola(
            grains,
            pitch_marks,
            pitch_shift=pitch_shift,
            time_stretch=time_stretch,
            output_length=None
        )
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * np.max(np.abs(signal))
        
        return output
