"""Peak-detection parameters for trace acquisitions."""

from __future__ import annotations

from dataclasses import dataclass

from acqstore.common_analysis.peak_detection_core import PeakDetectionCoreParams, PeakPolarity


@dataclass(frozen=True)
class TracePeakDetectionParams:
    """Parameters for ABF/trace peak detection.

    Args:
        polarity: Whether to detect positive or negative peaks.
        height: Optional minimum peak height in transformed detection-signal
            units. For negative peaks, values are internally inverted before
            detection.
        prominence: Optional minimum prominence in transformed detection-signal
            units.
        min_distance_sec: Optional minimum distance between detected peaks in
            seconds.
        width_rel_height: Relative height used for peak-width measurement.
    """

    polarity: PeakPolarity = 'positive'
    height: float | None = None
    prominence: float | None = None
    min_distance_sec: float | None = None
    width_rel_height: float = 0.5

    def to_core_params(self) -> PeakDetectionCoreParams:
        """Return modality-neutral peak-detection parameters.

        Returns:
            Core parameters with the same detection settings.

        Raises:
            ValueError: If a parameter is invalid.
        """
        return PeakDetectionCoreParams(
            polarity=self.polarity,
            height=self.height,
            prominence=self.prominence,
            min_distance_sec=self.min_distance_sec,
            width_rel_height=self.width_rel_height,
        )

    def as_dict(self) -> dict[str, object]:
        """Return JSON-friendly parameter values.

        Returns:
            Dictionary containing all parameter values.
        """
        return {
            'polarity': self.polarity,
            'height': self.height,
            'prominence': self.prominence,
            'min_distance_sec': self.min_distance_sec,
            'width_rel_height': self.width_rel_height,
        }
