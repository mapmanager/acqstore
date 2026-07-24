"""Radon-transform velocity analysis for kymograph ROIs.

This module provides the ``BaseAnalysis`` wrapper around the Radon velocity core
algorithm. It is the public analysis class used by CloudScope, batch workflows,
and scripts. The wrapper handles detection-parameter validation, full-resolution
ROI data access, progress/cancellation, multiprocessing execution options, and
conversion of core outputs into ``AnalysisResult`` and ``AnalysisPlotData``.
"""

from __future__ import annotations

from acqstore.acq_image.analysis.data_provider import AnalysisDataProvider
from acqstore.acq_image.analysis.model import (
    AnalysisCancelled,
    AnalysisPlotData,
    AnalysisResult,
    AnalysisRunContext,
    BaseAnalysis,
    DetectionParamSchema,
    DetectionValueType,
)
from acqstore.acq_image.analysis.registry import register_analysis_class
from acqstore.acq_image.analysis.velocity_analysis.radon_core import (
    RadonVelocityCancelled,
    run_radon_velocity,
)


@register_analysis_class
class RadonVelocityAnalysis(BaseAnalysis):
    """Measure velocity from one channel/ROI using a Radon transform.

    The analysis runs on a full-resolution rectangular ROI crop. For line-scan
    kymographs, rows correspond to time and columns correspond to distance. The
    Radon core analyzes sliding windows along the time axis and reports velocity
    as a function of time.

    Detection parameters are serialized with the analysis and affect scientific
    results. Execution options set with :meth:`set_execution_options` control
    multiprocessing for speed and are not serialized.

    Examples:
        Create and run one analysis through an ``AcqImage`` analysis set::

            key = acq.analysis_set.create(
                "radon_velocity",
                channel=0,
                roi_id=1,
                detection_params={"window_width": 64},
            ).key
            acq.analysis_set.run_analysis(key)
            plot = acq.analysis_set.get(key).get_plot_data()

    Args:
        channel: Zero-based channel index for analysis.
        roi_id: Rectangular ROI identifier for analysis.
        detection_params: Optional detection parameters. Missing values are
            filled from ``detection_schema`` defaults.
    """

    def __init__(
        self,
        *,
        channel: int,
        roi_id: int,
        detection_params: dict[str, object] | None = None,
    ) -> None:
        """Create a Radon velocity analysis instance.

        Args:
            channel: Channel index for analysis.
            roi_id: ROI identifier for analysis.
            detection_params: Optional detection parameters.
        """
        super().__init__(
            channel=channel,
            roi_id=roi_id,
            detection_params=detection_params,
        )
        self._use_multiprocessing = True
        self._processes: int | None = None

    def set_execution_options(
        self,
        *,
        use_multiprocessing: bool = True,
        processes: int | None = None,
    ) -> None:
        """Set runtime execution options for the next run.

        These options are not detection parameters and are not serialized. They
        control how the current Python process executes the expensive Radon
        computation.

        Args:
            use_multiprocessing: Whether to use multiprocessing for Radon
                windows.
            processes: Optional number of worker processes. ``None`` lets the
                core algorithm choose a CPU-count based default.

        Returns:
            None.
        """
        self._use_multiprocessing = bool(use_multiprocessing)
        self._processes = None if processes is None else int(processes)

    analysis_name = "radon_velocity"
    analysis_version = 1
    summary_columns = (
        "analysis_date",
        "analysis_time",
        "analysis_version",
        "num_windows",
        "velocity_mean",
        "velocity_median",
        "velocity_cv",
    )
    exclusive_group = "primary_kymograph"
    detection_schema = (
        DetectionParamSchema(
            name="window_width",
            display_name="Window Width",
            value_type=DetectionValueType.INT,
            default=64,
            description="Number of time samples per Radon analysis window.",
            choices=(16, 64, 128),
            visible=True,
            editable=True,
        ),
    )

    def run(
        self,
        data_provider: AnalysisDataProvider,
        *,
        context: AnalysisRunContext | None = None,
        dependencies: dict[str, BaseAnalysis] | None = None,
    ) -> AnalysisResult:
        """Run Radon velocity analysis on one ROI crop.

        Args:
            data_provider: Provider for full-resolution ROI image data and
                physical spacing. ``get_roi_image`` must return a 2D ``(Y, X)``
                array; for kymographs this is ``(time, space)``.
            context: Optional progress/cancellation context.
            dependencies: Dependency analyses. Radon velocity currently does not
                require dependencies.

        Returns:
            Analysis result populated with a summary dictionary and table. The
            table includes at least ``time_s`` and ``velocity`` when successful.

        Raises:
            AnalysisCancelled: If the run is cancelled through ``context``.
        """
        context = context or AnalysisRunContext()
        context.raise_if_cancelled()
        image = data_provider.get_roi_image(channel=self.key.channel, roi_id=self.key.roi_id)
        physical_units = data_provider.get_image_physical_units()
        window_width = int(self.detection_params["window_width"])

        try:
            result = run_radon_velocity(
                image,
                window_width=window_width,
                physical_units=physical_units,
                progress_callback=context.report_progress,
                cancel_callback=context.is_cancelled,
                use_multiprocessing=self._use_multiprocessing,
                processes=self._processes,
            )
        except RadonVelocityCancelled as exc:
            raise AnalysisCancelled(str(exc)) from exc

        self.result.summary = self.finalize_summary(result.summary)
        self.result.table = result.table
        self.set_dirty()
        return self.result

    def get_plot_data(self) -> AnalysisPlotData | None:
        """Return canonical velocity-versus-time plot data.

        Returns:
            Plot data using ``time_s`` for the x axis and ``velocity`` for the y
            axis, or ``None`` when the analysis has no table output yet.

        Raises:
            KeyError: If the table is present but missing required columns.
        """
        if self.result.table is None:
            return None
        table = self.result.table
        if "time_s" not in table.columns:
            raise KeyError("Radon velocity plot requires 'time_s' column")
        if "velocity" not in table.columns:
            raise KeyError("Radon velocity plot requires 'velocity' column")
        return AnalysisPlotData(
            x=tuple(float(value) for value in table["time_s"].tolist()),
            y=tuple(float(value) for value in table["velocity"].tolist()),
            x_label="Time (s)",
            y_label="Velocity",
            series_name="Radon velocity",
        )
