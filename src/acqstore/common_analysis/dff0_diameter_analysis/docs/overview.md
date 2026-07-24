# Overview

The package compares one reporter analysis and one diameter analysis for exactly one file, channel, and ROI. It supports sidecar inputs for isolated development and `Dff0DiameterAnalysis.from_acq_image(...)` for normal AcqStore use.

The generic triggered-event core knows only about regular time coordinates, one signal, seed indices, and parameters. It does not know about diameter, fluorescence, calcium, ATP, or AcqImage.
