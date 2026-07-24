"""Tree-row contract helpers for AcqStore acquisition file trees."""

from __future__ import annotations

ACQ_TREE_ROW_ID_FIELD = 'tree_row_id'
ACQ_TREE_PATH_FIELD = 'hierarchy_path'
ACQ_TREE_ROW_TYPE_FIELD = 'tree_row_type'

ACQ_TREE_ROW_TYPE_FILE = 'file'
ACQ_TREE_ROW_TYPE_ANALYSIS = 'analysis'

ACQ_TREE_ANALYSIS_NAME_FIELD = 'analysis_name'
ACQ_TREE_ANALYSIS_CHANNEL_FIELD = 'channel'
ACQ_TREE_ANALYSIS_ROI_ID_FIELD = 'roi_id'


def build_analysis_tree_row_id(
    file_id: str,
    analysis_name: str,
    channel: int,
    roi_id: int,
) -> str:
    """Return a stable tree row identifier for one analysis.

    Args:
        file_id: Stable acquisition file identifier.
        analysis_name: Stable analysis type name.
        channel: Analysis channel index.
        roi_id: Analysis ROI identifier.

    Returns:
        Globally unique row identifier within an acquisition tree row set.
    """
    return f'{file_id}::analysis::{analysis_name}::ch_{channel}::roi_{roi_id}'
