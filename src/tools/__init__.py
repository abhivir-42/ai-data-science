"""
Tools for AI Data Science agents.

This package provides tools for data processing, data loading, and other
operations used by the AI Data Science agents.
"""

from src.tools.dataframe import get_dataframe_summary
from src.tools.data_loader import (
    load_file,
    load_directory,
    list_directory_contents,
    list_directory_recursive,
    get_file_info,
    search_files_by_pattern
)

from src.tools.eda import (
    explain_data,
    describe_dataset,
    visualise_missing,
    generate_correlation_funnel,
    generate_sweetviz_report,
    generate_dtale_report,
)

from src.tools.pdf_processor import (
    extract_pdf_text,
    extract_pdf_tables,
    smart_extract_data_from_pdf,
    get_pdf_info
)

__all__ = [
    "get_dataframe_summary",
    "load_file",
    "load_directory",
    "list_directory_contents", 
    "list_directory_recursive",
    "get_file_info",
    "search_files_by_pattern",
    "explain_data",
    "describe_dataset",
    "visualise_missing",
    "generate_correlation_funnel",
    "generate_sweetviz_report",
    "generate_dtale_report",
    "extract_pdf_text",
    "extract_pdf_tables",
    "smart_extract_data_from_pdf",
    "get_pdf_info",
] 