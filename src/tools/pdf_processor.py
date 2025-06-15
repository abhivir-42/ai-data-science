"""
PDF processing tools for AI Data Science agents.

This module provides tools for extracting text, tables, and structured data
from PDF documents to enable data science workflows with PDF inputs.
"""

import os
import io
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from langchain.tools import tool

# PDF processing libraries (will be installed as dependencies)
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


def check_pdf_dependencies() -> Dict[str, bool]:
    """Check which PDF processing libraries are available."""
    return {
        "pdfplumber": PDFPLUMBER_AVAILABLE,
        "tabula": TABULA_AVAILABLE,
        "camelot": CAMELOT_AVAILABLE,
        "pypdf2": PYPDF2_AVAILABLE
    }


@tool
def extract_pdf_text(file_path: str, page_range: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract text content from a PDF file.
    
    This tool extracts all readable text from a PDF document, preserving
    basic formatting and structure where possible.
    
    Parameters
    ----------
    file_path : str
        Path to the PDF file to process
    page_range : str, optional
        Page range to extract (e.g., "1-5", "2,4,6", "all"). Default is "all"
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing extracted text and metadata
    """
    file_path = os.path.expanduser(file_path)
    
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    if not file_path.lower().endswith('.pdf'):
        return {"error": f"File is not a PDF: {file_path}"}
    
    # Check available libraries
    if not (PDFPLUMBER_AVAILABLE or PYPDF2_AVAILABLE):
        return {"error": "No PDF processing libraries available. Install: pip install pdfplumber PyPDF2"}
    
    try:
        extracted_text = ""
        page_info = []
        total_pages = 0
        
        # Try pdfplumber first (better text extraction)
        if PDFPLUMBER_AVAILABLE:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                pages_to_process = _parse_page_range(page_range, total_pages)
                
                for page_num in pages_to_process:
                    if page_num <= total_pages:
                        page = pdf.pages[page_num - 1]  # 0-indexed
                        page_text = page.extract_text()
                        
                        if page_text:
                            extracted_text += f"\n--- Page {page_num} ---\n"
                            extracted_text += page_text
                            
                            page_info.append({
                                "page_number": page_num,
                                "text_length": len(page_text),
                                "has_text": True
                            })
                        else:
                            page_info.append({
                                "page_number": page_num,
                                "text_length": 0,
                                "has_text": False
                            })
        
        # Fallback to PyPDF2 if pdfplumber is not available
        elif PYPDF2_AVAILABLE:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                pages_to_process = _parse_page_range(page_range, total_pages)
                
                for page_num in pages_to_process:
                    if page_num <= total_pages:
                        page = pdf_reader.pages[page_num - 1]  # 0-indexed
                        page_text = page.extract_text()
                        
                        if page_text:
                            extracted_text += f"\n--- Page {page_num} ---\n"
                            extracted_text += page_text
                            
                            page_info.append({
                                "page_number": page_num,
                                "text_length": len(page_text),
                                "has_text": True
                            })
                        else:
                            page_info.append({
                                "page_number": page_num,
                                "text_length": 0,
                                "has_text": False
                            })
        
        # Clean up the extracted text
        extracted_text = _clean_extracted_text(extracted_text)
        
        return {
            "text": extracted_text,
            "file_path": file_path,
            "total_pages": total_pages,
            "pages_processed": len(pages_to_process),
            "page_info": page_info,
            "text_length": len(extracted_text),
            "extraction_method": "pdfplumber" if PDFPLUMBER_AVAILABLE else "pypdf2",
            "has_text": len(extracted_text.strip()) > 0
        }
        
    except Exception as e:
        return {"error": f"Error extracting text from PDF: {str(e)}"}


@tool
def extract_pdf_tables(file_path: str, pages: Optional[str] = None, method: str = "auto") -> Dict[str, Any]:
    """
    Extract tables from a PDF file and convert them to DataFrames.
    
    This tool attempts to detect and extract tabular data from PDF documents
    using multiple extraction methods for best results.
    
    Parameters
    ----------
    file_path : str
        Path to the PDF file to process
    pages : str, optional
        Pages to extract tables from (e.g., "1-5", "2,4,6", "all"). Default is "all"
    method : str, optional
        Extraction method to use ("auto", "tabula", "pdfplumber"). Default is "auto"
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing extracted tables and metadata
    """
    file_path = os.path.expanduser(file_path)
    
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    if not file_path.lower().endswith('.pdf'):
        return {"error": f"File is not a PDF: {file_path}"}
    
    # Check available libraries based on method
    available_methods = []
    if TABULA_AVAILABLE:
        available_methods.append("tabula")
    if CAMELOT_AVAILABLE:
        available_methods.append("camelot")
    if PDFPLUMBER_AVAILABLE:
        available_methods.append("pdfplumber")
    
    if not available_methods:
        return {"error": "No table extraction libraries available. Install: pip install tabula-py camelot-py pdfplumber"}
    
    if method == "auto":
        # Use the best available method
        if "tabula" in available_methods:
            method = "tabula"
        elif "camelot" in available_methods:
            method = "camelot"
        else:
            method = "pdfplumber"
    
    if method not in available_methods:
        return {"error": f"Method '{method}' not available. Available methods: {available_methods}"}
    
    try:
        tables = []
        extraction_info = {
            "method_used": method,
            "pages_processed": 0,
            "tables_found": 0,
            "extraction_quality": "unknown"
        }
        
        if method == "tabula":
            # Use tabula-py for table extraction
            pages_param = pages if pages and pages != "all" else None
            
            tabula_tables = tabula.read_pdf(
                file_path,
                pages=pages_param,
                multiple_tables=True,
                pandas_options={'on_bad_lines': 'skip'}
            )
            
            for i, table_df in enumerate(tabula_tables):
                if not table_df.empty:
                    tables.append({
                        "table_id": i + 1,
                        "data": table_df.to_dict(),
                        "shape": table_df.shape,
                        "columns": list(table_df.columns),
                        "extraction_method": "tabula"
                    })
            
            extraction_info["tables_found"] = len(tables)
            extraction_info["extraction_quality"] = "high" if tables else "none"
        
        elif method == "camelot":
            # Use camelot for table extraction
            pages_param = pages if pages and pages != "all" else "all"
            
            camelot_tables = camelot.read_pdf(file_path, pages=pages_param)
            
            for i, table in enumerate(camelot_tables):
                table_df = table.df
                if not table_df.empty:
                    tables.append({
                        "table_id": i + 1,
                        "data": table_df.to_dict(),
                        "shape": table_df.shape,
                        "columns": list(table_df.columns),
                        "extraction_method": "camelot",
                        "accuracy": getattr(table, 'accuracy', None)
                    })
            
            extraction_info["tables_found"] = len(tables)
            extraction_info["extraction_quality"] = "high" if tables else "none"
        
        elif method == "pdfplumber":
            # Use pdfplumber for table extraction
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                pages_to_process = _parse_page_range(pages, total_pages) if pages else range(1, total_pages + 1)
                
                table_id = 1
                for page_num in pages_to_process:
                    if page_num <= total_pages:
                        page = pdf.pages[page_num - 1]  # 0-indexed
                        page_tables = page.extract_tables()
                        
                        for table in page_tables:
                            if table and len(table) > 1:  # At least header + 1 row
                                # Convert table to DataFrame
                                df = pd.DataFrame(table[1:], columns=table[0] if table[0] else [f"col_{i}" for i in range(len(table[1]))])
                                
                                if not df.empty:
                                    tables.append({
                                        "table_id": table_id,
                                        "page_number": page_num,
                                        "data": df.to_dict(),
                                        "shape": df.shape,
                                        "columns": list(df.columns),
                                        "extraction_method": "pdfplumber"
                                    })
                                    table_id += 1
                
                extraction_info["tables_found"] = len(tables)
                extraction_info["pages_processed"] = len(pages_to_process)
                extraction_info["extraction_quality"] = "medium" if tables else "none"
        
        return {
            "tables": tables,
            "file_path": file_path,
            "extraction_info": extraction_info,
            "total_tables": len(tables),
            "success": len(tables) > 0
        }
        
    except Exception as e:
        return {"error": f"Error extracting tables from PDF: {str(e)}"}


@tool
def smart_extract_data_from_pdf(file_path: str, extraction_strategy: str = "comprehensive") -> Dict[str, Any]:
    """
    Intelligently extract structured data from PDF using multiple methods.
    
    This tool tries multiple extraction approaches and returns the best results
    based on data quality and completeness.
    
    Parameters
    ----------
    file_path : str
        Path to the PDF file to process
    extraction_strategy : str, optional
        Strategy to use ("comprehensive", "tables_only", "text_only"). Default is "comprehensive"
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing extracted data with quality scores
    """
    file_path = os.path.expanduser(file_path)
    
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    if not file_path.lower().endswith('.pdf'):
        return {"error": f"File is not a PDF: {file_path}"}
    
    results = {
        "file_path": file_path,
        "extraction_strategy": extraction_strategy,
        "structured_data": [],
        "text_content": "",
        "extraction_summary": {
            "tables_found": 0,
            "text_extracted": False,
            "quality_score": 0.0,
            "recommended_use": "none"
        }
    }
    
    try:
        # Step 1: Try table extraction if strategy allows
        if extraction_strategy in ["comprehensive", "tables_only"]:
            table_result = extract_pdf_tables(file_path)
            
            if "tables" in table_result and table_result["tables"]:
                results["structured_data"] = table_result["tables"]
                results["extraction_summary"]["tables_found"] = len(table_result["tables"])
                
                # Calculate quality score based on table completeness
                total_cells = sum(table["shape"][0] * table["shape"][1] for table in table_result["tables"])
                if total_cells > 100:  # Substantial data
                    results["extraction_summary"]["quality_score"] = 0.9
                    results["extraction_summary"]["recommended_use"] = "data_analysis"
                elif total_cells > 20:  # Some data
                    results["extraction_summary"]["quality_score"] = 0.7
                    results["extraction_summary"]["recommended_use"] = "data_review"
                else:  # Minimal data
                    results["extraction_summary"]["quality_score"] = 0.4
                    results["extraction_summary"]["recommended_use"] = "manual_review"
        
        # Step 2: Extract text content if strategy allows or tables failed
        if (extraction_strategy in ["comprehensive", "text_only"] or 
            results["extraction_summary"]["tables_found"] == 0):
            
            text_result = extract_pdf_text(file_path)
            
            if "text" in text_result and text_result["text"]:
                results["text_content"] = text_result["text"]
                results["extraction_summary"]["text_extracted"] = True
                
                # If no tables were found, try to find structured patterns in text
                if results["extraction_summary"]["tables_found"] == 0:
                    structured_text_data = _extract_structured_patterns_from_text(text_result["text"])
                    
                    if structured_text_data:
                        results["structured_data"].extend(structured_text_data)
                        results["extraction_summary"]["quality_score"] = 0.6
                        results["extraction_summary"]["recommended_use"] = "text_analysis"
                    else:
                        results["extraction_summary"]["quality_score"] = 0.3
                        results["extraction_summary"]["recommended_use"] = "content_review"
        
        # Step 3: Determine final recommendations
        if results["extraction_summary"]["tables_found"] > 0:
            results["extraction_summary"]["primary_data_type"] = "tabular"
        elif results["extraction_summary"]["text_extracted"]:
            results["extraction_summary"]["primary_data_type"] = "textual"
        else:
            results["extraction_summary"]["primary_data_type"] = "none"
            results["extraction_summary"]["recommended_use"] = "manual_processing"
        
        return results
        
    except Exception as e:
        return {"error": f"Error in smart PDF extraction: {str(e)}"}


@tool
def get_pdf_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a PDF file.
    
    This tool extracts metadata, structure information, and basic content
    analysis to help determine the best extraction approach.
    
    Parameters
    ----------
    file_path : str
        Path to the PDF file to analyze
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing PDF metadata and structure information
    """
    file_path = os.path.expanduser(file_path)
    
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    if not file_path.lower().endswith('.pdf'):
        return {"error": f"File is not a PDF: {file_path}"}
    
    if not (PDFPLUMBER_AVAILABLE or PYPDF2_AVAILABLE):
        return {"error": "No PDF processing libraries available. Install: pip install pdfplumber PyPDF2"}
    
    try:
        info = {
            "file_path": file_path,
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
            "file_name": os.path.basename(file_path)
        }
        
        # Try pdfplumber first for better analysis
        if PDFPLUMBER_AVAILABLE:
            with pdfplumber.open(file_path) as pdf:
                info.update({
                    "total_pages": len(pdf.pages),
                    "metadata": pdf.metadata or {},
                    "analysis_method": "pdfplumber"
                })
                
                # Analyze first few pages for structure
                pages_to_analyze = min(3, len(pdf.pages))
                text_pages = 0
                table_pages = 0
                total_text_length = 0
                
                for i in range(pages_to_analyze):
                    page = pdf.pages[i]
                    
                    # Check for text
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 50:
                        text_pages += 1
                        total_text_length += len(page_text)
                    
                    # Check for tables
                    tables = page.extract_tables()
                    if tables:
                        table_pages += 1
                
                info["content_analysis"] = {
                    "pages_with_text": text_pages,
                    "pages_with_tables": table_pages,
                    "avg_text_length_per_page": total_text_length // max(text_pages, 1),
                    "likely_content_type": _determine_content_type(text_pages, table_pages, total_text_length),
                    "extraction_recommendation": _get_extraction_recommendation(text_pages, table_pages)
                }
        
        # Fallback to PyPDF2
        elif PYPDF2_AVAILABLE:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                info.update({
                    "total_pages": len(pdf_reader.pages),
                    "metadata": pdf_reader.metadata or {},
                    "analysis_method": "pypdf2"
                })
                
                # Basic text analysis
                text_pages = 0
                total_text_length = 0
                
                pages_to_analyze = min(3, len(pdf_reader.pages))
                for i in range(pages_to_analyze):
                    page = pdf_reader.pages[i]
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 50:
                        text_pages += 1
                        total_text_length += len(page_text)
                
                info["content_analysis"] = {
                    "pages_with_text": text_pages,
                    "pages_with_tables": "unknown",
                    "avg_text_length_per_page": total_text_length // max(text_pages, 1),
                    "likely_content_type": "text_document" if text_pages > 0 else "unknown",
                    "extraction_recommendation": "text_extraction"
                }
        
        return info
        
    except Exception as e:
        return {"error": f"Error analyzing PDF: {str(e)}"}


# Helper functions

def _parse_page_range(page_range: Optional[str], total_pages: int) -> List[int]:
    """Parse page range string into list of page numbers."""
    if not page_range or page_range == "all":
        return list(range(1, total_pages + 1))
    
    pages = []
    parts = page_range.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            pages.extend(range(start, min(end + 1, total_pages + 1)))
        else:
            page_num = int(part)
            if 1 <= page_num <= total_pages:
                pages.append(page_num)
    
    return sorted(list(set(pages)))


def _clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove page break artifacts
    text = re.sub(r'\n--- Page \d+ ---\n', '\n\n=== PAGE BREAK ===\n\n', text)
    
    return text.strip()


def _extract_structured_patterns_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract structured data patterns from text content."""
    structured_data = []
    
    # Look for key-value pairs
    kv_pattern = r'^([A-Za-z][A-Za-z\s]+):\s*(.+)$'
    kv_matches = re.findall(kv_pattern, text, re.MULTILINE)
    
    if len(kv_matches) > 3:  # At least 3 key-value pairs
        kv_data = {match[0].strip(): match[1].strip() for match in kv_matches}
        structured_data.append({
            "data_type": "key_value_pairs",
            "data": kv_data,
            "extraction_method": "regex_pattern"
        })
    
    # Look for lists or bullet points
    list_pattern = r'^[\s]*[•\-\*]\s*(.+)$'
    list_matches = re.findall(list_pattern, text, re.MULTILINE)
    
    if len(list_matches) > 3:  # At least 3 list items
        structured_data.append({
            "data_type": "list_items",
            "data": [item.strip() for item in list_matches],
            "extraction_method": "regex_pattern"
        })
    
    return structured_data


def _determine_content_type(text_pages: int, table_pages: int, total_text_length: int) -> str:
    """Determine the likely content type of the PDF."""
    if table_pages > 0 and table_pages >= text_pages:
        return "data_document"
    elif text_pages > 0 and total_text_length > 1000:
        return "text_document"
    elif text_pages > 0:
        return "mixed_document"
    else:
        return "unknown"


def _get_extraction_recommendation(text_pages: int, table_pages: int) -> str:
    """Get recommendation for best extraction method."""
    if table_pages > 0:
        return "table_extraction"
    elif text_pages > 0:
        return "text_extraction"
    else:
        return "manual_review" 