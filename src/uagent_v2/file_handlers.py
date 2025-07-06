"""
Secure file handling module for the enhanced uAgent implementation.

This module provides secure file upload, download, and processing capabilities
to replace the insecure file handling in the original code.
"""

import os
import time
import requests
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
from .config import UAgentConfig
from .exceptions import SecurityError, FileProcessingError, NetworkError
from .utils import MemoryEfficientCSVProcessor, FileValidator, format_file_size, sanitize_filename


logger = logging.getLogger(__name__)


class SecureFileUploader:
    """Secure file upload handler with validation and rate limiting."""
    
    def __init__(self, config: UAgentConfig):
        self.config = config
        self.validator = FileValidator(config)
        self.processor = MemoryEfficientCSVProcessor(config)
        self._upload_attempts = {}
        self._last_upload_time = {}
    
    def upload_csv_secure(self, file_path: str, file_description: str = "Processed Data") -> Dict[str, Any]:
        """
        Securely upload CSV file with validation and rate limiting.
        
        Args:
            file_path: Local path to CSV file
            file_description: Description for the file
            
        Returns:
            Dictionary with upload result
        """
        try:
            # Validate file security
            if not self.validator.validate_file_security(file_path):
                raise SecurityError("File failed security validation")
            
            # Check upload rate limiting
            if not self._check_rate_limit(file_path):
                raise SecurityError("Upload rate limit exceeded")
            
            # Get file information
            file_size = self.processor.get_file_size_safe(file_path)
            
            # Sanitize filename
            original_name = os.path.basename(file_path)
            safe_name = sanitize_filename(original_name)
            
            # Attempt upload to multiple services
            upload_result = self._attempt_upload_to_services(file_path, safe_name, file_size)
            
            # Update rate limiting
            self._update_rate_limit(file_path)
            
            return upload_result
            
        except Exception as e:
            logger.error(f"Secure upload failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": None,
                "security_check": "failed"
            }
    
    def _check_rate_limit(self, file_path: str) -> bool:
        """Check if upload is within rate limits."""
        current_time = time.time()
        
        # Check attempts
        if file_path in self._upload_attempts:
            if self._upload_attempts[file_path] >= self.config.max_upload_attempts:
                return False
        
        # Check time-based rate limiting
        if file_path in self._last_upload_time:
            time_since_last = current_time - self._last_upload_time[file_path]
            if time_since_last < 60:  # 1 minute minimum between uploads
                return False
        
        return True
    
    def _update_rate_limit(self, file_path: str):
        """Update rate limiting counters."""
        current_time = time.time()
        
        if file_path not in self._upload_attempts:
            self._upload_attempts[file_path] = 0
        
        self._upload_attempts[file_path] += 1
        self._last_upload_time[file_path] = current_time
    
    def _attempt_upload_to_services(self, file_path: str, safe_name: str, file_size: int) -> Dict[str, Any]:
        """Attempt upload to multiple hosting services."""
        
        # List of hosting services (prioritized by reliability)
        services = [
            {
                "name": "tmpfiles.org",
                "url": "https://tmpfiles.org/api/v1/upload",
                "method": "POST",
                "timeout": self.config.upload_timeout_seconds
            }
        ]
        
        last_error = None
        
        for service in services:
            try:
                logger.info(f"Attempting upload to {service['name']}")
                
                result = self._upload_to_service(file_path, service, safe_name, file_size)
                
                if result.get("success"):
                    return result
                else:
                    last_error = result.get("error", "Unknown error")
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Upload to {service['name']} failed: {e}")
                continue
        
        # All services failed
        return {
            "success": False,
            "error": f"All upload services failed. Last error: {last_error}",
            "url": None,
            "fallback_available": True
        }
    
    def _upload_to_service(self, file_path: str, service: Dict[str, Any], safe_name: str, file_size: int) -> Dict[str, Any]:
        """Upload to a specific service."""
        try:
            with open(file_path, 'rb') as file:
                files = {'file': (safe_name, file, 'text/csv')}
                
                response = requests.post(
                    service['url'],
                    files=files,
                    timeout=service['timeout']
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Parse result based on service
                    if service['name'] == "tmpfiles.org":
                        if result.get('status') == 'success':
                            file_url = result['data']['url']
                            
                            # Ensure HTTPS
                            if file_url.startswith('http://'):
                                file_url = file_url.replace('http://', 'https://')
                            
                            return {
                                "success": True,
                                "url": file_url,
                                "service": service['name'],
                                "file_id": file_url.split('/')[-2] if '/' in file_url else 'unknown',
                                "size_mb": file_size / (1024 * 1024),
                                "expires": "60 minutes",
                                "security_validated": True
                            }
                
                # Service returned error
                return {
                    "success": False,
                    "error": f"Service returned status {response.status_code}",
                    "url": None
                }
                
        except requests.exceptions.Timeout:
            raise NetworkError(f"Upload timeout to {service['name']}")
        except requests.exceptions.ConnectionError:
            raise NetworkError(f"Connection error to {service['name']}")
        except Exception as e:
            raise FileProcessingError(f"Upload failed: {e}")


class SecureFileDownloader:
    """Secure file download handler with validation."""
    
    def __init__(self, config: UAgentConfig):
        self.config = config
        self.validator = FileValidator(config)
    
    def download_csv_secure(self, url: str, local_path: str) -> Dict[str, Any]:
        """
        Securely download CSV file with validation.
        
        Args:
            url: URL to download from
            local_path: Local path to save file
            
        Returns:
            Dictionary with download result
        """
        try:
            # Validate URL
            if not self._validate_url(url):
                raise SecurityError("Invalid or unsafe URL")
            
            # Create secure local path
            safe_path = self._create_safe_path(local_path)
            
            # Download with validation
            result = self._download_with_validation(url, safe_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Secure download failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "local_path": None
            }
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL for security."""
        try:
            # Check protocol
            if not url.startswith(('http://', 'https://')):
                return False
            
            # Check for suspicious patterns
            suspicious_patterns = ['..', '<', '>', '"', "'", '`', '|', '&', ';']
            if any(pattern in url for pattern in suspicious_patterns):
                return False
            
            # Check file extension
            if not url.lower().endswith('.csv'):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _create_safe_path(self, local_path: str) -> str:
        """Create a safe local path."""
        # Sanitize the path
        safe_path = sanitize_filename(os.path.basename(local_path))
        
        # Ensure it's in the temp directory
        temp_dir = Path(self.config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        return str(temp_dir / safe_path)
    
    def _download_with_validation(self, url: str, local_path: str) -> Dict[str, Any]:
        """Download file with validation."""
        try:
            response = requests.get(
                url,
                timeout=self.config.request_timeout_seconds,
                stream=True
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'csv' not in content_type.lower() and 'text' not in content_type.lower():
                logger.warning(f"Unexpected content type: {content_type}")
            
            # Download with size limit
            total_size = 0
            max_size = self.config.get_file_size_bytes()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        total_size += len(chunk)
                        if total_size > max_size:
                            raise SecurityError("File too large during download")
                        f.write(chunk)
            
            # Validate downloaded file
            if self.validator.validate_file_security(local_path):
                return {
                    "success": True,
                    "local_path": local_path,
                    "size_bytes": total_size,
                    "size_formatted": format_file_size(total_size),
                    "validated": True
                }
            else:
                # Remove invalid file
                os.remove(local_path)
                raise SecurityError("Downloaded file failed validation")
                
        except requests.exceptions.Timeout:
            raise NetworkError("Download timeout")
        except requests.exceptions.ConnectionError:
            raise NetworkError("Connection error during download")
        except Exception as e:
            # Clean up on error
            if os.path.exists(local_path):
                os.remove(local_path)
            raise


class FileContentHandler:
    """Handle file content display and formatting."""
    
    def __init__(self, config: UAgentConfig):
        self.config = config
        self.processor = MemoryEfficientCSVProcessor(config)
    
    def create_file_display(self, file_path: str, file_name: str) -> List[str]:
        """
        Create secure file display with appropriate strategy.
        
        Args:
            file_path: Path to file
            file_name: Display name for file
            
        Returns:
            List of formatted display lines
        """
        try:
            if not os.path.exists(file_path):
                return [
                    f"📄 **{file_name.replace('_', ' ').title()}**:",
                    f"   ⚠️ File not found: {file_path}",
                    ""
                ]
            
            # Get file information
            file_size = self.processor.get_file_size_safe(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Handle different file types
            if file_ext == '.csv':
                return self._create_csv_display(file_path, file_name, file_size)
            elif file_ext in ['.txt', '.log']:
                return self._create_text_display(file_path, file_name, file_size)
            elif file_ext in ['.py', '.r', '.sql']:
                return self._create_code_display(file_path, file_name, file_size)
            else:
                return self._create_generic_display(file_path, file_name, file_size)
                
        except Exception as e:
            logger.error(f"Error creating file display: {e}")
            return [
                f"📄 **{file_name.replace('_', ' ').title()}**:",
                f"   ❌ Error displaying file: {str(e)}",
                ""
            ]
    
    def _create_csv_display(self, file_path: str, file_name: str, file_size: int) -> List[str]:
        """Create display for CSV files."""
        lines = [
            f"📊 **{file_name.replace('_', ' ').title()}** (CSV File - {format_file_size(file_size)}):",
            ""
        ]
        
        try:
            # Read CSV safely
            df = pd.read_csv(file_path, nrows=5)  # Preview only
            
            lines.extend([
                f"   📊 Dataset: {len(df):,} rows × {len(df.columns)} columns (preview)",
                f"   📅 File size: {format_file_size(file_size)}",
                ""
            ])
            
            # Add preview
            lines.extend([
                "📋 **CSV Preview**:",
                "```csv",
                df.to_csv(index=False),
                "```",
                ""
            ])
            
        except Exception as e:
            lines.extend([
                f"   ⚠️ Could not read CSV: {str(e)}",
                f"   📁 File location: {file_path}",
                ""
            ])
        
        return lines
    
    def _create_text_display(self, file_path: str, file_name: str, file_size: int) -> List[str]:
        """Create display for text files."""
        lines = [
            f"📝 **{file_name.replace('_', ' ').title()}** (Text File - {format_file_size(file_size)}):",
            ""
        ]
        
        try:
            if file_size < 10000:  # Small text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines.extend([
                    "```",
                    content,
                    "```",
                    ""
                ])
            else:  # Large text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(5000)  # First 5KB
                
                lines.extend([
                    "📄 **Content Preview** (First 5KB):",
                    "```",
                    content + "\n\n... (truncated for display) ...",
                    "```",
                    "",
                    f"📁 **Full file location**: {file_path}",
                    ""
                ])
                
        except Exception as e:
            lines.extend([
                f"   ⚠️ Could not read text file: {str(e)}",
                f"   📁 File location: {file_path}",
                ""
            ])
        
        return lines
    
    def _create_code_display(self, file_path: str, file_name: str, file_size: int) -> List[str]:
        """Create display for code files."""
        lines = [
            f"💻 **{file_name.replace('_', ' ').title()}** (Code File - {format_file_size(file_size)}):",
            ""
        ]
        
        try:
            # Determine language
            ext = os.path.splitext(file_path)[1].lower()
            lang_map = {'.py': 'python', '.r': 'r', '.sql': 'sql'}
            lang = lang_map.get(ext, 'text')
            
            if file_size < 8000:  # Small code file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines.extend([
                    f"```{lang}",
                    content,
                    "```",
                    ""
                ])
            else:  # Large code file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(4000)  # First 4KB
                
                lines.extend([
                    f"📄 **Code Preview** (First 4KB):",
                    f"```{lang}",
                    content + "\n\n# ... (truncated for display) ...",
                    "```",
                    "",
                    f"📁 **Full file location**: {file_path}",
                    ""
                ])
                
        except Exception as e:
            lines.extend([
                f"   ⚠️ Could not read code file: {str(e)}",
                f"   📁 File location: {file_path}",
                ""
            ])
        
        return lines
    
    def _create_generic_display(self, file_path: str, file_name: str, file_size: int) -> List[str]:
        """Create display for generic files."""
        ext = os.path.splitext(file_path)[1].upper()
        
        return [
            f"📄 **{file_name.replace('_', ' ').title()}** ({ext} File - {format_file_size(file_size)}):",
            f"   📁 File location: {file_path}",
            "",
            "💡 **Note**: This file type cannot be displayed inline for security reasons.",
            ""
        ] 