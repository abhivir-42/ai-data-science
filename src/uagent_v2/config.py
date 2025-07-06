"""
Configuration module for the enhanced uAgent implementation.

This module centralizes all configuration values that were previously hard-coded
throughout the application, making the system more maintainable and configurable.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class UAgentConfig:
    """Configuration for the uAgent with environment variable support."""
    
    # Core uAgent settings
    port: int = 8102
    name: str = "AI Data Science Agent"
    description: str = "🤖 AI Data Analysis Agent - Send me a CSV URL and analysis request. I'll clean your data, engineer features, and build ML models."
    
    # File processing limits
    max_file_size_mb: int = 50
    small_file_threshold_kb: int = 50
    medium_file_threshold_kb: int = 200
    max_display_rows: int = 10
    max_display_actions: int = 10
    
    # Network settings
    upload_timeout_seconds: int = 30
    request_timeout_seconds: int = 60
    max_retries: int = 3
    
    # Session management
    session_timeout_hours: int = 1
    max_concurrent_sessions: int = 10
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_async: bool = False
    max_workers: int = 4
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Output settings
    output_dir: str = "output/data_analysis_uagent/"
    temp_dir: str = "temp/"
    
    # ML settings
    intent_parser_model: str = "gpt-4o-mini"
    enable_ml_verbose: bool = False
    
    # Security settings
    allowed_file_types: tuple = (".csv",)
    max_upload_attempts: int = 3
    enable_file_validation: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")
        
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        
        if self.session_timeout_hours <= 0:
            raise ValueError("session_timeout_hours must be positive")
        
        # Ensure directories exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "UAgentConfig":
        """Create configuration from environment variables."""
        return cls(
            port=int(os.getenv("UAGENT_PORT", "8102")),
            name=os.getenv("UAGENT_NAME", "AI Data Science Agent"),
            description=os.getenv("UAGENT_DESCRIPTION", "🤖 AI Data Analysis Chatbot - Send me a CSV URL and analysis request."),
            
            # File processing
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "50")),
            small_file_threshold_kb=int(os.getenv("SMALL_FILE_THRESHOLD_KB", "50")),
            medium_file_threshold_kb=int(os.getenv("MEDIUM_FILE_THRESHOLD_KB", "200")),
            
            # Network
            upload_timeout_seconds=int(os.getenv("UPLOAD_TIMEOUT_SECONDS", "30")),
            request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            
            # Session
            session_timeout_hours=int(os.getenv("SESSION_TIMEOUT_HOURS", "1")),
            max_concurrent_sessions=int(os.getenv("MAX_CONCURRENT_SESSIONS", "10")),
            
            # Performance
            enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
            enable_async=os.getenv("ENABLE_ASYNC", "false").lower() == "true",
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            
            # Logging
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            
            # Output
            output_dir=os.getenv("OUTPUT_DIR", "output/data_analysis_uagent/"),
            temp_dir=os.getenv("TEMP_DIR", "temp/"),
            
            # ML
            intent_parser_model=os.getenv("INTENT_PARSER_MODEL", "gpt-4o-mini"),
            enable_ml_verbose=os.getenv("ENABLE_ML_VERBOSE", "false").lower() == "true",
            
            # Security
            max_upload_attempts=int(os.getenv("MAX_UPLOAD_ATTEMPTS", "3")),
            enable_file_validation=os.getenv("ENABLE_FILE_VALIDATION", "true").lower() == "true",
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def get_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    def get_small_file_size_bytes(self) -> int:
        """Get small file threshold in bytes."""
        return self.small_file_threshold_kb * 1024
    
    def get_medium_file_size_bytes(self) -> int:
        """Get medium file threshold in bytes."""
        return self.medium_file_threshold_kb * 1024
    
    def get_session_timeout_seconds(self) -> int:
        """Get session timeout in seconds."""
        return self.session_timeout_hours * 3600
    
    def validate(self) -> bool:
        """Validate the current configuration."""
        try:
            if self.max_file_size_mb <= 0:
                raise ValueError("max_file_size_mb must be positive")
            
            if self.port <= 0 or self.port > 65535:
                raise ValueError("port must be between 1 and 65535")
            
            if self.session_timeout_hours <= 0:
                raise ValueError("session_timeout_hours must be positive")
            
            if self.small_file_threshold_kb <= 0:
                raise ValueError("small_file_threshold_kb must be positive")
            
            if self.medium_file_threshold_kb <= self.small_file_threshold_kb:
                raise ValueError("medium_file_threshold_kb must be larger than small_file_threshold_kb")
            
            return True
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def update_runtime_settings(self, **kwargs) -> None:
        """Update configuration settings at runtime."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"UAgentConfig has no attribute '{key}'")
        
        # Re-validate after updates
        self.validate()
    
    @property
    def agent_port(self) -> int:
        """Get agent port (alias for backward compatibility)."""
        return self.port
    
    @property
    def rate_limit_cooldown_seconds(self) -> int:
        """Get rate limit cooldown time in seconds."""
        return 60  # Default cooldown time
    
    @property
    def upload_rate_limit_mb_per_minute(self) -> int:
        """Upload rate limit in MB per minute."""
        return 100  # Default rate limit
    
    @property
    def request_rate_limit_per_minute(self) -> int:
        """Request rate limit per minute."""
        return 60  # Default rate limit


# Properties for backward compatibility
# (Removed duplicate property definitions)

# Create default configuration instance
default_config = UAgentConfig.from_env() 