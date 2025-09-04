"""
Advanced INI file configuration management for trading systems.

This module provides comprehensive INI file handling including:
- Reading and writing INI configuration files
- Section and key management
- Data type conversion and validation
- Configuration templates and defaults
- Environment variable substitution
"""

import os
import sys
import configparser
from typing import List, Optional, Dict, Union, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
import json

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.base import CBase, SystemProtocol


class IniValueType(Enum):
    """INI value data types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    PATH = "path"
    AUTO = "auto"


class IniOperationType(Enum):
    """INI file operation types."""
    READ = "read"
    WRITE = "write"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    BACKUP = "backup"
    RESTORE = "restore"


@dataclass
class IniKeyValue:
    """INI key-value pair with metadata."""
    
    key: str
    value: Any
    section: str = "DEFAULT"
    value_type: IniValueType = IniValueType.STRING
    comment: str = ""
    is_required: bool = False
    default_value: Any = None
    valid_values: List[Any] = field(default_factory=list)
    
    def __post_init__(self):
        if self.default_value is None:
            self.default_value = self.value


@dataclass
class IniSection:
    """INI section with key-value pairs."""
    
    name: str
    description: str = ""
    keys: Dict[str, IniKeyValue] = field(default_factory=dict)
    is_required: bool = True
    
    def add_key(self, key: str, value: Any, **kwargs) -> 'IniSection':
        """Add key-value pair to section."""
        self.keys[key] = IniKeyValue(key=key, value=value, section=self.name, **kwargs)
        return self
    
    def get_key(self, key: str) -> Optional[IniKeyValue]:
        """Get key-value pair from section."""
        return self.keys.get(key)
    
    def remove_key(self, key: str) -> bool:
        """Remove key from section."""
        return self.keys.pop(key, None) is not None


@dataclass
class IniOperationResult:
    """Result of INI file operation."""
    
    success: bool
    operation: IniOperationType
    file_path: str = ""
    sections_processed: int = 0
    keys_processed: int = 0
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    data: Any = None


class CIniFile(CBase):
    """
    Advanced INI file configuration management.
    
    Provides comprehensive INI file operations including:
    - Reading and writing INI files with type conversion
    - Section and key management with validation
    - Configuration templates and defaults
    - Environment variable substitution
    - Backup and restore capabilities
    """
    
    def __init__(self, system_id: int = 0):
        super().__init__(system_id)
        
        # Configuration
        self.default_encoding = "utf-8"
        self.comment_prefixes = ('#', ';')
        self.allow_no_value = False
        self.interpolation = True
        self.case_sensitive = False
        
        # INI parser configuration
        self.parser = configparser.ConfigParser(
            allow_no_value=self.allow_no_value,
            interpolation=configparser.ExtendedInterpolation() if self.interpolation else None
        )
        
        if not self.case_sensitive:
            self.parser.optionxform = str  # Preserve case
        
        # Statistics
        self.files_processed = 0
        self.sections_created = 0
        self.keys_written = 0
        
        # Internal state
        self.current_file_path = ""
        self.sections: Dict[str, IniSection] = {}
        self.file_loaded = False
        
        # Environment variable pattern
        self.env_var_pattern = re.compile(r'\$\{(\w+)\}')
    
    def initialize(self, system: SystemProtocol, file_path: str = None) -> 'CIniFile':
        """Initialize INI file handler with optional file loading."""
        if file_path:
            self.load_file(file_path)
        return self
    
    # File Operations
    def load_file(self, file_path: str, create_if_missing: bool = True) -> IniOperationResult:
        """Load INI file into memory."""
        start_time = datetime.now()
        
        if not os.path.exists(file_path) and not create_if_missing:
            return IniOperationResult(
                success=False,
                operation=IniOperationType.READ,
                file_path=file_path,
                error_message="File does not exist and create_if_missing is False"
            )
        
        try:
            self.current_file_path = file_path
            
            if os.path.exists(file_path):
                # Read existing file
                self.parser.read(file_path, encoding=self.default_encoding)
                self._parse_sections_from_parser()
            else:
                # Create new file
                self.parser.clear()
                self.sections.clear()
            
            self.file_loaded = True
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            self.files_processed += 1
            
            return IniOperationResult(
                success=True,
                operation=IniOperationType.READ if os.path.exists(file_path) else IniOperationType.CREATE,
                file_path=file_path,
                sections_processed=len(self.sections),
                keys_processed=sum(len(section.keys) for section in self.sections.values()),
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return IniOperationResult(
                success=False,
                operation=IniOperationType.READ,
                file_path=file_path,
                error_message=str(e),
                execution_time_ms=execution_time
            )
    
    def save_file(self, file_path: str = None, create_backup: bool = True) -> IniOperationResult:
        """Save current configuration to INI file."""
        start_time = datetime.now()
        target_path = file_path or self.current_file_path
        
        if not target_path:
            return IniOperationResult(
                success=False,
                operation=IniOperationType.WRITE,
                error_message="No file path specified"
            )
        
        try:
            # Create backup if requested
            if create_backup and os.path.exists(target_path):
                self._create_backup(target_path)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Update parser from sections
            self._update_parser_from_sections()
            
            # Write to file
            with open(target_path, 'w', encoding=self.default_encoding) as f:
                self.parser.write(f)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return IniOperationResult(
                success=True,
                operation=IniOperationType.WRITE,
                file_path=target_path,
                sections_processed=len(self.sections),
                keys_processed=sum(len(section.keys) for section in self.sections.values()),
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return IniOperationResult(
                success=False,
                operation=IniOperationType.WRITE,
                file_path=target_path,
                error_message=str(e),
                execution_time_ms=execution_time
            )
    
    # Section Management
    def add_section(self, section_name: str, description: str = "", is_required: bool = True) -> bool:
        """Add new section to configuration."""
        try:
            if section_name not in self.sections:
                self.sections[section_name] = IniSection(
                    name=section_name,
                    description=description,
                    is_required=is_required
                )
                self.sections_created += 1
                return True
            return False
        except Exception:
            return False
    
    def remove_section(self, section_name: str) -> bool:
        """Remove section from configuration."""
        try:
            if section_name in self.sections:
                del self.sections[section_name]
                return True
            return False
        except Exception:
            return False
    
    def has_section(self, section_name: str) -> bool:
        """Check if section exists."""
        return section_name in self.sections
    
    def get_sections(self) -> List[str]:
        """Get list of all section names."""
        return list(self.sections.keys())
    
    # Key-Value Operations
    def set_value(
        self, 
        section: str, 
        key: str, 
        value: Any, 
        value_type: IniValueType = IniValueType.AUTO,
        comment: str = ""
    ) -> bool:
        """Set value in specified section and key."""
        try:
            # Create section if it doesn't exist
            if section not in self.sections:
                self.add_section(section)
            
            # Auto-detect value type
            if value_type == IniValueType.AUTO:
                value_type = self._detect_value_type(value)
            
            # Convert value to appropriate string representation
            string_value = self._value_to_string(value, value_type)
            
            # Add key to section
            self.sections[section].add_key(
                key=key,
                value=string_value,
                value_type=value_type,
                comment=comment
            )
            
            self.keys_written += 1
            return True
            
        except Exception:
            return False
    
    def get_value(
        self, 
        section: str, 
        key: str, 
        default: Any = None,
        value_type: IniValueType = IniValueType.AUTO
    ) -> Any:
        """Get value from specified section and key."""
        try:
            if section in self.sections and key in self.sections[section].keys:
                key_value = self.sections[section].keys[key]
                
                # Use stored type if available, otherwise auto-detect
                if value_type == IniValueType.AUTO:
                    value_type = key_value.value_type
                
                return self._string_to_value(key_value.value, value_type)
            
            return default
            
        except Exception:
            return default
    
    def get_string(self, section: str, key: str, default: str = "") -> str:
        """Get string value."""
        return self.get_value(section, key, default, IniValueType.STRING)
    
    def get_int(self, section: str, key: str, default: int = 0) -> int:
        """Get integer value."""
        return self.get_value(section, key, default, IniValueType.INTEGER)
    
    def get_float(self, section: str, key: str, default: float = 0.0) -> float:
        """Get float value."""
        return self.get_value(section, key, default, IniValueType.FLOAT)
    
    def get_bool(self, section: str, key: str, default: bool = False) -> bool:
        """Get boolean value."""
        return self.get_value(section, key, default, IniValueType.BOOLEAN)
    
    def get_list(self, section: str, key: str, default: List[str] = None) -> List[str]:
        """Get list value."""
        if default is None:
            default = []
        return self.get_value(section, key, default, IniValueType.LIST)
    
    def remove_key(self, section: str, key: str) -> bool:
        """Remove key from section."""
        try:
            if section in self.sections:
                return self.sections[section].remove_key(key)
            return False
        except Exception:
            return False
    
    def has_key(self, section: str, key: str) -> bool:
        """Check if key exists in section."""
        return (section in self.sections and 
                key in self.sections[section].keys)
    
    # Configuration Templates
    def create_trading_system_template(self) -> bool:
        """Create template configuration for trading system."""
        try:
            # Database section
            self.add_section("Database", "Database connection settings")
            self.set_value("Database", "host", "localhost", IniValueType.STRING, "Database server host")
            self.set_value("Database", "port", 5432, IniValueType.INTEGER, "Database server port")
            self.set_value("Database", "name", "trading_db", IniValueType.STRING, "Database name")
            self.set_value("Database", "username", "trader", IniValueType.STRING, "Database username")
            self.set_value("Database", "password", "password", IniValueType.STRING, "Database password")
            
            # Trading section
            self.add_section("Trading", "Trading system settings")
            self.set_value("Trading", "initial_balance", 100000.0, IniValueType.FLOAT, "Initial trading balance")
            self.set_value("Trading", "max_position_size", 10000.0, IniValueType.FLOAT, "Maximum position size")
            self.set_value("Trading", "commission_rate", 0.001, IniValueType.FLOAT, "Commission rate per trade")
            self.set_value("Trading", "enable_live_trading", False, IniValueType.BOOLEAN, "Enable live trading")
            self.set_value("Trading", "supported_symbols", "EURUSD,GBPUSD,USDJPY", IniValueType.LIST, "Supported trading symbols")
            
            # Risk Management section
            self.add_section("RiskManagement", "Risk management parameters")
            self.set_value("RiskManagement", "max_daily_loss", 5000.0, IniValueType.FLOAT, "Maximum daily loss")
            self.set_value("RiskManagement", "stop_loss_percentage", 2.0, IniValueType.FLOAT, "Default stop loss percentage")
            self.set_value("RiskManagement", "take_profit_percentage", 4.0, IniValueType.FLOAT, "Default take profit percentage")
            
            # Paths section
            self.add_section("Paths", "File and directory paths")
            self.set_value("Paths", "data_directory", "./data", IniValueType.PATH, "Market data directory")
            self.set_value("Paths", "log_directory", "./logs", IniValueType.PATH, "Log files directory")
            self.set_value("Paths", "config_directory", "./config", IniValueType.PATH, "Configuration files directory")
            
            # Logging section
            self.add_section("Logging", "Logging configuration")
            self.set_value("Logging", "log_level", "INFO", IniValueType.STRING, "Logging level")
            self.set_value("Logging", "log_to_file", True, IniValueType.BOOLEAN, "Enable file logging")
            self.set_value("Logging", "max_log_size_mb", 10, IniValueType.INTEGER, "Maximum log file size")
            
            return True
            
        except Exception:
            return False
    
    # Advanced Features
    def substitute_environment_variables(self, value: str) -> str:
        """Substitute environment variables in value string."""
        try:
            def replace_env_var(match):
                env_var = match.group(1)
                return os.environ.get(env_var, match.group(0))
            
            return self.env_var_pattern.sub(replace_env_var, value)
            
        except Exception:
            return value
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate current configuration against requirements."""
        errors = []
        
        try:
            for section_name, section in self.sections.items():
                if section.is_required:
                    # Check if required keys exist
                    for key_name, key_value in section.keys.items():
                        if key_value.is_required and not key_value.value:
                            errors.append(f"Required key '{key_name}' in section '{section_name}' is empty")
                        
                        # Validate against allowed values
                        if key_value.valid_values:
                            actual_value = self._string_to_value(key_value.value, key_value.value_type)
                            if actual_value not in key_value.valid_values:
                                errors.append(f"Invalid value '{actual_value}' for key '{key_name}' in section '{section_name}'. Valid values: {key_value.valid_values}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    def merge_from_dict(self, config_dict: Dict[str, Dict[str, Any]]) -> bool:
        """Merge configuration from dictionary."""
        try:
            for section_name, section_data in config_dict.items():
                if not self.has_section(section_name):
                    self.add_section(section_name)
                
                for key, value in section_data.items():
                    self.set_value(section_name, key, value)
            
            return True
            
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert configuration to dictionary."""
        try:
            result = {}
            
            for section_name, section in self.sections.items():
                result[section_name] = {}
                
                for key_name, key_value in section.keys.items():
                    result[section_name][key_name] = self._string_to_value(
                        key_value.value, key_value.value_type
                    )
            
            return result
            
        except Exception:
            return {}
    
    def export_to_json(self, file_path: str) -> bool:
        """Export configuration to JSON file."""
        try:
            config_dict = self.to_dict()
            with open(file_path, 'w', encoding=self.default_encoding) as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception:
            return False
    
    def import_from_json(self, file_path: str) -> bool:
        """Import configuration from JSON file."""
        try:
            with open(file_path, 'r', encoding=self.default_encoding) as f:
                config_dict = json.load(f)
            return self.merge_from_dict(config_dict)
        except Exception:
            return False
    
    # Internal Methods
    def _parse_sections_from_parser(self):
        """Parse sections from ConfigParser into internal structure."""
        self.sections.clear()
        
        for section_name in self.parser.sections():
            section = IniSection(name=section_name)
            
            for key in self.parser[section_name]:
                value = self.parser[section_name][key]
                value_type = self._detect_value_type(value)
                
                section.add_key(
                    key=key,
                    value=value,
                    value_type=value_type
                )
            
            self.sections[section_name] = section
    
    def _update_parser_from_sections(self):
        """Update ConfigParser from internal sections."""
        self.parser.clear()
        
        for section_name, section in self.sections.items():
            self.parser.add_section(section_name)
            
            for key_name, key_value in section.keys.items():
                self.parser.set(section_name, key_name, str(key_value.value))
    
    def _detect_value_type(self, value: Any) -> IniValueType:
        """Auto-detect value type."""
        if isinstance(value, bool):
            return IniValueType.BOOLEAN
        elif isinstance(value, int):
            return IniValueType.INTEGER
        elif isinstance(value, float):
            return IniValueType.FLOAT
        elif isinstance(value, (list, tuple)):
            return IniValueType.LIST
        elif isinstance(value, str):
            # Try to detect if it's a path
            if os.path.sep in value or value.startswith('.'):
                return IniValueType.PATH
            # Try to detect if it's a list (comma-separated)
            elif ',' in value:
                return IniValueType.LIST
            # Try to detect boolean
            elif value.lower() in ('true', 'false', 'yes', 'no', '1', '0'):
                return IniValueType.BOOLEAN
            # Try to detect numeric
            else:
                try:
                    int(value)
                    return IniValueType.INTEGER
                except ValueError:
                    try:
                        float(value)
                        return IniValueType.FLOAT
                    except ValueError:
                        pass
        
        return IniValueType.STRING
    
    def _value_to_string(self, value: Any, value_type: IniValueType) -> str:
        """Convert value to string representation for INI file."""
        if value_type == IniValueType.LIST:
            if isinstance(value, (list, tuple)):
                return ','.join(str(item) for item in value)
            elif isinstance(value, str):
                return value
        elif value_type == IniValueType.BOOLEAN:
            if isinstance(value, bool):
                return 'true' if value else 'false'
            elif isinstance(value, str):
                return value.lower()
        
        return str(value)
    
    def _string_to_value(self, string_value: str, value_type: IniValueType) -> Any:
        """Convert string value to appropriate Python type."""
        try:
            if value_type == IniValueType.INTEGER:
                return int(string_value)
            elif value_type == IniValueType.FLOAT:
                return float(string_value)
            elif value_type == IniValueType.BOOLEAN:
                return string_value.lower() in ('true', 'yes', '1', 'on')
            elif value_type == IniValueType.LIST:
                return [item.strip() for item in string_value.split(',')]
            elif value_type == IniValueType.PATH:
                # Apply environment variable substitution
                return self.substitute_environment_variables(string_value)
            else:
                return string_value
        except (ValueError, TypeError):
            return string_value
    
    def _create_backup(self, file_path: str) -> bool:
        """Create backup of existing INI file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{file_path}.{timestamp}.bak"
            
            import shutil
            shutil.copy2(file_path, backup_path)
            return True
        except Exception:
            return False
    
    # Statistics and Information
    def get_statistics(self) -> Dict[str, Any]:
        """Get INI file operation statistics."""
        return {
            "files_processed": self.files_processed,
            "sections_created": self.sections_created,
            "keys_written": self.keys_written,
            "current_file": self.current_file_path,
            "file_loaded": self.file_loaded,
            "total_sections": len(self.sections),
            "total_keys": sum(len(section.keys) for section in self.sections.values())
        }
    
    def __str__(self) -> str:
        """String representation of INI file handler."""
        return (
            f"CIniFile(file='{os.path.basename(self.current_file_path) if self.current_file_path else 'None'}', "
            f"sections={len(self.sections)}, "
            f"keys={sum(len(section.keys) for section in self.sections.values())})"
        )