"""
Advanced text file reading for trading systems.

This module provides comprehensive text file reading capabilities including:
- Multi-format text file parsing (CSV, TSV, custom delimited)
- Large file handling with streaming
- Data validation and type conversion
- Market data format detection
- Error handling and recovery
"""

import os
import sys
import csv
import io
from typing import List, Optional, Dict, Union, Any, Iterator, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import re
import codecs

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.base import CBase, SystemProtocol


class TextFileFormat(Enum):
    """Supported text file formats."""
    CSV = "csv"
    TSV = "tsv"
    PIPE_DELIMITED = "pipe"
    SEMICOLON_DELIMITED = "semicolon"
    SPACE_DELIMITED = "space"
    FIXED_WIDTH = "fixed_width"
    JSON_LINES = "jsonl"
    CUSTOM_DELIMITED = "custom"
    AUTO_DETECT = "auto"


class DataType(Enum):
    """Data type detection and conversion."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    AUTO = "auto"


@dataclass
class ReadResult:
    """Result of text file reading operation."""
    
    success: bool
    file_path: str = ""
    format_detected: TextFileFormat = TextFileFormat.CSV
    encoding_detected: str = "utf-8"
    rows_read: int = 0
    columns_detected: int = 0
    headers: List[str] = field(default_factory=list)
    data: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    file_size_bytes: int = 0


@dataclass
class ParsingConfig:
    """Configuration for text file parsing."""
    
    delimiter: str = ","
    quote_char: str = '"'
    escape_char: str = None
    skip_rows: int = 0
    max_rows: Optional[int] = None
    header_row: int = 0
    encoding: str = "utf-8"
    date_format: str = "%Y-%m-%d"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    decimal_separator: str = "."
    thousands_separator: str = ""
    na_values: List[str] = field(default_factory=lambda: ['', 'NA', 'NULL', 'nan'])
    strip_whitespace: bool = True
    ignore_errors: bool = False
    chunk_size: int = 10000


class CTxtFileReader(CBase):
    """
    Advanced text file reader for trading systems.
    
    Provides comprehensive text file reading capabilities including:
    - Multiple format support (CSV, TSV, custom delimited)
    - Automatic format and encoding detection
    - Large file handling with streaming
    - Data validation and type conversion
    - Error handling and recovery
    """
    
    def __init__(self, system_id: int = 0):
        super().__init__(system_id)
        
        # Default configuration
        self.default_config = ParsingConfig()
        
        # Statistics
        self.files_read = 0
        self.total_rows_read = 0
        self.total_bytes_read = 0
        self.encoding_detections = {}
        self.format_detections = {}
        
        # Supported encodings for auto-detection
        self.supported_encodings = ['utf-8', 'utf-16', 'latin1', 'cp1252', 'iso-8859-1']
        
        # Common date/datetime patterns
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{2}\.\d{2}\.\d{4}', # DD.MM.YYYY
        ]
        
        self.datetime_patterns = [
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',  # MM/DD/YYYY HH:MM:SS
        ]
    
    def initialize(self, system: SystemProtocol, config: ParsingConfig = None) -> 'CTxtFileReader':
        """Initialize text file reader with system and configuration."""
        if config:
            self.default_config = config
        return self
    
    # Main Reading Methods
    def read_file(self, file_path: str, config: ParsingConfig = None) -> ReadResult:
        """Read text file with automatic format detection."""
        start_time = datetime.now()
        
        if not os.path.exists(file_path):
            return ReadResult(
                success=False,
                file_path=file_path,
                errors=["File does not exist"]
            )
        
        config = config or self.default_config
        file_size = os.path.getsize(file_path)
        
        try:
            # Detect encoding if not specified
            encoding = self._detect_encoding(file_path) if config.encoding == "auto" else config.encoding
            
            # Detect format if auto-detect is enabled
            if config.delimiter == "auto":
                detected_format, detected_delimiter = self._detect_format(file_path, encoding)
                config.delimiter = detected_delimiter
                format_detected = detected_format
            else:
                format_detected = self._delimiter_to_format(config.delimiter)
            
            # Read the file
            if file_size > 50 * 1024 * 1024:  # > 50MB - use chunked reading
                data, headers, rows_read, errors, warnings = self._read_large_file(file_path, config, encoding)
            else:
                data, headers, rows_read, errors, warnings = self._read_regular_file(file_path, config, encoding)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update statistics
            self.files_read += 1
            self.total_rows_read += rows_read
            self.total_bytes_read += file_size
            
            self.encoding_detections[encoding] = self.encoding_detections.get(encoding, 0) + 1
            self.format_detections[format_detected.value] = self.format_detections.get(format_detected.value, 0) + 1
            
            return ReadResult(
                success=True,
                file_path=file_path,
                format_detected=format_detected,
                encoding_detected=encoding,
                rows_read=rows_read,
                columns_detected=len(headers) if headers else 0,
                headers=headers,
                data=data,
                errors=errors,
                warnings=warnings,
                execution_time_ms=execution_time,
                file_size_bytes=file_size
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return ReadResult(
                success=False,
                file_path=file_path,
                errors=[str(e)],
                execution_time_ms=execution_time,
                file_size_bytes=file_size
            )
    
    def read_csv(self, file_path: str, **kwargs) -> ReadResult:
        """Read CSV file with pandas backend."""
        config = ParsingConfig(delimiter=",", **kwargs)
        return self.read_file(file_path, config)
    
    def read_tsv(self, file_path: str, **kwargs) -> ReadResult:
        """Read TSV file with pandas backend."""
        config = ParsingConfig(delimiter="\t", **kwargs)
        return self.read_file(file_path, config)
    
    def read_market_data(self, file_path: str, **kwargs) -> ReadResult:
        """Read market data with automatic column detection and validation."""
        result = self.read_file(file_path, **kwargs)
        
        if result.success and result.data is not None:
            # Try to identify market data columns
            df = result.data if isinstance(result.data, pd.DataFrame) else pd.DataFrame(result.data)
            
            try:
                # Auto-detect common market data columns
                market_columns = self._detect_market_data_columns(df)
                
                if market_columns:
                    # Rename columns to standard names
                    df = df.rename(columns=market_columns)
                    
                    # Validate and convert data types
                    df = self._validate_market_data(df)
                    
                    result.data = df
                    result.warnings.append("Market data columns auto-detected and validated")
                else:
                    result.warnings.append("Could not auto-detect market data format")
                    
            except Exception as e:
                result.warnings.append(f"Market data validation failed: {str(e)}")
        
        return result
    
    # Streaming and Large File Handling
    def read_file_chunks(self, file_path: str, config: ParsingConfig = None) -> Iterator[pd.DataFrame]:
        """Read file in chunks for memory-efficient processing."""
        config = config or self.default_config
        
        try:
            encoding = self._detect_encoding(file_path) if config.encoding == "auto" else config.encoding
            
            # Use pandas chunked reading
            reader = pd.read_csv(
                file_path,
                delimiter=config.delimiter,
                quotechar=config.quote_char,
                escapechar=config.escape_char,
                skiprows=config.skip_rows,
                nrows=config.max_rows,
                header=config.header_row if config.header_row >= 0 else None,
                encoding=encoding,
                na_values=config.na_values,
                chunksize=config.chunk_size,
                error_bad_lines=not config.ignore_errors,
                warn_bad_lines=True
            )
            
            for chunk in reader:
                if config.strip_whitespace:
                    # Strip whitespace from string columns
                    string_columns = chunk.select_dtypes(include=['object']).columns
                    chunk[string_columns] = chunk[string_columns].apply(lambda x: x.str.strip() if x.dtype == "object" else x)
                
                yield chunk
                
        except Exception as e:
            # Return empty iterator on error
            return iter([])
    
    def count_lines(self, file_path: str, encoding: str = "utf-8") -> int:
        """Efficiently count lines in text file."""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
    
    # Format and Encoding Detection
    def _detect_encoding(self, file_path: str, sample_size: int = 10000) -> str:
        """Detect file encoding by trying different encodings."""
        for encoding in self.supported_encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(sample_size)
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # Fallback to utf-8 with error handling
        return 'utf-8'
    
    def _detect_format(self, file_path: str, encoding: str) -> Tuple[TextFileFormat, str]:
        """Detect file format by analyzing delimiters."""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                # Read first few lines for analysis
                sample_lines = [f.readline().strip() for _ in range(5) if f.readline()]
            
            if not sample_lines:
                return TextFileFormat.CSV, ","
            
            # Count potential delimiters
            delimiter_counts = {
                ',': sum(line.count(',') for line in sample_lines),
                '\t': sum(line.count('\t') for line in sample_lines),
                ';': sum(line.count(';') for line in sample_lines),
                '|': sum(line.count('|') for line in sample_lines),
            }
            
            # Find most common delimiter
            best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            
            if delimiter_counts[best_delimiter] == 0:
                # Try space delimiter
                space_count = sum(len(line.split()) - 1 for line in sample_lines)
                if space_count > 0:
                    return TextFileFormat.SPACE_DELIMITED, ' '
                else:
                    return TextFileFormat.CSV, ","  # Default fallback
            
            # Map delimiter to format
            format_map = {
                ',': TextFileFormat.CSV,
                '\t': TextFileFormat.TSV,
                ';': TextFileFormat.SEMICOLON_DELIMITED,
                '|': TextFileFormat.PIPE_DELIMITED,
            }
            
            return format_map.get(best_delimiter, TextFileFormat.CSV), best_delimiter
            
        except Exception:
            return TextFileFormat.CSV, ","
    
    def _delimiter_to_format(self, delimiter: str) -> TextFileFormat:
        """Convert delimiter character to format enum."""
        format_map = {
            ',': TextFileFormat.CSV,
            '\t': TextFileFormat.TSV,
            ';': TextFileFormat.SEMICOLON_DELIMITED,
            '|': TextFileFormat.PIPE_DELIMITED,
            ' ': TextFileFormat.SPACE_DELIMITED,
        }
        return format_map.get(delimiter, TextFileFormat.CUSTOM_DELIMITED)
    
    # File Reading Implementation
    def _read_regular_file(self, file_path: str, config: ParsingConfig, encoding: str) -> Tuple[pd.DataFrame, List[str], int, List[str], List[str]]:
        """Read regular-sized file using pandas."""
        errors = []
        warnings = []
        
        try:
            df = pd.read_csv(
                file_path,
                delimiter=config.delimiter,
                quotechar=config.quote_char,
                escapechar=config.escape_char,
                skiprows=config.skip_rows,
                nrows=config.max_rows,
                header=config.header_row if config.header_row >= 0 else None,
                encoding=encoding,
                na_values=config.na_values,
                error_bad_lines=not config.ignore_errors,
                warn_bad_lines=True,
                thousands=config.thousands_separator if config.thousands_separator else None
            )
            
            headers = list(df.columns) if df.columns is not None else []
            
            # Strip whitespace if requested
            if config.strip_whitespace:
                string_columns = df.select_dtypes(include=['object']).columns
                df[string_columns] = df[string_columns].apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            
            return df, headers, len(df), errors, warnings
            
        except pd.errors.EmptyDataError:
            return pd.DataFrame(), [], 0, ["File is empty"], warnings
        except pd.errors.ParserError as e:
            errors.append(f"Parser error: {str(e)}")
            return pd.DataFrame(), [], 0, errors, warnings
        except Exception as e:
            errors.append(f"Reading error: {str(e)}")
            return pd.DataFrame(), [], 0, errors, warnings
    
    def _read_large_file(self, file_path: str, config: ParsingConfig, encoding: str) -> Tuple[pd.DataFrame, List[str], int, List[str], List[str]]:
        """Read large file using chunked processing."""
        errors = []
        warnings = []
        chunks = []
        total_rows = 0
        headers = []
        
        try:
            for i, chunk in enumerate(self.read_file_chunks(file_path, config)):
                chunks.append(chunk)
                total_rows += len(chunk)
                
                if i == 0:
                    headers = list(chunk.columns)
                
                # Limit memory usage by concatenating periodically
                if len(chunks) > 10:
                    combined = pd.concat(chunks, ignore_index=True)
                    chunks = [combined]
            
            if chunks:
                final_df = pd.concat(chunks, ignore_index=True) if len(chunks) > 1 else chunks[0]
                return final_df, headers, total_rows, errors, warnings
            else:
                return pd.DataFrame(), [], 0, ["No data found"], warnings
                
        except Exception as e:
            errors.append(f"Large file reading error: {str(e)}")
            return pd.DataFrame(), [], 0, errors, warnings
    
    # Market Data Specific Methods
    def _detect_market_data_columns(self, df: pd.DataFrame) -> Optional[Dict[str, str]]:
        """Detect market data columns by analyzing column names and data."""
        if df.empty:
            return None
        
        columns_lower = [col.lower() for col in df.columns]
        mapping = {}
        
        # Common column name patterns
        patterns = {
            'date': ['date', 'time', 'datetime', 'timestamp', 'tarih'],
            'open': ['open', 'o', 'opening', 'açılış'],
            'high': ['high', 'h', 'maximum', 'max', 'yüksek'],
            'low': ['low', 'l', 'minimum', 'min', 'düşük'],
            'close': ['close', 'c', 'closing', 'kapanış'],
            'volume': ['volume', 'vol', 'v', 'hacim', 'miktar']
        }
        
        for standard_name, possible_names in patterns.items():
            for col_idx, col_name in enumerate(columns_lower):
                if any(pattern in col_name for pattern in possible_names):
                    mapping[df.columns[col_idx]] = standard_name
                    break
        
        # Must have at least OHLC to be considered market data
        required = ['open', 'high', 'low', 'close']
        if all(col in mapping.values() for col in required):
            return mapping
        
        return None
    
    def _validate_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert market data types."""
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Convert price columns to numeric
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in price_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Basic validation - high >= low, etc.
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Remove rows where high < low (invalid data)
            invalid_rows = df['high'] < df['low']
            if invalid_rows.any():
                df = df[~invalid_rows].copy()
        
        return df
    
    # Utility Methods
    def detect_data_types(self, file_path: str, sample_rows: int = 1000) -> Dict[str, str]:
        """Analyze file to detect appropriate data types for each column."""
        try:
            result = self.read_file(file_path, ParsingConfig(max_rows=sample_rows))
            if result.success and result.data is not None:
                df = result.data
                dtypes = {}
                
                for col in df.columns:
                    series = df[col].dropna()
                    if series.empty:
                        dtypes[col] = 'string'
                        continue
                    
                    # Try to detect data type
                    sample_values = series.head(100).astype(str)
                    
                    # Check for date/datetime
                    if any(re.match(pattern, val) for pattern in self.date_patterns for val in sample_values):
                        if any(re.match(pattern, val) for pattern in self.datetime_patterns for val in sample_values):
                            dtypes[col] = 'datetime'
                        else:
                            dtypes[col] = 'date'
                    # Check for numeric
                    elif series.dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(series):
                        if series.dtype == 'int64' or all(str(val).replace('.', '').replace('-', '').isdigit() for val in sample_values):
                            dtypes[col] = 'integer'
                        else:
                            dtypes[col] = 'float'
                    # Check for boolean
                    elif set(sample_values.str.lower().unique()) <= {'true', 'false', '1', '0', 'yes', 'no'}:
                        dtypes[col] = 'boolean'
                    else:
                        dtypes[col] = 'string'
                
                return dtypes
        except Exception:
            pass
        
        return {}
    
    def get_file_preview(self, file_path: str, num_rows: int = 5) -> ReadResult:
        """Get a preview of the file (first few rows)."""
        config = ParsingConfig(max_rows=num_rows)
        return self.read_file(file_path, config)
    
    # Statistics and Information
    def get_statistics(self) -> Dict[str, Any]:
        """Get reading statistics."""
        return {
            "files_read": self.files_read,
            "total_rows_read": self.total_rows_read,
            "total_bytes_read": self.total_bytes_read,
            "encoding_detections": dict(self.encoding_detections),
            "format_detections": dict(self.format_detections),
            "supported_encodings": self.supported_encodings
        }
    
    def __str__(self) -> str:
        """String representation of text file reader."""
        return (
            f"CTxtFileReader(files_read={self.files_read}, "
            f"rows_read={self.total_rows_read}, "
            f"bytes_read={self.total_bytes_read})"
        )