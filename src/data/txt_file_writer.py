"""
Advanced text file writing for trading systems.

This module provides comprehensive text file writing capabilities including:
- Multi-format text file writing (CSV, TSV, custom delimited)
- Data formatting and validation
- Append and batch writing modes
- Trading data export formatting
- Error handling and recovery
"""

import os
import sys
import csv
import json
from typing import List, Optional, Dict, Union, Any, TextIO
from datetime import datetime, date
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.base import CBase, SystemProtocol


class WriteMode(Enum):
    """File writing modes."""
    WRITE = "w"  # Overwrite file
    APPEND = "a"  # Append to file
    WRITE_BINARY = "wb"
    APPEND_BINARY = "ab"
    EXCLUSIVE = "x"  # Fail if file exists


class TextFormat(Enum):
    """Text output formats."""
    CSV = "csv"
    TSV = "tsv"
    PIPE_DELIMITED = "pipe"
    SEMICOLON_DELIMITED = "semicolon"
    SPACE_DELIMITED = "space"
    FIXED_WIDTH = "fixed_width"
    JSON_LINES = "jsonl"
    JSON = "json"
    CUSTOM = "custom"


@dataclass
class WriteResult:
    """Result of text file writing operation."""
    
    success: bool
    file_path: str = ""
    rows_written: int = 0
    columns_written: int = 0
    bytes_written: int = 0
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    format_used: TextFormat = TextFormat.CSV


@dataclass
class WriterConfig:
    """Configuration for text file writing."""
    
    format: TextFormat = TextFormat.CSV
    delimiter: str = ","
    quote_char: str = '"'
    escape_char: str = None
    line_terminator: str = "\n"
    encoding: str = "utf-8"
    write_header: bool = True
    date_format: str = "%Y-%m-%d"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    decimal_places: int = 6
    na_representation: str = ""
    quote_all: bool = False
    create_backup: bool = False
    buffer_size: int = 8192


class CTxtFileWriter(CBase):
    """
    Advanced text file writer for trading systems.
    
    Provides comprehensive text file writing capabilities including:
    - Multiple format support (CSV, TSV, JSON, custom delimited)
    - Flexible data formatting and validation
    - Append and batch writing modes
    - Trading data export with proper formatting
    - Error handling and recovery
    """
    
    def __init__(self, system_id: int = 0):
        super().__init__(system_id)
        
        # Default configuration
        self.default_config = WriterConfig()
        
        # Statistics
        self.files_written = 0
        self.total_rows_written = 0
        self.total_bytes_written = 0
        self.format_usage = {}
        
        # Open file handles for batch writing
        self._open_files: Dict[str, TextIO] = {}
        self._file_configs: Dict[str, WriterConfig] = {}
    
    def initialize(self, system: SystemProtocol, config: WriterConfig = None) -> 'CTxtFileWriter':
        """Initialize text file writer with system and configuration."""
        if config:
            self.default_config = config
        return self
    
    # Main Writing Methods
    def write_file(
        self, 
        data: Union[pd.DataFrame, List[List[Any]], List[Dict]], 
        file_path: str,
        config: WriterConfig = None,
        mode: WriteMode = WriteMode.WRITE
    ) -> WriteResult:
        """Write data to text file with specified format."""
        start_time = datetime.now()
        config = config or self.default_config
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Create backup if requested
            if config.create_backup and os.path.exists(file_path):
                self._create_backup(file_path)
            
            # Convert data to appropriate format
            df, headers = self._prepare_data(data, config)
            
            # Write based on format
            if config.format == TextFormat.JSON:
                result = self._write_json(df, file_path, config, mode)
            elif config.format == TextFormat.JSON_LINES:
                result = self._write_jsonl(df, file_path, config, mode)
            elif config.format == TextFormat.FIXED_WIDTH:
                result = self._write_fixed_width(df, file_path, config, mode)
            else:
                result = self._write_delimited(df, file_path, config, mode)
            
            # Update statistics
            if result.success:
                self.files_written += 1
                self.total_rows_written += result.rows_written
                self.total_bytes_written += result.bytes_written
                
                format_key = config.format.value
                self.format_usage[format_key] = self.format_usage.get(format_key, 0) + 1
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return WriteResult(
                success=False,
                file_path=file_path,
                error_message=str(e),
                execution_time_ms=execution_time,
                format_used=config.format
            )
    
    def write_csv(self, data: Union[pd.DataFrame, List[List[Any]]], file_path: str, **kwargs) -> WriteResult:
        """Write CSV file."""
        config = WriterConfig(format=TextFormat.CSV, delimiter=",", **kwargs)
        return self.write_file(data, file_path, config)
    
    def write_tsv(self, data: Union[pd.DataFrame, List[List[Any]]], file_path: str, **kwargs) -> WriteResult:
        """Write TSV file."""
        config = WriterConfig(format=TextFormat.TSV, delimiter="\t", **kwargs)
        return self.write_file(data, file_path, config)
    
    def write_json(self, data: Union[pd.DataFrame, List[Dict]], file_path: str, **kwargs) -> WriteResult:
        """Write JSON file."""
        config = WriterConfig(format=TextFormat.JSON, **kwargs)
        return self.write_file(data, file_path, config)
    
    # Specialized Writing Methods
    def write_trading_data(
        self, 
        trades_data: pd.DataFrame, 
        file_path: str,
        include_summary: bool = True,
        config: WriterConfig = None
    ) -> WriteResult:
        """Write trading data with proper formatting."""
        config = config or self.default_config.copy()
        
        # Set appropriate formatting for trading data
        config.decimal_places = 4
        config.date_format = "%Y-%m-%d"
        config.datetime_format = "%Y-%m-%d %H:%M:%S"
        
        try:
            # Format trading data
            formatted_data = self._format_trading_data(trades_data)
            
            # Write main data
            result = self.write_file(formatted_data, file_path, config)
            
            if result.success and include_summary:
                # Write summary file
                summary_path = self._get_summary_path(file_path)
                summary_data = self._create_trading_summary(trades_data)
                summary_result = self.write_file(summary_data, summary_path, config)
                
                if summary_result.success:
                    result.warnings.append(f"Summary written to: {summary_path}")
            
            return result
            
        except Exception as e:
            return WriteResult(
                success=False,
                file_path=file_path,
                error_message=f"Trading data export failed: {str(e)}",
                format_used=config.format
            )
    
    def write_market_data(
        self, 
        market_data: pd.DataFrame, 
        file_path: str,
        config: WriterConfig = None
    ) -> WriteResult:
        """Write market data with OHLCV formatting."""
        config = config or self.default_config.copy()
        
        # Set appropriate formatting for market data
        config.decimal_places = 4
        config.date_format = "%Y-%m-%d"
        
        try:
            # Validate and format market data
            formatted_data = self._format_market_data(market_data)
            return self.write_file(formatted_data, file_path, config)
            
        except Exception as e:
            return WriteResult(
                success=False,
                file_path=file_path,
                error_message=f"Market data export failed: {str(e)}",
                format_used=config.format
            )
    
    # Batch Writing Operations
    def open_file_for_writing(self, file_path: str, config: WriterConfig = None, mode: WriteMode = WriteMode.WRITE) -> bool:
        """Open file for batch writing operations."""
        try:
            config = config or self.default_config
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Open file
            file_handle = open(file_path, mode.value, encoding=config.encoding, buffering=config.buffer_size)
            
            self._open_files[file_path] = file_handle
            self._file_configs[file_path] = config
            
            # Write header if needed
            if config.write_header and mode == WriteMode.WRITE:
                # Header will be written with first data batch
                pass
            
            return True
            
        except Exception:
            return False
    
    def append_data(self, file_path: str, data: Union[List[Any], Dict[str, Any]]) -> bool:
        """Append single row/record to open file."""
        if file_path not in self._open_files:
            return False
        
        try:
            file_handle = self._open_files[file_path]
            config = self._file_configs[file_path]
            
            if config.format == TextFormat.JSON_LINES:
                # JSON Lines format
                if isinstance(data, dict):
                    json.dump(data, file_handle, ensure_ascii=False, default=self._json_serializer)
                    file_handle.write(config.line_terminator)
                else:
                    return False
            else:
                # Delimited format
                if isinstance(data, (list, tuple)):
                    formatted_row = self._format_row_values(data, config)
                    row_text = config.delimiter.join(formatted_row) + config.line_terminator
                    file_handle.write(row_text)
                else:
                    return False
            
            file_handle.flush()
            return True
            
        except Exception:
            return False
    
    def close_file(self, file_path: str) -> bool:
        """Close file opened for batch writing."""
        try:
            if file_path in self._open_files:
                self._open_files[file_path].close()
                del self._open_files[file_path]
                del self._file_configs[file_path]
                return True
            return False
        except Exception:
            return False
    
    def close_all_files(self) -> int:
        """Close all open files."""
        closed_count = 0
        file_paths = list(self._open_files.keys())
        
        for file_path in file_paths:
            if self.close_file(file_path):
                closed_count += 1
        
        return closed_count
    
    # Format-Specific Writing Methods
    def _write_delimited(self, df: pd.DataFrame, file_path: str, config: WriterConfig, mode: WriteMode) -> WriteResult:
        """Write delimited format (CSV, TSV, etc.)."""
        try:
            # Configure delimiter
            delimiter = self._get_delimiter(config.format, config.delimiter)
            
            # Prepare pandas parameters
            pandas_params = {
                'sep': delimiter,
                'encoding': config.encoding,
                'index': False,
                'header': config.write_header,
                'mode': mode.value,
                'lineterminator': config.line_terminator,
                'na_rep': config.na_representation,
                'float_format': f'%.{config.decimal_places}f',
                'date_format': config.datetime_format,
                'quoting': csv.QUOTE_ALL if config.quote_all else csv.QUOTE_MINIMAL,
                'quotechar': config.quote_char
            }
            
            if config.escape_char:
                pandas_params['escapechar'] = config.escape_char
            
            # Write file
            df.to_csv(file_path, **pandas_params)
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            return WriteResult(
                success=True,
                file_path=file_path,
                rows_written=len(df),
                columns_written=len(df.columns),
                bytes_written=file_size,
                format_used=config.format
            )
            
        except Exception as e:
            return WriteResult(
                success=False,
                file_path=file_path,
                error_message=str(e),
                format_used=config.format
            )
    
    def _write_json(self, df: pd.DataFrame, file_path: str, config: WriterConfig, mode: WriteMode) -> WriteResult:
        """Write JSON format."""
        try:
            # Convert DataFrame to JSON
            json_data = df.to_dict(orient='records')
            
            write_mode = mode.value
            if 'b' in write_mode:
                write_mode = write_mode.replace('b', '')
            
            with open(file_path, write_mode, encoding=config.encoding) as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
            
            file_size = os.path.getsize(file_path)
            
            return WriteResult(
                success=True,
                file_path=file_path,
                rows_written=len(df),
                columns_written=len(df.columns),
                bytes_written=file_size,
                format_used=TextFormat.JSON
            )
            
        except Exception as e:
            return WriteResult(
                success=False,
                file_path=file_path,
                error_message=str(e),
                format_used=TextFormat.JSON
            )
    
    def _write_jsonl(self, df: pd.DataFrame, file_path: str, config: WriterConfig, mode: WriteMode) -> WriteResult:
        """Write JSON Lines format."""
        try:
            write_mode = mode.value
            if 'b' in write_mode:
                write_mode = write_mode.replace('b', '')
            
            with open(file_path, write_mode, encoding=config.encoding) as f:
                for _, row in df.iterrows():
                    json.dump(row.to_dict(), f, ensure_ascii=False, default=self._json_serializer)
                    f.write(config.line_terminator)
            
            file_size = os.path.getsize(file_path)
            
            return WriteResult(
                success=True,
                file_path=file_path,
                rows_written=len(df),
                columns_written=len(df.columns),
                bytes_written=file_size,
                format_used=TextFormat.JSON_LINES
            )
            
        except Exception as e:
            return WriteResult(
                success=False,
                file_path=file_path,
                error_message=str(e),
                format_used=TextFormat.JSON_LINES
            )
    
    def _write_fixed_width(self, df: pd.DataFrame, file_path: str, config: WriterConfig, mode: WriteMode) -> WriteResult:
        """Write fixed-width format."""
        try:
            # Calculate column widths
            col_widths = {}
            for col in df.columns:
                max_len = max(
                    len(str(col)),
                    df[col].astype(str).str.len().max()
                )
                col_widths[col] = max_len + 2  # Add padding
            
            write_mode = mode.value
            if 'b' in write_mode:
                write_mode = write_mode.replace('b', '')
            
            with open(file_path, write_mode, encoding=config.encoding) as f:
                # Write header
                if config.write_header:
                    header_line = ""
                    for col in df.columns:
                        header_line += str(col).ljust(col_widths[col])
                    f.write(header_line.rstrip() + config.line_terminator)
                
                # Write data
                for _, row in df.iterrows():
                    data_line = ""
                    for col in df.columns:
                        value = self._format_value(row[col], config)
                        data_line += str(value).ljust(col_widths[col])
                    f.write(data_line.rstrip() + config.line_terminator)
            
            file_size = os.path.getsize(file_path)
            
            return WriteResult(
                success=True,
                file_path=file_path,
                rows_written=len(df),
                columns_written=len(df.columns),
                bytes_written=file_size,
                format_used=TextFormat.FIXED_WIDTH
            )
            
        except Exception as e:
            return WriteResult(
                success=False,
                file_path=file_path,
                error_message=str(e),
                format_used=TextFormat.FIXED_WIDTH
            )
    
    # Data Processing and Formatting
    def _prepare_data(self, data: Union[pd.DataFrame, List, Dict], config: WriterConfig) -> tuple:
        """Prepare data for writing."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            headers = list(df.columns)
        elif isinstance(data, list) and data:
            if isinstance(data[0], dict):
                # List of dictionaries
                df = pd.DataFrame(data)
                headers = list(df.columns)
            elif isinstance(data[0], (list, tuple)):
                # List of lists/tuples
                df = pd.DataFrame(data)
                headers = [f"col_{i}" for i in range(len(df.columns))]
                df.columns = headers
            else:
                # Single column data
                df = pd.DataFrame({'value': data})
                headers = ['value']
        else:
            # Empty or invalid data
            df = pd.DataFrame()
            headers = []
        
        # Apply formatting
        if not df.empty:
            df = self._apply_data_formatting(df, config)
        
        return df, headers
    
    def _apply_data_formatting(self, df: pd.DataFrame, config: WriterConfig) -> pd.DataFrame:
        """Apply formatting rules to DataFrame."""
        formatted_df = df.copy()
        
        for col in formatted_df.columns:
            series = formatted_df[col]
            
            # Format dates and datetimes
            if pd.api.types.is_datetime64_any_dtype(series):
                if series.dt.time.eq(series.dt.time.iloc[0]).all() and series.dt.time.iloc[0] == pd.Timestamp('00:00:00').time():
                    # Date only
                    formatted_df[col] = series.dt.strftime(config.date_format)
                else:
                    # DateTime
                    formatted_df[col] = series.dt.strftime(config.datetime_format)
            
            # Format numeric values
            elif pd.api.types.is_numeric_dtype(series):
                if pd.api.types.is_integer_dtype(series):
                    # Keep integers as integers
                    pass
                else:
                    # Format floats with specified decimal places
                    formatted_df[col] = series.round(config.decimal_places)
        
        return formatted_df
    
    def _format_trading_data(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Format trading data for export."""
        formatted_df = trades_df.copy()
        
        # Standard column order for trading data
        preferred_order = ['date', 'symbol', 'side', 'quantity', 'entry_price', 'exit_price', 'pnl', 'commission']
        
        # Reorder columns if they exist
        existing_cols = [col for col in preferred_order if col in formatted_df.columns]
        other_cols = [col for col in formatted_df.columns if col not in existing_cols]
        formatted_df = formatted_df[existing_cols + other_cols]
        
        return formatted_df
    
    def _format_market_data(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """Format market data for export."""
        formatted_df = market_df.copy()
        
        # Standard column order for market data
        preferred_order = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        # Reorder columns if they exist
        existing_cols = [col for col in preferred_order if col in formatted_df.columns]
        other_cols = [col for col in formatted_df.columns if col not in existing_cols]
        formatted_df = formatted_df[existing_cols + other_cols]
        
        return formatted_df
    
    def _create_trading_summary(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Create trading summary from trades data."""
        try:
            summary_data = []
            
            if 'pnl' in trades_df.columns and not trades_df.empty:
                total_trades = len(trades_df)
                winning_trades = len(trades_df[trades_df['pnl'] > 0])
                losing_trades = len(trades_df[trades_df['pnl'] < 0])
                
                total_pnl = trades_df['pnl'].sum()
                avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
                avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
                
                summary_data = [
                    ['Total Trades', total_trades],
                    ['Winning Trades', winning_trades],
                    ['Losing Trades', losing_trades],
                    ['Win Rate %', (winning_trades / total_trades * 100) if total_trades > 0 else 0],
                    ['Total P&L', round(total_pnl, 2)],
                    ['Average Win', round(avg_win, 2)],
                    ['Average Loss', round(avg_loss, 2)],
                    ['Profit Factor', round((avg_win * winning_trades) / abs(avg_loss * losing_trades), 2) if avg_loss != 0 else 0],
                ]
            
            return pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            
        except Exception:
            return pd.DataFrame([['Error', 'Could not calculate summary']], columns=['Metric', 'Value'])
    
    # Utility Methods
    def _get_delimiter(self, format_type: TextFormat, custom_delimiter: str) -> str:
        """Get delimiter character for format."""
        delimiter_map = {
            TextFormat.CSV: ",",
            TextFormat.TSV: "\t",
            TextFormat.PIPE_DELIMITED: "|",
            TextFormat.SEMICOLON_DELIMITED: ";",
            TextFormat.SPACE_DELIMITED: " ",
            TextFormat.CUSTOM: custom_delimiter
        }
        return delimiter_map.get(format_type, custom_delimiter)
    
    def _format_value(self, value: Any, config: WriterConfig) -> str:
        """Format individual value for writing."""
        if pd.isna(value):
            return config.na_representation
        elif isinstance(value, (date, datetime)):
            if isinstance(value, datetime):
                return value.strftime(config.datetime_format)
            else:
                return value.strftime(config.date_format)
        elif isinstance(value, float):
            return f"{value:.{config.decimal_places}f}"
        else:
            return str(value)
    
    def _format_row_values(self, row_data: List[Any], config: WriterConfig) -> List[str]:
        """Format all values in a row."""
        return [self._format_value(value, config) for value in row_data]
    
    def _json_serializer(self, obj: Any) -> str:
        """Custom JSON serializer for special types."""
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return str(obj)
    
    def _create_backup(self, file_path: str) -> bool:
        """Create backup of existing file."""
        try:
            if os.path.exists(file_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{file_path}.{timestamp}.bak"
                
                import shutil
                shutil.copy2(file_path, backup_path)
                return True
        except Exception:
            pass
        return False
    
    def _get_summary_path(self, file_path: str) -> str:
        """Generate summary file path."""
        path_obj = Path(file_path)
        return str(path_obj.parent / f"{path_obj.stem}_summary{path_obj.suffix}")
    
    # Statistics and Information
    def get_statistics(self) -> Dict[str, Any]:
        """Get writing statistics."""
        return {
            "files_written": self.files_written,
            "total_rows_written": self.total_rows_written,
            "total_bytes_written": self.total_bytes_written,
            "format_usage": dict(self.format_usage),
            "open_files": len(self._open_files),
            "supported_formats": [f.value for f in TextFormat]
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close all files."""
        self.close_all_files()
    
    def __str__(self) -> str:
        """String representation of text file writer."""
        return (
            f"CTxtFileWriter(files_written={self.files_written}, "
            f"rows_written={self.total_rows_written}, "
            f"open_files={len(self._open_files)})"
        )