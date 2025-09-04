"""
Data management module for algorithmic trading system.

This module provides comprehensive data handling functionality including:
- File I/O operations (CFileUtils)
- Excel file integration (CExcelFileHandler)
- Text file reading and writing (CTxtFileReader, CTxtFileWriter)
- INI configuration management (CIniFile)
- Data format conversions and utilities
"""

from .file_utils import (
    CFileUtils,
    FileOperationType,
    FileOperationResult
)

from .excel_handler import (
    CExcelFileHandler,
    ExcelOperationType,
    ExcelSheet,
    ExcelWorkbook
)

from .txt_file_reader import (
    CTxtFileReader,
    TextFileFormat,
    ReadResult
)

from .txt_file_writer import (
    CTxtFileWriter,
    WriteMode,
    WriteResult
)

from .ini_file import (
    CIniFile,
    IniSection,
    IniKeyValue
)

__all__ = [
    # File Operations
    "CFileUtils",
    "FileOperationType",
    "FileOperationResult",
    
    # Excel Operations
    "CExcelFileHandler", 
    "ExcelOperationType",
    "ExcelSheet",
    "ExcelWorkbook",
    
    # Text File Operations
    "CTxtFileReader",
    "TextFileFormat", 
    "ReadResult",
    "CTxtFileWriter",
    "WriteMode",
    "WriteResult",
    
    # INI File Operations
    "CIniFile",
    "IniSection",
    "IniKeyValue"
]