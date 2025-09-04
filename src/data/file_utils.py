"""
Comprehensive file utilities for trading system.

This module provides advanced file operations including:
- File and directory management
- Path manipulation and validation
- File system monitoring
- Backup and archival operations
- Cross-platform compatibility
"""

import os
import sys
import shutil
import glob
import stat
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import tempfile
import zipfile

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.base import CBase, SystemProtocol


class FileOperationType(Enum):
    """Types of file operations."""
    CREATE = "create"
    READ = "read"
    WRITE = "write"
    APPEND = "append"
    DELETE = "delete"
    COPY = "copy"
    MOVE = "move"
    RENAME = "rename"
    BACKUP = "backup"
    ARCHIVE = "archive"


class FileAccessMode(Enum):
    """File access modes."""
    READ_ONLY = "r"
    WRITE_ONLY = "w"
    APPEND_ONLY = "a"
    READ_WRITE = "r+"
    WRITE_READ = "w+"
    APPEND_READ = "a+"
    BINARY_READ = "rb"
    BINARY_WRITE = "wb"
    BINARY_APPEND = "ab"


@dataclass
class FileOperationResult:
    """Result of a file operation."""
    
    operation_type: FileOperationType
    success: bool
    file_path: str = ""
    error_message: str = ""
    bytes_processed: int = 0
    execution_time_ms: float = 0.0
    additional_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}


@dataclass
class FileInfo:
    """Comprehensive file information."""
    
    file_path: str
    file_name: str
    file_extension: str
    file_size_bytes: int
    creation_time: datetime
    modification_time: datetime
    access_time: datetime
    is_directory: bool = False
    is_hidden: bool = False
    is_readonly: bool = False
    permissions: str = ""
    file_hash: str = ""


@dataclass
class DirectoryInfo:
    """Directory information and statistics."""
    
    directory_path: str
    total_files: int = 0
    total_directories: int = 0
    total_size_bytes: int = 0
    file_count_by_extension: Dict[str, int] = None
    largest_file: str = ""
    newest_file: str = ""
    oldest_file: str = ""
    
    def __post_init__(self):
        if self.file_count_by_extension is None:
            self.file_count_by_extension = {}


class CFileUtils(CBase):
    """
    Comprehensive file utilities for trading systems.
    
    Provides advanced file and directory operations including:
    - File creation, reading, writing, and deletion
    - Directory management and analysis
    - File system monitoring and backup operations
    - Cross-platform path handling
    - File integrity verification
    """
    
    def __init__(self, system_id: int = 0):
        super().__init__(system_id)
        
        # Configuration
        self.default_encoding = "utf-8"
        self.buffer_size = 8192  # 8KB buffer for file operations
        self.backup_extension = ".bak"
        self.temp_directory = tempfile.gettempdir()
        
        # Statistics
        self.operations_count = 0
        self.total_bytes_processed = 0
        self.last_operation_time = 0.0
        
        # Path separators for cross-platform compatibility
        self.path_separator = os.path.sep
        self.path_separator_alt = "/" if os.path.sep == "\\" else "\\"
        
    def initialize(
        self, 
        system: SystemProtocol,
        default_encoding: str = "utf-8"
    ) -> 'CFileUtils':
        """Initialize file utils with system."""
        self.default_encoding = default_encoding
        return self
    
    # Basic File Operations
    def file_exists(self, file_path: str) -> bool:
        """Check if file exists."""
        try:
            return os.path.exists(file_path) and os.path.isfile(file_path)
        except (OSError, ValueError):
            return False
    
    def directory_exists(self, directory_path: str) -> bool:
        """Check if directory exists."""
        try:
            return os.path.exists(directory_path) and os.path.isdir(directory_path)
        except (OSError, ValueError):
            return False
    
    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        try:
            return os.path.getsize(file_path) if self.file_exists(file_path) else 0
        except (OSError, ValueError):
            return 0
    
    def get_file_info(self, file_path: str) -> Optional[FileInfo]:
        """Get comprehensive file information."""
        if not self.file_exists(file_path):
            return None
        
        try:
            path_obj = Path(file_path)
            stat_info = path_obj.stat()
            
            # Calculate file hash for integrity
            file_hash = self._calculate_file_hash(file_path)
            
            return FileInfo(
                file_path=str(path_obj.absolute()),
                file_name=path_obj.name,
                file_extension=path_obj.suffix.lower(),
                file_size_bytes=stat_info.st_size,
                creation_time=datetime.fromtimestamp(stat_info.st_ctime),
                modification_time=datetime.fromtimestamp(stat_info.st_mtime),
                access_time=datetime.fromtimestamp(stat_info.st_atime),
                is_directory=path_obj.is_dir(),
                is_hidden=self._is_hidden_file(file_path),
                is_readonly=not (stat_info.st_mode & stat.S_IWRITE),
                permissions=oct(stat_info.st_mode)[-3:],
                file_hash=file_hash
            )
        except (OSError, ValueError) as e:
            return None
    
    def create_directory(self, directory_path: str, create_parents: bool = True) -> FileOperationResult:
        """Create directory with optional parent creation."""
        start_time = datetime.now()
        
        try:
            if self.directory_exists(directory_path):
                return FileOperationResult(
                    operation_type=FileOperationType.CREATE,
                    success=True,
                    file_path=directory_path,
                    additional_info={"already_exists": True}
                )
            
            if create_parents:
                os.makedirs(directory_path, exist_ok=True)
            else:
                os.mkdir(directory_path)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.operations_count += 1
            self.last_operation_time = execution_time
            
            return FileOperationResult(
                operation_type=FileOperationType.CREATE,
                success=True,
                file_path=directory_path,
                execution_time_ms=execution_time
            )
            
        except (OSError, ValueError) as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return FileOperationResult(
                operation_type=FileOperationType.CREATE,
                success=False,
                file_path=directory_path,
                error_message=str(e),
                execution_time_ms=execution_time
            )
    
    def delete_file(self, file_path: str, create_backup: bool = False) -> FileOperationResult:
        """Delete file with optional backup creation."""
        start_time = datetime.now()
        
        if not self.file_exists(file_path):
            return FileOperationResult(
                operation_type=FileOperationType.DELETE,
                success=False,
                file_path=file_path,
                error_message="File does not exist"
            )
        
        try:
            # Create backup if requested
            backup_path = ""
            if create_backup:
                backup_result = self.create_backup(file_path)
                if backup_result.success:
                    backup_path = backup_result.additional_info.get("backup_path", "")
            
            # Get file size before deletion
            file_size = self.get_file_size(file_path)
            
            # Delete the file
            os.remove(file_path)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.operations_count += 1
            self.last_operation_time = execution_time
            
            return FileOperationResult(
                operation_type=FileOperationType.DELETE,
                success=True,
                file_path=file_path,
                bytes_processed=file_size,
                execution_time_ms=execution_time,
                additional_info={"backup_created": create_backup, "backup_path": backup_path}
            )
            
        except (OSError, ValueError) as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return FileOperationResult(
                operation_type=FileOperationType.DELETE,
                success=False,
                file_path=file_path,
                error_message=str(e),
                execution_time_ms=execution_time
            )
    
    def copy_file(self, source_path: str, destination_path: str, overwrite: bool = False) -> FileOperationResult:
        """Copy file from source to destination."""
        start_time = datetime.now()
        
        if not self.file_exists(source_path):
            return FileOperationResult(
                operation_type=FileOperationType.COPY,
                success=False,
                file_path=source_path,
                error_message="Source file does not exist"
            )
        
        if self.file_exists(destination_path) and not overwrite:
            return FileOperationResult(
                operation_type=FileOperationType.COPY,
                success=False,
                file_path=destination_path,
                error_message="Destination file exists and overwrite is False"
            )
        
        try:
            # Ensure destination directory exists
            dest_dir = os.path.dirname(destination_path)
            if dest_dir and not self.directory_exists(dest_dir):
                self.create_directory(dest_dir)
            
            # Copy the file
            shutil.copy2(source_path, destination_path)  # copy2 preserves metadata
            
            file_size = self.get_file_size(destination_path)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            self.operations_count += 1
            self.total_bytes_processed += file_size
            self.last_operation_time = execution_time
            
            return FileOperationResult(
                operation_type=FileOperationType.COPY,
                success=True,
                file_path=destination_path,
                bytes_processed=file_size,
                execution_time_ms=execution_time,
                additional_info={"source_path": source_path}
            )
            
        except (OSError, ValueError, shutil.Error) as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return FileOperationResult(
                operation_type=FileOperationType.COPY,
                success=False,
                file_path=destination_path,
                error_message=str(e),
                execution_time_ms=execution_time
            )
    
    def move_file(self, source_path: str, destination_path: str, overwrite: bool = False) -> FileOperationResult:
        """Move file from source to destination."""
        start_time = datetime.now()
        
        if not self.file_exists(source_path):
            return FileOperationResult(
                operation_type=FileOperationType.MOVE,
                success=False,
                file_path=source_path,
                error_message="Source file does not exist"
            )
        
        if self.file_exists(destination_path) and not overwrite:
            return FileOperationResult(
                operation_type=FileOperationType.MOVE,
                success=False,
                file_path=destination_path,
                error_message="Destination file exists and overwrite is False"
            )
        
        try:
            # Ensure destination directory exists
            dest_dir = os.path.dirname(destination_path)
            if dest_dir and not self.directory_exists(dest_dir):
                self.create_directory(dest_dir)
            
            file_size = self.get_file_size(source_path)
            
            # Move the file
            shutil.move(source_path, destination_path)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            self.operations_count += 1
            self.total_bytes_processed += file_size
            self.last_operation_time = execution_time
            
            return FileOperationResult(
                operation_type=FileOperationType.MOVE,
                success=True,
                file_path=destination_path,
                bytes_processed=file_size,
                execution_time_ms=execution_time,
                additional_info={"source_path": source_path}
            )
            
        except (OSError, ValueError, shutil.Error) as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return FileOperationResult(
                operation_type=FileOperationType.MOVE,
                success=False,
                file_path=destination_path,
                error_message=str(e),
                execution_time_ms=execution_time
            )
    
    # Advanced Operations
    def create_backup(self, file_path: str, backup_suffix: str = None) -> FileOperationResult:
        """Create backup of existing file."""
        if not self.file_exists(file_path):
            return FileOperationResult(
                operation_type=FileOperationType.BACKUP,
                success=False,
                file_path=file_path,
                error_message="Source file does not exist"
            )
        
        try:
            if backup_suffix is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_suffix = f"_{timestamp}{self.backup_extension}"
            
            path_obj = Path(file_path)
            backup_path = str(path_obj.parent / f"{path_obj.stem}{backup_suffix}{path_obj.suffix}")
            
            return self.copy_file(file_path, backup_path, overwrite=True)
            
        except (OSError, ValueError) as e:
            return FileOperationResult(
                operation_type=FileOperationType.BACKUP,
                success=False,
                file_path=file_path,
                error_message=str(e)
            )
    
    def find_files(
        self, 
        directory_path: str, 
        pattern: str = "*",
        recursive: bool = True,
        include_directories: bool = False
    ) -> List[str]:
        """Find files matching pattern in directory."""
        try:
            if not self.directory_exists(directory_path):
                return []
            
            if recursive:
                # Use glob for recursive search
                search_pattern = os.path.join(directory_path, "**", pattern)
                files = glob.glob(search_pattern, recursive=True)
            else:
                search_pattern = os.path.join(directory_path, pattern)
                files = glob.glob(search_pattern)
            
            # Filter based on include_directories flag
            if not include_directories:
                files = [f for f in files if os.path.isfile(f)]
            
            return sorted(files)
            
        except (OSError, ValueError):
            return []
    
    def get_directory_info(self, directory_path: str, recursive: bool = True) -> Optional[DirectoryInfo]:
        """Get comprehensive directory information and statistics."""
        if not self.directory_exists(directory_path):
            return None
        
        try:
            info = DirectoryInfo(directory_path=directory_path)
            
            if recursive:
                for root, dirs, files in os.walk(directory_path):
                    info.total_directories += len(dirs)
                    info.total_files += len(files)
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_size = self.get_file_size(file_path)
                        info.total_size_bytes += file_size
                        
                        # Track file extensions
                        ext = os.path.splitext(file)[1].lower()
                        info.file_count_by_extension[ext] = info.file_count_by_extension.get(ext, 0) + 1
                        
                        # Track largest file
                        if not info.largest_file or file_size > self.get_file_size(info.largest_file):
                            info.largest_file = file_path
            else:
                # Non-recursive - just immediate contents
                items = os.listdir(directory_path)
                for item in items:
                    item_path = os.path.join(directory_path, item)
                    if os.path.isdir(item_path):
                        info.total_directories += 1
                    else:
                        info.total_files += 1
                        file_size = self.get_file_size(item_path)
                        info.total_size_bytes += file_size
                        
                        ext = os.path.splitext(item)[1].lower()
                        info.file_count_by_extension[ext] = info.file_count_by_extension.get(ext, 0) + 1
            
            return info
            
        except (OSError, ValueError):
            return None
    
    def clean_directory(
        self, 
        directory_path: str, 
        older_than_days: int = 30,
        pattern: str = "*",
        create_backup: bool = False
    ) -> List[FileOperationResult]:
        """Clean old files from directory."""
        results = []
        
        if not self.directory_exists(directory_path):
            results.append(FileOperationResult(
                operation_type=FileOperationType.DELETE,
                success=False,
                file_path=directory_path,
                error_message="Directory does not exist"
            ))
            return results
        
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        files = self.find_files(directory_path, pattern, recursive=True)
        
        for file_path in files:
            file_info = self.get_file_info(file_path)
            if file_info and file_info.modification_time < cutoff_date:
                result = self.delete_file(file_path, create_backup=create_backup)
                results.append(result)
        
        return results
    
    # Archive Operations
    def create_zip_archive(self, file_paths: List[str], archive_path: str) -> FileOperationResult:
        """Create ZIP archive from list of files."""
        start_time = datetime.now()
        
        try:
            total_bytes = 0
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in file_paths:
                    if self.file_exists(file_path):
                        # Add file to archive with relative path
                        arcname = os.path.basename(file_path)
                        zipf.write(file_path, arcname)
                        total_bytes += self.get_file_size(file_path)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            self.operations_count += 1
            self.total_bytes_processed += total_bytes
            self.last_operation_time = execution_time
            
            return FileOperationResult(
                operation_type=FileOperationType.ARCHIVE,
                success=True,
                file_path=archive_path,
                bytes_processed=total_bytes,
                execution_time_ms=execution_time,
                additional_info={"files_archived": len(file_paths)}
            )
            
        except (OSError, zipfile.BadZipFile) as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return FileOperationResult(
                operation_type=FileOperationType.ARCHIVE,
                success=False,
                file_path=archive_path,
                error_message=str(e),
                execution_time_ms=execution_time
            )
    
    def extract_zip_archive(self, archive_path: str, extract_to: str = None) -> FileOperationResult:
        """Extract ZIP archive to specified directory."""
        start_time = datetime.now()
        
        if not self.file_exists(archive_path):
            return FileOperationResult(
                operation_type=FileOperationType.READ,
                success=False,
                file_path=archive_path,
                error_message="Archive file does not exist"
            )
        
        try:
            if extract_to is None:
                extract_to = os.path.splitext(archive_path)[0]
            
            # Ensure extraction directory exists
            if not self.directory_exists(extract_to):
                self.create_directory(extract_to)
            
            files_extracted = 0
            total_bytes = 0
            
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(extract_to)
                files_extracted = len(zipf.namelist())
                
                # Calculate total extracted size
                for info in zipf.infolist():
                    total_bytes += info.file_size
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            self.operations_count += 1
            self.total_bytes_processed += total_bytes
            self.last_operation_time = execution_time
            
            return FileOperationResult(
                operation_type=FileOperationType.READ,
                success=True,
                file_path=extract_to,
                bytes_processed=total_bytes,
                execution_time_ms=execution_time,
                additional_info={"files_extracted": files_extracted}
            )
            
        except (OSError, zipfile.BadZipFile) as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return FileOperationResult(
                operation_type=FileOperationType.READ,
                success=False,
                file_path=archive_path,
                error_message=str(e),
                execution_time_ms=execution_time
            )
    
    # Path Utilities
    def normalize_path(self, path: str) -> str:
        """Normalize path for cross-platform compatibility."""
        try:
            return os.path.normpath(os.path.expanduser(path))
        except (OSError, ValueError):
            return path
    
    def get_relative_path(self, file_path: str, base_path: str = None) -> str:
        """Get relative path from base path."""
        try:
            if base_path is None:
                base_path = os.getcwd()
            
            return os.path.relpath(file_path, base_path)
        except (OSError, ValueError):
            return file_path
    
    def join_paths(self, *paths: str) -> str:
        """Join multiple path components."""
        try:
            return os.path.join(*paths)
        except (OSError, ValueError):
            return ""
    
    # Utility Methods
    def _calculate_file_hash(self, file_path: str, algorithm: str = "md5") -> str:
        """Calculate hash of file for integrity checking."""
        try:
            hash_func = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                while chunk := f.read(self.buffer_size):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
            
        except (OSError, ValueError):
            return ""
    
    def _is_hidden_file(self, file_path: str) -> bool:
        """Check if file is hidden (cross-platform)."""
        try:
            if os.name == 'nt':  # Windows
                import ctypes
                attrs = ctypes.windll.kernel32.GetFileAttributesW(file_path)
                return attrs != -1 and (attrs & 2) != 0
            else:  # Unix/Linux/Mac
                return os.path.basename(file_path).startswith('.')
        except:
            return False
    
    # Statistics and Information
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get file operation statistics."""
        return {
            "operations_count": self.operations_count,
            "total_bytes_processed": self.total_bytes_processed,
            "last_operation_time_ms": self.last_operation_time,
            "average_operation_time_ms": self.last_operation_time,  # Could track this better
            "temp_directory": self.temp_directory,
            "default_encoding": self.default_encoding
        }
    
    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        try:
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.1f} PB"
        except (ValueError, OverflowError):
            return "0 B"
    
    def __str__(self) -> str:
        """String representation of file utils."""
        return (
            f"CFileUtils(operations={self.operations_count}, "
            f"bytes_processed={self.format_file_size(self.total_bytes_processed)}, "
            f"encoding={self.default_encoding})"
        )