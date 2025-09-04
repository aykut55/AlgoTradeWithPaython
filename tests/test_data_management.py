"""
Comprehensive tests for data management module.
"""

import pytest
import unittest
import os
import tempfile
import shutil
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.file_utils import CFileUtils, FileOperationType, FileOperationResult
from src.data.excel_handler import CExcelFileHandler, ExcelOperationType
from src.data.txt_file_reader import CTxtFileReader, TextFileFormat, ReadResult
from src.data.txt_file_writer import CTxtFileWriter, WriteMode, TextFormat
from src.data.ini_file import CIniFile, IniValueType


class TestCFileUtils(unittest.TestCase):
    """Test cases for CFileUtils class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = Mock()
        self.file_utils = CFileUtils()
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_file.txt")
        
        # Create test file
        with open(self.test_file, 'w') as f:
            f.write("Test content")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertIsInstance(self.file_utils, CFileUtils)
        self.assertEqual(self.file_utils.operations_count, 0)
        self.assertEqual(self.file_utils.default_encoding, "utf-8")
    
    def test_file_exists(self):
        """Test file existence checking."""
        self.assertTrue(self.file_utils.file_exists(self.test_file))
        self.assertFalse(self.file_utils.file_exists("nonexistent_file.txt"))
    
    def test_directory_exists(self):
        """Test directory existence checking."""
        self.assertTrue(self.file_utils.directory_exists(self.temp_dir))
        self.assertFalse(self.file_utils.directory_exists("nonexistent_directory"))
    
    def test_get_file_size(self):
        """Test file size retrieval."""
        size = self.file_utils.get_file_size(self.test_file)
        self.assertGreater(size, 0)
        self.assertEqual(size, len("Test content"))
    
    def test_get_file_info(self):
        """Test comprehensive file information."""
        info = self.file_utils.get_file_info(self.test_file)
        
        self.assertIsNotNone(info)
        self.assertEqual(info.file_name, "test_file.txt")
        self.assertEqual(info.file_extension, ".txt")
        self.assertGreater(info.file_size_bytes, 0)
        self.assertIsInstance(info.creation_time, datetime)
    
    def test_create_directory(self):
        """Test directory creation."""
        new_dir = os.path.join(self.temp_dir, "new_directory")
        result = self.file_utils.create_directory(new_dir)
        
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(new_dir))
        self.assertEqual(result.operation_type, FileOperationType.CREATE)
    
    def test_copy_file(self):
        """Test file copying."""
        dest_file = os.path.join(self.temp_dir, "copied_file.txt")
        result = self.file_utils.copy_file(self.test_file, dest_file)
        
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(dest_file))
        self.assertEqual(result.operation_type, FileOperationType.COPY)
    
    def test_move_file(self):
        """Test file moving."""
        dest_file = os.path.join(self.temp_dir, "moved_file.txt")
        result = self.file_utils.move_file(self.test_file, dest_file)
        
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(dest_file))
        self.assertFalse(os.path.exists(self.test_file))
        self.assertEqual(result.operation_type, FileOperationType.MOVE)
    
    def test_delete_file(self):
        """Test file deletion."""
        result = self.file_utils.delete_file(self.test_file)
        
        self.assertTrue(result.success)
        self.assertFalse(os.path.exists(self.test_file))
        self.assertEqual(result.operation_type, FileOperationType.DELETE)
    
    def test_find_files(self):
        """Test file finding with patterns."""
        # Create multiple test files
        for i in range(3):
            with open(os.path.join(self.temp_dir, f"test_{i}.txt"), 'w') as f:
                f.write(f"Content {i}")
        
        files = self.file_utils.find_files(self.temp_dir, "*.txt")
        self.assertGreaterEqual(len(files), 3)  # At least our test files
        
        # Test recursive search
        sub_dir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(sub_dir)
        with open(os.path.join(sub_dir, "sub_test.txt"), 'w') as f:
            f.write("Sub content")
        
        recursive_files = self.file_utils.find_files(self.temp_dir, "*.txt", recursive=True)
        self.assertGreater(len(recursive_files), len(files))
    
    def test_create_backup(self):
        """Test file backup creation."""
        result = self.file_utils.create_backup(self.test_file)
        
        self.assertTrue(result.success)
        # Should create a backup file with timestamp
        backup_files = self.file_utils.find_files(self.temp_dir, "test_file*")
        self.assertGreater(len(backup_files), 1)  # Original + backup
    
    def test_create_zip_archive(self):
        """Test ZIP archive creation."""
        # Create multiple files
        test_files = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"archive_test_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"Archive content {i}")
            test_files.append(file_path)
        
        archive_path = os.path.join(self.temp_dir, "test_archive.zip")
        result = self.file_utils.create_zip_archive(test_files, archive_path)
        
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(archive_path))
        self.assertEqual(result.operation_type, FileOperationType.ARCHIVE)
    
    def test_normalize_path(self):
        """Test path normalization."""
        test_path = "path/with/../normalized/./components"
        normalized = self.file_utils.normalize_path(test_path)
        self.assertIn("normalized", normalized)
        self.assertNotIn("..", normalized)


class TestCExcelFileHandler(unittest.TestCase):
    """Test cases for CExcelFileHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = Mock()
        self.excel_handler = CExcelFileHandler()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test DataFrame
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=5),
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'Close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertIsInstance(self.excel_handler, CExcelFileHandler)
        self.assertEqual(self.excel_handler.files_processed, 0)
    
    def test_write_excel_file(self):
        """Test Excel file writing."""
        excel_file = os.path.join(self.temp_dir, "test_data.xlsx")
        result = self.excel_handler.write_excel_file(self.test_data, excel_file)
        
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(excel_file))
        self.assertEqual(result.rows_processed, len(self.test_data))
        self.assertEqual(result.columns_processed, len(self.test_data.columns))
    
    def test_read_excel_file(self):
        """Test Excel file reading."""
        # First write a file
        excel_file = os.path.join(self.temp_dir, "test_data.xlsx")
        self.excel_handler.write_excel_file(self.test_data, excel_file)
        
        # Then read it back
        result = self.excel_handler.read_excel_file(excel_file)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.data)
        
        # Should have our sheet data
        if isinstance(result.data, dict):
            df = list(result.data.values())[0]
        else:
            df = result.data
        
        self.assertEqual(len(df), len(self.test_data))
    
    def test_export_trading_data(self):
        """Test trading data export."""
        # Create trading data
        trading_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'symbol': ['EURUSD'] * 10,
            'side': ['BUY'] * 5 + ['SELL'] * 5,
            'quantity': [10000] * 10,
            'entry_price': [1.1000 + i*0.001 for i in range(10)],
            'exit_price': [1.1005 + i*0.001 for i in range(10)],
            'pnl': [50.0] * 7 + [-20.0] * 3  # Some wins and losses
        })
        
        excel_file = os.path.join(self.temp_dir, "trading_data.xlsx")
        result = self.excel_handler.export_trading_data(trading_data, excel_file)
        
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(excel_file))
    
    def test_import_market_data(self):
        """Test market data import with validation."""
        # Create and save market data
        excel_file = os.path.join(self.temp_dir, "market_data.xlsx")
        self.excel_handler.write_excel_file(self.test_data, excel_file)
        
        # Import with validation
        result = self.excel_handler.import_market_data(excel_file)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.data)


class TestCTxtFileReader(unittest.TestCase):
    """Test cases for CTxtFileReader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = Mock()
        self.txt_reader = CTxtFileReader()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV file
        self.csv_file = os.path.join(self.temp_dir, "test_data.csv")
        with open(self.csv_file, 'w') as f:
            f.write("Date,Open,High,Low,Close,Volume\n")
            f.write("2024-01-01,100.0,102.0,99.0,101.0,1000\n")
            f.write("2024-01-02,101.0,103.0,100.0,102.0,1100\n")
            f.write("2024-01-03,102.0,104.0,101.0,103.0,1200\n")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertIsInstance(self.txt_reader, CTxtFileReader)
        self.assertEqual(self.txt_reader.files_read, 0)
    
    def test_read_csv_file(self):
        """Test CSV file reading."""
        result = self.txt_reader.read_csv(self.csv_file)
        
        self.assertTrue(result.success)
        self.assertEqual(result.format_detected, TextFileFormat.CSV)
        self.assertEqual(result.rows_read, 3)
        self.assertEqual(result.columns_detected, 6)
        self.assertIn('Date', result.headers)
        self.assertIn('Close', result.headers)
    
    def test_read_market_data(self):
        """Test market data reading with validation."""
        result = self.txt_reader.read_market_data(self.csv_file)
        
        self.assertTrue(result.success)
        self.assertIsInstance(result.data, pd.DataFrame)
        
        # Check if market data columns were detected
        df = result.data
        expected_columns = ['date', 'open', 'high', 'low', 'close']
        has_market_columns = any(col.lower() in df.columns.str.lower() for col in expected_columns)
        self.assertTrue(has_market_columns)
    
    def test_format_detection(self):
        """Test automatic format detection."""
        # Create TSV file
        tsv_file = os.path.join(self.temp_dir, "test_data.tsv")
        with open(tsv_file, 'w') as f:
            f.write("Date\tOpen\tHigh\tLow\tClose\n")
            f.write("2024-01-01\t100.0\t102.0\t99.0\t101.0\n")
        
        result = self.txt_reader.read_file(tsv_file)
        
        self.assertTrue(result.success)
        # Format detection should identify TSV
        self.assertEqual(result.format_detected, TextFileFormat.TSV)
    
    def test_get_file_preview(self):
        """Test file preview functionality."""
        result = self.txt_reader.get_file_preview(self.csv_file, num_rows=2)
        
        self.assertTrue(result.success)
        self.assertEqual(result.rows_read, 2)  # Should only read 2 rows
    
    def test_count_lines(self):
        """Test line counting."""
        line_count = self.txt_reader.count_lines(self.csv_file)
        self.assertEqual(line_count, 4)  # Header + 3 data rows


class TestCTxtFileWriter(unittest.TestCase):
    """Test cases for CTxtFileWriter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = Mock()
        self.txt_writer = CTxtFileWriter()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test DataFrame
        self.test_data = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Value': [100.0, 200.0, 300.0],
            'Count': [1, 2, 3]
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertIsInstance(self.txt_writer, CTxtFileWriter)
        self.assertEqual(self.txt_writer.files_written, 0)
    
    def test_write_csv_file(self):
        """Test CSV file writing."""
        csv_file = os.path.join(self.temp_dir, "output.csv")
        result = self.txt_writer.write_csv(self.test_data, csv_file)
        
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(csv_file))
        self.assertEqual(result.rows_written, len(self.test_data))
        self.assertEqual(result.columns_written, len(self.test_data.columns))
    
    def test_write_json_file(self):
        """Test JSON file writing."""
        json_file = os.path.join(self.temp_dir, "output.json")
        result = self.txt_writer.write_json(self.test_data, json_file)
        
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(json_file))
        
        # Verify JSON content
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), len(self.test_data))
    
    def test_write_trading_data(self):
        """Test trading data writing with formatting."""
        trading_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'symbol': ['EURUSD', 'GBPUSD'],
            'side': ['BUY', 'SELL'],
            'entry_price': [1.1000, 1.2000],
            'exit_price': [1.1050, 1.1950],
            'pnl': [50.0, -50.0]
        })
        
        output_file = os.path.join(self.temp_dir, "trades.csv")
        result = self.txt_writer.write_trading_data(trading_data, output_file)
        
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(output_file))
    
    def test_batch_writing(self):
        """Test batch writing operations."""
        batch_file = os.path.join(self.temp_dir, "batch_output.csv")
        
        # Open file for batch writing
        success = self.txt_writer.open_file_for_writing(batch_file)
        self.assertTrue(success)
        
        # Append rows
        for i in range(5):
            row_data = [f"2024-01-{i+1:02d}", 100.0 + i, i + 1]
            success = self.txt_writer.append_data(batch_file, row_data)
            self.assertTrue(success)
        
        # Close file
        success = self.txt_writer.close_file(batch_file)
        self.assertTrue(success)
        
        # Verify file exists and has content
        self.assertTrue(os.path.exists(batch_file))


class TestCIniFile(unittest.TestCase):
    """Test cases for CIniFile class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = Mock()
        self.ini_file = CIniFile()
        self.temp_dir = tempfile.mkdtemp()
        self.ini_path = os.path.join(self.temp_dir, "test_config.ini")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertIsInstance(self.ini_file, CIniFile)
        self.assertEqual(self.ini_file.files_processed, 0)
    
    def test_create_trading_system_template(self):
        """Test trading system template creation."""
        success = self.ini_file.create_trading_system_template()
        self.assertTrue(success)
        
        # Check if sections were created
        self.assertTrue(self.ini_file.has_section("Trading"))
        self.assertTrue(self.ini_file.has_section("Database"))
        self.assertTrue(self.ini_file.has_section("RiskManagement"))
        
        # Check specific values
        initial_balance = self.ini_file.get_float("Trading", "initial_balance")
        self.assertEqual(initial_balance, 100000.0)
    
    def test_set_and_get_values(self):
        """Test setting and getting values."""
        # Test different data types
        self.ini_file.set_value("TestSection", "string_key", "test_value", IniValueType.STRING)
        self.ini_file.set_value("TestSection", "int_key", 42, IniValueType.INTEGER)
        self.ini_file.set_value("TestSection", "float_key", 3.14, IniValueType.FLOAT)
        self.ini_file.set_value("TestSection", "bool_key", True, IniValueType.BOOLEAN)
        self.ini_file.set_value("TestSection", "list_key", ["a", "b", "c"], IniValueType.LIST)
        
        # Test retrieval
        self.assertEqual(self.ini_file.get_string("TestSection", "string_key"), "test_value")
        self.assertEqual(self.ini_file.get_int("TestSection", "int_key"), 42)
        self.assertAlmostEqual(self.ini_file.get_float("TestSection", "float_key"), 3.14)
        self.assertEqual(self.ini_file.get_bool("TestSection", "bool_key"), True)
        self.assertEqual(self.ini_file.get_list("TestSection", "list_key"), ["a", "b", "c"])
    
    def test_save_and_load_file(self):
        """Test file saving and loading."""
        # Create configuration
        self.ini_file.set_value("Config", "app_name", "Trading System")
        self.ini_file.set_value("Config", "version", "1.0")
        
        # Save to file
        result = self.ini_file.save_file(self.ini_path)
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(self.ini_path))
        
        # Create new instance and load
        new_ini = CIniFile()
        result = new_ini.load_file(self.ini_path)
        self.assertTrue(result.success)
        
        # Verify values
        self.assertEqual(new_ini.get_string("Config", "app_name"), "Trading System")
        self.assertEqual(new_ini.get_string("Config", "version"), "1.0")
    
    def test_section_management(self):
        """Test section management operations."""
        # Add section
        success = self.ini_file.add_section("NewSection", "Test description")
        self.assertTrue(success)
        self.assertTrue(self.ini_file.has_section("NewSection"))
        
        # Remove section
        success = self.ini_file.remove_section("NewSection")
        self.assertTrue(success)
        self.assertFalse(self.ini_file.has_section("NewSection"))
    
    def test_validation(self):
        """Test configuration validation."""
        # Create configuration with required values
        self.ini_file.add_section("Required", "Required section")
        self.ini_file.set_value("Required", "mandatory_key", "value")
        
        # Mark as required (would need to modify internal structure for real test)
        is_valid, errors = self.ini_file.validate_configuration()
        
        # Should pass basic validation
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_dict_conversion(self):
        """Test conversion to/from dictionary."""
        # Set some values
        self.ini_file.set_value("Section1", "key1", "value1")
        self.ini_file.set_value("Section1", "key2", 42)
        self.ini_file.set_value("Section2", "key3", True)
        
        # Convert to dict
        config_dict = self.ini_file.to_dict()
        
        self.assertIn("Section1", config_dict)
        self.assertIn("Section2", config_dict)
        self.assertEqual(config_dict["Section1"]["key1"], "value1")
        self.assertEqual(config_dict["Section1"]["key2"], 42)
        self.assertEqual(config_dict["Section2"]["key3"], True)
    
    def test_json_export_import(self):
        """Test JSON export and import."""
        # Create configuration
        self.ini_file.set_value("App", "name", "Test App")
        self.ini_file.set_value("App", "debug", True)
        
        # Export to JSON
        json_path = os.path.join(self.temp_dir, "config.json")
        success = self.ini_file.export_to_json(json_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(json_path))
        
        # Import from JSON
        new_ini = CIniFile()
        success = new_ini.import_from_json(json_path)
        self.assertTrue(success)
        
        # Verify values
        self.assertEqual(new_ini.get_string("App", "name"), "Test App")
        self.assertEqual(new_ini.get_bool("App", "debug"), True)


if __name__ == '__main__':
    unittest.main()