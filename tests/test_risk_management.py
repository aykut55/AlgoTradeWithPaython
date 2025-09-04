"""
Comprehensive tests for risk management module.
"""

import pytest
import unittest
from datetime import datetime
from unittest.mock import Mock, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.risk_management.take_profit_stop_loss import (
    CKarAlZararKes, 
    TPSLType, 
    TPSLResult, 
    TPSLConfiguration
)


class TestCKarAlZararKes(unittest.TestCase):
    """Test cases for CKarAlZararKes (Take Profit/Stop Loss) class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = Mock()
        self.system.system_id = 1
        
        # Mock market data
        self.system.market_data = Mock()
        self.system.market_data.closes = [100, 105, 102, 108, 104, 110, 107]
        self.system.market_data.highs = [102, 107, 104, 110, 106, 112, 109]
        self.system.market_data.lows = [98, 103, 100, 106, 102, 108, 105]
        
        # Mock position
        self.system.position = Mock()
        self.system.position.size = 0
        self.system.position.entry_price = 100.0
        
        self.tpsl = CKarAlZararKes()
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertIsInstance(self.tpsl, CKarAlZararKes)
        self.assertEqual(self.tpsl.system_id, 0)
        self.assertFalse(self.tpsl.enabled)
    
    def test_enable_disable(self):
        """Test enable/disable functionality."""
        self.tpsl.enable()
        self.assertTrue(self.tpsl.enabled)
        
        self.tpsl.disable()
        self.assertFalse(self.tpsl.enabled)
    
    def test_configuration_setup(self):
        """Test configuration setup."""
        config = TPSLConfiguration(
            take_profit_percentage=5.0,
            stop_loss_percentage=3.0,
            trailing_stop_enabled=True
        )
        
        self.tpsl.configure(config)
        self.assertEqual(self.tpsl.configuration.take_profit_percentage, 5.0)
        self.assertEqual(self.tpsl.configuration.stop_loss_percentage, 3.0)
        self.assertTrue(self.tpsl.configuration.trailing_stop_enabled)
    
    def test_percentage_based_take_profit_long(self):
        """Test percentage-based take profit for long positions."""
        self.system.position.size = 1  # Long position
        self.system.position.entry_price = 100.0
        
        config = TPSLConfiguration(
            take_profit_percentage=5.0,
            tpsl_type=TPSLType.PERCENTAGE_BASED
        )
        self.tpsl.configure(config)
        self.tpsl.enable()
        
        # Test TP calculation
        tp_price = self.tpsl.calculate_take_profit_percentage(self.system, 100.0, 5.0)
        self.assertEqual(tp_price, 105.0)
        
        # Test SL calculation  
        sl_price = self.tpsl.calculate_stop_loss_percentage(self.system, 100.0, 3.0)
        self.assertEqual(sl_price, 97.0)
    
    def test_percentage_based_take_profit_short(self):
        """Test percentage-based take profit for short positions."""
        self.system.position.size = -1  # Short position
        self.system.position.entry_price = 100.0
        
        config = TPSLConfiguration(
            take_profit_percentage=5.0,
            tpsl_type=TPSLType.PERCENTAGE_BASED
        )
        self.tpsl.configure(config)
        
        # Test TP calculation for short
        tp_price = self.tpsl.calculate_take_profit_percentage(self.system, 100.0, 5.0)
        self.assertEqual(tp_price, 95.0)
        
        # Test SL calculation for short
        sl_price = self.tpsl.calculate_stop_loss_percentage(self.system, 100.0, 3.0)
        self.assertEqual(sl_price, 103.0)
    
    def test_trailing_stop_long(self):
        """Test trailing stop for long positions."""
        self.system.position.size = 1
        self.system.position.entry_price = 100.0
        
        config = TPSLConfiguration(
            trailing_stop_percentage=2.0,
            tpsl_type=TPSLType.TRAILING_STOP,
            trailing_stop_enabled=True
        )
        self.tpsl.configure(config)
        self.tpsl.enable()
        
        # Price moves up, trailing stop should follow
        result = self.tpsl.calculate_trailing_stop_percentage(self.system, 110.0, 2.0)
        self.assertEqual(result, 107.8)  # 110 * (1 - 0.02)
    
    def test_multi_level_take_profit(self):
        """Test multi-level take profit system."""
        config = TPSLConfiguration(
            tpsl_type=TPSLType.MULTI_LEVEL,
            take_profit_levels=[2.0, 5.0, 10.0],
            take_profit_percentages=[25.0, 50.0, 25.0]  # Exit percentages
        )
        self.tpsl.configure(config)
        
        tp_levels = self.tpsl.calculate_take_profit_multi_level(self.system, 100.0)
        expected_levels = [102.0, 105.0, 110.0]
        
        self.assertEqual(len(tp_levels), 3)
        self.assertEqual(tp_levels, expected_levels)
    
    def test_should_take_profit_trigger(self):
        """Test take profit triggering logic."""
        self.system.position.size = 1
        self.system.position.entry_price = 100.0
        
        config = TPSLConfiguration(
            take_profit_percentage=5.0,
            tpsl_type=TPSLType.PERCENTAGE_BASED
        )
        self.tpsl.configure(config)
        self.tpsl.enable()
        
        # Current price above TP level should trigger
        result = self.tpsl.should_take_profit(self.system, 106.0)
        self.assertTrue(result.should_trigger)
        self.assertEqual(result.trigger_type, TPSLType.PERCENTAGE_BASED)
    
    def test_should_stop_loss_trigger(self):
        """Test stop loss triggering logic."""
        self.system.position.size = 1
        self.system.position.entry_price = 100.0
        
        config = TPSLConfiguration(
            stop_loss_percentage=3.0,
            tpsl_type=TPSLType.PERCENTAGE_BASED
        )
        self.tpsl.configure(config)
        self.tpsl.enable()
        
        # Current price below SL level should trigger
        result = self.tpsl.should_stop_loss(self.system, 96.0)
        self.assertTrue(result.should_trigger)
        self.assertEqual(result.trigger_type, TPSLType.PERCENTAGE_BASED)
    
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        self.tpsl.enable()
        
        # Simulate some TP/SL triggers
        self.tpsl.statistics.take_profit_triggers = 5
        self.tpsl.statistics.stop_loss_triggers = 3
        self.tpsl.statistics.total_profit_from_tp = 150.0
        self.tpsl.statistics.total_loss_from_sl = -90.0
        
        stats = self.tpsl.get_statistics()
        
        self.assertEqual(stats["take_profit_triggers"], 5)
        self.assertEqual(stats["stop_loss_triggers"], 3)
        self.assertEqual(stats["total_profit_from_tp"], 150.0)
        self.assertEqual(stats["total_loss_from_sl"], -90.0)
        self.assertEqual(stats["net_result"], 60.0)
    
    def test_reset_functionality(self):
        """Test reset functionality."""
        self.tpsl.statistics.take_profit_triggers = 5
        self.tpsl.statistics.total_profit_from_tp = 100.0
        
        self.tpsl.reset(self.system)
        
        self.assertEqual(self.tpsl.statistics.take_profit_triggers, 0)
        self.assertEqual(self.tpsl.statistics.total_profit_from_tp, 0.0)
    
    def test_absolute_levels(self):
        """Test absolute TP/SL levels."""
        config = TPSLConfiguration(
            tpsl_type=TPSLType.ABSOLUTE_LEVELS,
            take_profit_absolute=105.0,
            stop_loss_absolute=95.0
        )
        self.tpsl.configure(config)
        
        tp_result = self.tpsl.calculate_take_profit_absolute(self.system, 105.0)
        sl_result = self.tpsl.calculate_stop_loss_absolute(self.system, 95.0)
        
        self.assertEqual(tp_result, 105.0)
        self.assertEqual(sl_result, 95.0)
    
    def test_disabled_tpsl_no_triggers(self):
        """Test that disabled TP/SL doesn't trigger."""
        self.tpsl.disable()
        
        result = self.tpsl.should_take_profit(self.system, 150.0)
        self.assertFalse(result.should_trigger)
        
        result = self.tpsl.should_stop_loss(self.system, 50.0)
        self.assertFalse(result.should_trigger)


class TestTPSLConfiguration(unittest.TestCase):
    """Test TP/SL configuration class."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = TPSLConfiguration()
        
        self.assertEqual(config.take_profit_percentage, 0.0)
        self.assertEqual(config.stop_loss_percentage, 0.0)
        self.assertEqual(config.tpsl_type, TPSLType.PERCENTAGE_BASED)
        self.assertFalse(config.trailing_stop_enabled)
    
    def test_validation(self):
        """Test configuration validation."""
        config = TPSLConfiguration(
            take_profit_percentage=5.0,
            stop_loss_percentage=3.0
        )
        
        self.assertTrue(config.is_valid())
        
        # Invalid configuration (negative percentages)
        invalid_config = TPSLConfiguration(
            take_profit_percentage=-5.0
        )
        
        self.assertFalse(invalid_config.is_valid())


class TestTPSLResult(unittest.TestCase):
    """Test TP/SL result class."""
    
    def test_result_creation(self):
        """Test result object creation."""
        result = TPSLResult(
            should_trigger=True,
            trigger_type=TPSLType.PERCENTAGE_BASED,
            trigger_price=105.0,
            current_price=106.0,
            profit_loss=6.0,
            trigger_reason="Take profit percentage reached"
        )
        
        self.assertTrue(result.should_trigger)
        self.assertEqual(result.trigger_type, TPSLType.PERCENTAGE_BASED)
        self.assertEqual(result.trigger_price, 105.0)
        self.assertEqual(result.current_price, 106.0)
        self.assertEqual(result.profit_loss, 6.0)
        self.assertIn("Take profit", result.trigger_reason)


if __name__ == '__main__':
    unittest.main()