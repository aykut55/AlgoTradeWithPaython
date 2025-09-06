import ctypes
from CTrader import CTrader
from CVarlikManager import CVarlikManager
from CUtils import CUtils
from CTimeUtils import CTimeUtils
from CBarUtils import CBarUtils
from CFileUtils import CFileUtils
from CExcelFileHandler import CExcelFileHandler
from CBirlesikSistemManager import CBirlesikSistemManager
from CSharedMemory import CSharedMemory
from CZigZagAnalyzer import CZigZagAnalyzer
from CConfigManager import CConfigManager
from CSystemWrapper import CSystemWrapper
from CIndicatorManager import CIndicatorManager

class Lib:
    def __init__(self):
        self.TraderArray = [None] * 100
        self.VarlikManager = None
        self.Utils = None
        self.TimeUtils = None
        self.BarUtils = None
        self.FileUtils = None
        self.ExcelFileHandler = None
        self.BirlesikSistemManager = None
        self.SharedMemory = None
        self.ZigZagAnalyzer = None
        self.ConfigManager = None
        self.SystemWrapper = None
        self.IndicatorManager = None
        self.kernel32 = ctypes.windll.kernel32
        self.GetTickCount64 = self.kernel32.GetTickCount64
        self.GetTickCount64.restype = ctypes.c_ulonglong

    def __del__(self):
        pass

    def ShowMessage(self, Sistem, Message="This message comes from Lib::ShowMessage()"):
        Sistem.Mesaj(Message)

    def GetTrader(self, Sistem, Index=0):
        if 0 <= Index < 100:
            if self.TraderArray[Index] is None:
                self.TraderArray[Index] = CTrader(Sistem, Index)
            return self.TraderArray[Index]
        return None

    def DeleteTrader(self, Trader):
        if Trader is not None:
            Trader = None

    def GetVarlik(self, Sistem):
        if self.VarlikManager is None:
            self.VarlikManager = CVarlikManager()
        return self.VarlikManager

    def DeleteVarlik(self, VarlikManager):
        if VarlikManager is not None:
            VarlikManager = None

    def GetUtils(self, Sistem):
        if self.Utils is None:
            self.Utils = CUtils()
        return self.Utils

    def DeleteUtils(self, Utils):
        if Utils is not None:
            Utils = None

    def GetTimeUtils(self, Sistem):
        if self.TimeUtils is None:
            self.TimeUtils = CTimeUtils()
        return self.TimeUtils

    def DeleteTimeUtils(self, TimeUtils):
        if TimeUtils is not None:
            TimeUtils = None

    def GetBarUtils(self, Sistem):
        if self.BarUtils is None:
            self.BarUtils = CBarUtils()
        return self.BarUtils

    def DeleteBarUtils(self, BarUtils):
        if BarUtils is not None:
            BarUtils = None

    def GetFileUtils(self, Sistem):
        if self.FileUtils is None:
            self.FileUtils = CFileUtils()
        return self.FileUtils

    def DeleteFileUtils(self, FileUtils):
        if FileUtils is not None:
            FileUtils = None

    def GetExcelFileHandler(self, Sistem):
        if self.ExcelFileHandler is None:
            self.ExcelFileHandler = CExcelFileHandler()
        return self.ExcelFileHandler

    def DeleteExcelFileHandler(self, ExcelFileHandler):
        if ExcelFileHandler is not None:
            ExcelFileHandler = None

    def GetBirlesikSistemManager(self, Sistem):
        if self.BirlesikSistemManager is None:
            self.BirlesikSistemManager = CBirlesikSistemManager()
        return self.BirlesikSistemManager

    def DeleteBirlesikSistemManager(self, BirlesikSistemManager):
        if BirlesikSistemManager is not None:
            BirlesikSistemManager = None

    def GetSharedMemory(self, Sistem):
        if self.SharedMemory is None:
            self.SharedMemory = CSharedMemory()
        return self.SharedMemory

    def DeleteSharedMemory(self, SharedMemory):
        if SharedMemory is not None:
            SharedMemory = None

    def GetZigZagAnalyzer(self, Sistem):
        if self.ZigZagAnalyzer is None:
            self.ZigZagAnalyzer = CZigZagAnalyzer()
        return self.ZigZagAnalyzer

    def DeleteZigZagAnalyzer(self, ZigZagAnalyzer):
        if ZigZagAnalyzer is not None:
            ZigZagAnalyzer = None

    def GetConfig(self, Sistem):
        if self.ConfigManager is None:
            self.ConfigManager = CConfigManager()
        return self.ConfigManager

    def DeleteConfig(self, ConfigManager):
        if ConfigManager is not None:
            ConfigManager = None

    def GetSystemWrapper(self, Sistem):
        if self.SystemWrapper is None:
            self.SystemWrapper = CSystemWrapper()
        return self.SystemWrapper

    def DeleteSystemWrapper(self, SystemWrapper):
        if SystemWrapper is not None:
            SystemWrapper = None

    def GetIndicatorManager(self, Sistem):
        if self.IndicatorManager is None:
            self.IndicatorManager = CIndicatorManager()
        return self.IndicatorManager

    def DeleteIndicatorManager(self, IndicatorManager):
        if IndicatorManager is not None:
            IndicatorManager = None
