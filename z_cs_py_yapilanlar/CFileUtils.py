from datetime import datetime
from CTxtFileWriter import CTxtFileWriter

class CFileUtils:
    def __init__(self):
        self.SistemId = 0
        self.SistemAdi = ""
        self.GrafikSembol = ""
        self.GrafikPeriyot = 0
        self.BarCount = 0
        self.LogEnabled = False
        self.LogFileName = ""
        self.LogFileManager = CTxtFileWriter()
        self.LogFileIsOpened = False
        self.LogMessage = ""

    def __del__(self):
        self.LogFileManager = None

    def initialize(self, Sistem):
        self.SistemId = 0
        self.SistemAdi = "SistemAdi"
        self.GrafikSembol = "GrafikSembol"
        self.GrafikPeriyot = 1
        self.BarCount = 0
        self.reset(Sistem)
        return 0

    def reset(self, Sistem):
        return self

    def enable_logging(self, Sistem):
        self.LogEnabled = True
        return self

    def disable_logging(self, Sistem):
        self.LogEnabled = False
        return self

    def open_log_file(self, Sistem, LogFileName, LastUpdatTimeEnabled=True, SummaryEnabled=False):
        if self.LogEnabled and not self.LogFileIsOpened:
            self.LogFileIsOpened = True
            self.LogFileManager.open_file(LogFileName)

            if self.LogFileIsOpened:
                if LastUpdatTimeEnabled:
                    self.write_last_update_time_to_log_file(Sistem)

                if SummaryEnabled:
                    self.LogMessage = f"#  {'Bar Sayisi':<10}   : {self.BarCount} 	"
                    self.LogFileManager.write_line(self.LogMessage)
                    self.LogMessage = f"#  {'Sistem Id':<10}    : {self.SistemId} 	"
                    self.LogFileManager.write_line(self.LogMessage)
                    self.LogMessage = f"#  {'Sistem Adi':<10}   : {self.SistemAdi} 	"
                    self.LogFileManager.write_line(self.LogMessage)
                    self.LogMessage = f"#  {'Sembol':<10}       : {self.GrafikSembol} 	"
                    self.LogFileManager.write_line(self.LogMessage)
                    self.LogMessage = f"#  {'Periyot':<10}      : {self.GrafikPeriyot} 	"
                    self.LogFileManager.write_line(self.LogMessage)
        return self

    def write_last_update_time_to_log_file(self, Sistem):
        if self.LogEnabled and self.LogFileIsOpened:
            self.LogMessage = f"#  {'Log Zamani':<10}   : {datetime.now().strftime('%Y.%m.%d %H:%M:%S')} 	"
            self.LogFileManager.write_line(self.LogMessage)
        return self

    def write_to_log_file(self, Sistem, LogMessage, AppendNewLine=True):
        if self.LogEnabled and self.LogFileIsOpened:
            if AppendNewLine:
                self.LogFileManager.write_line(LogMessage)
            else:
                self.LogFileManager.write(LogMessage)
        return self

    def close_log_file(self, Sistem):
        if self.LogEnabled and self.LogFileIsOpened:
            self.LogFileManager.close_file()
            self.LogFileIsOpened = False
        return self
