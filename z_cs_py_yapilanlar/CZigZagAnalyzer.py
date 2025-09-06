from CBase import CBase
from CFileUtils import CFileUtils
from CTimeUtils import CTimeUtils

class CZigZagAnalyzer(CBase):
    def __init__(self):
        super().__init__()
        self.ZigZagParam = 0.0
        self.ZigZag = []
        self.EmirList = []
        self.LogMessageList = []
        self.PercentageChangeList = []
        self.delimiter = ";"

    def initialize(self, Sistem, V, Open, High, Low, Close, Volume, Lot):
        self.set_data(Sistem, V, Open, High, Low, Close, Volume, Lot)
        return self

    def reset(self, Sistem):
        self.ZigZag = None
        self.EmirList.clear()
        self.LogMessageList.clear()
        return self

    def calculate(self, Sistem, Source, Percentage):
        self.ZigZagParam = Percentage
        self.ZigZag = Sistem.ZigZagPercent(Source, Percentage)

    def fill_emir_list(self, Sistem):
        Emir = ""
        BarIndex = 0
        ListeyeEkle = False
        self.EmirList.clear()
        for i in range(self.BarCount):
            self.EmirList.append("")
            if i < 2:
                continue
            al1 = True
            al1 = al1 and (self.ZigZag[i - 2] > self.ZigZag[i - 1]) and (self.ZigZag[i - 1] < self.ZigZag[i])
            sat1 = True
            sat1 = sat1 and (self.ZigZag[i - 2] < self.ZigZag[i - 1]) and (self.ZigZag[i - 1] > self.ZigZag[i])
            if al1:
                Emir = "A"
                BarIndex = i - 1
                self.EmirList[BarIndex] = Emir
                ListeyeEkle = True
            if sat1:
                Emir = "S"
                BarIndex = i - 1
                self.EmirList[BarIndex] = Emir
                ListeyeEkle = True
            if ListeyeEkle:
                ListeyeEkle = False

    def calculate_percentage_change(self, Sistem):
        PrevValue = self.ZigZag[0]
        self.PercentageChangeList = Sistem.Liste(self.BarCount, 0.0)
        for i in range(self.BarCount):
            PercentageChange = (self.ZigZag[i] - PrevValue) * 100.0 / PrevValue
            self.PercentageChangeList[i] = PercentageChange
            if i < 1:
                continue
            al1 = True
            sat1 = True
            al1 = al1 and (self.EmirList[i] == "A")
            sat1 = sat1 and (self.EmirList[i] == "S")
            if al1:
                PrevValue = self.ZigZag[i]
            if sat1:
                PrevValue = self.ZigZag[i]

    def create_log_message(self, Sistem, BarIndex, ZigZagValue, SistemYon, Param1=0.0, Param2=0.0, Param3=0.0):
        i = BarIndex
        return f'{self.delimiter} {i:<5} {self.delimiter} {ZigZagValue:10.2f} {self.delimiter} {SistemYon:>10} {self.delimiter} {BarIndex:10} {self.delimiter} {Param1:10.2f} {self.delimiter} {Param2:10.2f} {self.delimiter} {Param3:10.2f} {self.delimiter}'

    def add_log_message(self, Sistem, LogMessage):
        self.LogMessageList.append(LogMessage)

    def update_log_message(self, Sistem, BarIndex, LogMessage):
        i = BarIndex
        self.LogMessageList[i] = LogMessage

    def write_to_file(self, Sistem, FileName):
        myFileUtils = CFileUtils()
        myTimeUtils = CTimeUtils()
        myTimeUtils.initialize(Sistem, self.V, self.Open, self.High, self.Low, self.Close, self.Volume, self.Lot)
        myTimeUtils.gecen_zaman_bilgilerini_al(Sistem)
        aciklama1 = "..."
        aciklama2 = "..."
        aciklama3 = "..."
        aciklama4 = "..."
        aciklama5 = "..."
        aciklama6 = "..."
        logFileFullName = FileName.strip()
