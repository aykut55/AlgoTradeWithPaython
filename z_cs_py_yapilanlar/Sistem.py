import datetime

class Sistem:
    def __init__(self):
        self.Name = ""
        self.Sembol = ""
        self.Periyot = ""
        self.SelectBarNo = 0
        self.SelectTarih = datetime.datetime.now()
        self.GetiriMaxDD = 0.0
        self.GetiriMaxDDTarih = datetime.datetime.now()
        self.ProfitFactor = 0.0
        self.GrafikVerileri = []
        self.GetiriKZ = []
        self.Yon = []
        self.Seviye = []
        self.Cizgiler = [Cizgi() for _ in range(50)]
        self.BarSayisi = 0
        self.Parametreler = ["" for _ in range(20)]

    def Mesaj(self, Message):
        print(Message)

    def Liste(self, BarCount, initial_value):
        return [initial_value] * BarCount

    def GrafikFiyatOku(self, Data, type):
        pass

    def ZeminYazisiEkle(self, Message, PanelNo, ColNum, RowNum, MessageColor, FontName, FontSize):
        pass

    def DikeyCizgiEkle(self, i, Color, width, style):
        pass

    def SistemGetir(self, SistemAdi, Sembol, Periyot):
        pass

    def DolguEkle(self, cizgi1, cizgi2, color1, color2):
        pass

    def GetiriHesapla(self, BaslangicTarihi, KaymaMiktari):
        pass

    def GetiriMaxDDHesapla(self, BaslangicTarihi, BitisTarihi):
        pass

    def NesneGetir(self, Key):
        pass

    def NesneKaydet(self, Key, Value):
        pass

    def ZigZagPercent(self, Source, Percentage):
        pass

    def BarCiz(self, BarPanelNo, BarTip, BarOpen, BarHigh, BarLow, BarClose, ColorGreen, ColorRed):
        pass

    def ExcelKopyala(self, ExcelArray, FileName):
        pass

    def ExcelOku(self, FileName):
        pass

    def MA(self, Source, Method, Periyod):
        pass

    def MA2(self, Source, Method, Periyod):
        pass

    def MA3(self, Source, Method, Periyod):
        pass

    def MAM(self, Source, Method, p1, p2, p3, p4, p5, p6, p7, p8):
        pass

    def RSI(self, Source, Periyod):
        pass

    def MACD(self, Source, PeriyodFast, PeriyodSlow):
        pass

    def StochasticFast(self, Source, Periyod1, Periyod2):
        pass

    def StochasticSlow(self, Source, Periyod1, Periyod2):
        pass

    def StochasticRSI(self, Source, Periyod):
        pass

    def StochasticOsc(self, Source, Periyod1, Periyod2):
        pass

    def TOMA(self, Source, Periyod, Percentage, Method):
        pass

class Cizgi:
    def __init__(self):
        self.Deger = []
        self.Aciklama = ""
        self.Renk = ""
