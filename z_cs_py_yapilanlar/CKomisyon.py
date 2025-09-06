class CKomisyon:
    def __init__(self):
        self.Trader = None

    def __del__(self):
        pass

    def initialize(self, Sistem, Trader):
        self.Trader = Trader
        return self

    def reset(self, Sistem):
        return self

    def hesapla(self, Sistem, BarIndex):
        result = 0
        i = BarIndex
        self.Trader.Status.KomisyonFiyat = self.Trader.Lists.KomisyonIslemSayisiList[i] * float(self.Trader.Status.KomisyonCarpan) * self.Trader.Status.KomisyonVarlikAdedSayisi
        self.Trader.Lists.KomisyonFiyatList[i] = self.Trader.Status.KomisyonFiyat
        return result
