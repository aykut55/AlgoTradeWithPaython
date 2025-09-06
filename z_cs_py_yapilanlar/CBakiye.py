class CBakiye:
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

        # Bakiye (Puan)
        self.Trader.Lists.BakiyePuanList[i] = self.Trader.Status.BakiyePuan + self.Trader.Lists.KarZararPuanList[i]
        self.Trader.Lists.GetiriPuanList[i] = self.Trader.Lists.BakiyePuanList[i] - self.Trader.Status.IlkBakiyePuan

        if self.Trader.Flags.BakiyeGuncelle:
            self.Trader.Status.BakiyePuan = self.Trader.Lists.BakiyePuanList[i]
            self.Trader.Status.GetiriPuan = self.Trader.Lists.GetiriPuanList[i]

            if self.Trader.Lists.KarZararPuanList[i] >= 0:
                self.Trader.Status.ToplamKarPuan += self.Trader.Lists.KarZararPuanList[i]
            elif self.Trader.Lists.KarZararPuanList[i] < 0:
                self.Trader.Status.ToplamZararPuan += self.Trader.Lists.KarZararPuanList[i]

            self.Trader.Status.NetKarPuan = self.Trader.Status.ToplamKarPuan + self.Trader.Status.ToplamZararPuan

        # Bakiye (Fiyat)
        self.Trader.Lists.BakiyeFiyatList[i] = self.Trader.Status.BakiyeFiyat + self.Trader.Lists.KarZararFiyatList[i]
        self.Trader.Lists.GetiriFiyatList[i] = self.Trader.Lists.BakiyeFiyatList[i] - self.Trader.Status.IlkBakiyeFiyat

        if self.Trader.Flags.BakiyeGuncelle:
            self.Trader.Status.BakiyeFiyat = self.Trader.Lists.BakiyeFiyatList[i]
            self.Trader.Status.GetiriFiyat = self.Trader.Lists.GetiriFiyatList[i]

            if self.Trader.Lists.KarZararFiyatList[i] >= 0:
                self.Trader.Status.ToplamKarFiyat += self.Trader.Lists.KarZararFiyatList[i]
            elif self.Trader.Lists.KarZararFiyatList[i] < 0:
                self.Trader.Status.ToplamZararFiyat += self.Trader.Lists.KarZararFiyatList[i]

            self.Trader.Status.NetKarFiyat = self.Trader.Status.ToplamKarFiyat + self.Trader.Status.ToplamZararFiyat

        if self.Trader.Status.IlkBakiyePuan != 0.0:
            self.Trader.Lists.GetiriPuanYuzdeList[i] = 100.0 * self.Trader.Lists.GetiriPuanList[i] / float(self.Trader.Status.IlkBakiyePuan)
        else:
            self.Trader.Lists.GetiriPuanYuzdeList[i] = 0.0

        if self.Trader.Status.IlkBakiyeFiyat != 0.0:
            self.Trader.Lists.GetiriFiyatYuzdeList[i] = 100.0 * self.Trader.Lists.GetiriFiyatList[i] / float(self.Trader.Status.IlkBakiyeFiyat)
        else:
            self.Trader.Lists.GetiriFiyatYuzdeList[i] = 0.0

        if self.Trader.Flags.BakiyeGuncelle:
            self.Trader.Status.GetiriPuanYuzde = self.Trader.Lists.GetiriPuanYuzdeList[i]
            self.Trader.Status.GetiriFiyatYuzde = self.Trader.Lists.GetiriFiyatYuzdeList[i]

        k = 1.0 if self.Trader.Status.KomisyonCarpan != 0.0 else 0.0

        self.Trader.Lists.GetiriFiyatNetList[i] = self.Trader.Lists.GetiriFiyatList[i] - self.Trader.Lists.KomisyonFiyatList[i] * k
        self.Trader.Lists.BakiyeFiyatNetList[i] = self.Trader.Lists.GetiriFiyatNetList[i] + self.Trader.Status.IlkBakiyeFiyat

        self.Trader.Lists.GetiriFiyatYuzdeNetList[i] = 0.0
        if self.Trader.Status.IlkBakiyeFiyat != 0.0:
            self.Trader.Lists.GetiriFiyatYuzdeNetList[i] = 100.0 * self.Trader.Lists.GetiriFiyatNetList[i] / float(self.Trader.Status.IlkBakiyeFiyat)

        self.Trader.Lists.GetiriKz[i] = self.Trader.Lists.GetiriFiyatList[i] / self.Trader.Status.VarlikAdedSayisi
        self.Trader.Lists.GetiriKzNet[i] = self.Trader.Lists.GetiriFiyatNetList[i] / self.Trader.Status.VarlikAdedSayisi

        if i == self.Trader.BarCount - 1:
            self.Trader.Status.BakiyeFiyat = self.Trader.Lists.BakiyeFiyatList[-1]
            self.Trader.Status.GetiriFiyat = self.Trader.Lists.GetiriFiyatList[-1]
            self.Trader.Status.GetiriKz = self.Trader.Lists.GetiriKz[-1]
            self.Trader.Status.GetiriFiyatYuzde = self.Trader.Lists.GetiriFiyatYuzdeList[-1]
            self.Trader.Status.BakiyeFiyatNet = self.Trader.Lists.BakiyeFiyatNetList[-1]
            self.Trader.Status.GetiriFiyatNet = self.Trader.Lists.GetiriFiyatNetList[-1]
            self.Trader.Status.GetiriKzNet = self.Trader.Lists.GetiriKzNet[-1]
            self.Trader.Status.GetiriFiyatYuzdeNet = self.Trader.Lists.GetiriFiyatYuzdeNetList[-1]
            self.Trader.Status.BakiyePuan = self.Trader.Lists.BakiyePuanList[-1]
            self.Trader.Status.GetiriPuan = self.Trader.Lists.GetiriPuanList[-1]
            self.Trader.Status.BakiyePuanNet = self.Trader.Lists.BakiyePuanNetList[-1]
            self.Trader.Status.GetiriPuanNet = self.Trader.Lists.GetiriPuanNetList[-1]
            self.Trader.Status.GetiriPuanYuzdeNet = self.Trader.Lists.GetiriPuanYuzdeNetList[-1]

        return result
