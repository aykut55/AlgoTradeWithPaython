class CKarZarar:
    def __init__(self):
        self.V = None
        self.Trader = None

    def __del__(self):
        pass

    def initialize(self, Sistem, Trader):
        self.Trader = Trader
        self.V = Trader.V
        return self

    def reset(self, Sistem):
        return self

    def anlik_kar_zarar_hesapla(self, Sistem, BarIndex, Type="C"):
        result = 0
        i = BarIndex
        AnlikFiyat = self.V[i].Close

        AnlikKarZararHesaplaEnabled = self.Trader.Flags.AnlikKarZararHesaplaEnabled
        SonYon = self.Trader.Signals.SonYon
        SonFiyat = self.Trader.Signals.SonFiyat
        VarlikAdedSayisi = self.Trader.VarlikManager.VarlikAdedSayisi

        if AnlikKarZararHesaplaEnabled:
            if Type != "C":
                if Type == "O":
                    AnlikFiyat = self.V[i].Open
                elif Type == "H":
                    AnlikFiyat = self.V[i].High
                elif Type == "L":
                    AnlikFiyat = self.V[i].Low

            if SonYon == "A":
                self.Trader.Status.KarZararPuan = AnlikFiyat - SonFiyat
                self.Trader.Status.KarZararFiyat = self.Trader.Status.KarZararPuan * VarlikAdedSayisi
                self.Trader.Lists.KarZararPuanList[i] = self.Trader.Status.KarZararPuan
                self.Trader.Lists.KarZararFiyatList[i] = self.Trader.Status.KarZararFiyat
                if SonFiyat != 0:
                    self.Trader.Status.KarZararFiyatYuzde = 100.0 * self.Trader.Status.KarZararPuan / SonFiyat
                else:
                    self.Trader.Status.KarZararFiyatYuzde = 0.0
                self.Trader.Lists.KarZararFiyatYuzdeList[i] = self.Trader.Status.KarZararFiyatYuzde
            
            elif SonYon == "S":
                self.Trader.Status.KarZararPuan = SonFiyat - AnlikFiyat
                self.Trader.Status.KarZararFiyat = self.Trader.Status.KarZararPuan * VarlikAdedSayisi
                self.Trader.Lists.KarZararPuanList[i] = self.Trader.Status.KarZararPuan
                self.Trader.Lists.KarZararFiyatList[i] = self.Trader.Status.KarZararFiyat
                if SonFiyat != 0:
                    self.Trader.Status.KarZararFiyatYuzde = 100.0 * self.Trader.Status.KarZararPuan / SonFiyat
                else:
                    self.Trader.Status.KarZararFiyatYuzde = 0.0
                self.Trader.Lists.KarZararFiyatYuzdeList[i] = self.Trader.Status.KarZararFiyatYuzde

            if self.Trader.Status.KarZararPuan > 0:
                self.Trader.Status.KardaBarSayisi += 1
                self.Trader.Status.ZarardaBarSayisi -= 1
            elif self.Trader.Status.KarZararPuan == 0:
                self.Trader.Status.KardaBarSayisi = 0
                self.Trader.Status.ZarardaBarSayisi = 0
            else:
                self.Trader.Status.KardaBarSayisi -= 1
                self.Trader.Status.ZarardaBarSayisi += 1
                
        return result
