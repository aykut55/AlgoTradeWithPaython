class CLists:
    def __init__(self):
        self.BarIndexList = []
        self.YonList = []
        self.SeviyeList = []
        self.SinyalList = []
        self.KarZararPuanList = []
        self.KarZararFiyatList = []
        self.KarZararFiyatYuzdeList = []
        self.KarAlList = []
        self.IzleyenStopList = []
        self.IslemSayisiList = []
        self.AlisSayisiList = []
        self.SatisSayisiList = []
        self.FlatSayisiList = []
        self.PassSayisiList = []
        self.KontratSayisiList = []
        self.VarlikAdedSayisiList = []
        self.KomisyonVarlikAdedSayisiList = []
        self.KomisyonIslemSayisiList = []
        self.KomisyonFiyatList = []
        self.KardaBarSayisiList = []
        self.ZarardaBarSayisiList = []
        self.BakiyePuanList = []
        self.BakiyeFiyatList = []
        self.GetiriPuanList = []
        self.GetiriFiyatList = []
        self.GetiriPuanYuzdeList = []
        self.GetiriFiyatYuzdeList = []
        self.BakiyePuanNetList = []
        self.BakiyeFiyatNetList = []
        self.GetiriPuanNetList = []
        self.GetiriFiyatNetList = []
        self.GetiriPuanYuzdeNetList = []
        self.GetiriFiyatYuzdeNetList = []
        self.GetiriKz = []
        self.GetiriKzNet = []
        self.GetiriKzSistem = []
        self.GetiriKzNetSistem = []
        self.EmirKomutList = []
        self.EmirStatusList = []

    def __del__(self):
        pass

    def initialize(self, Sistem):
        return self

    def reset(self, Sistem):
        self.create_lists(Sistem, Sistem.BarSayisi)
        return self

    def create_lists(self, Sistem, BarCount):
        self.BarIndexList = [0] * BarCount
        self.YonList = [""] * BarCount
        self.SeviyeList = [0.0] * BarCount
        self.SinyalList = [0.0] * BarCount
        self.KarZararPuanList = [0.0] * BarCount
        self.KarZararFiyatList = [0.0] * BarCount
        self.KarZararFiyatYuzdeList = [0.0] * BarCount
        self.KarAlList = [0.0] * BarCount
        self.IzleyenStopList = [0.0] * BarCount
        self.IslemSayisiList = [0.0] * BarCount
        self.AlisSayisiList = [0.0] * BarCount
        self.SatisSayisiList = [0.0] * BarCount
        self.FlatSayisiList = [0.0] * BarCount
        self.PassSayisiList = [0.0] * BarCount
        self.KontratSayisiList = [0.0] * BarCount
        self.VarlikAdedSayisiList = [0.0] * BarCount
        self.KomisyonVarlikAdedSayisiList = [0.0] * BarCount
        self.KomisyonIslemSayisiList = [0.0] * BarCount
        self.KomisyonFiyatList = [0.0] * BarCount
        self.KardaBarSayisiList = [0.0] * BarCount
        self.ZarardaBarSayisiList = [0.0] * BarCount
        self.BakiyePuanList = [0.0] * BarCount
        self.BakiyeFiyatList = [0.0] * BarCount
        self.GetiriPuanList = [0.0] * BarCount
        self.GetiriFiyatList = [0.0] * BarCount
        self.GetiriPuanYuzdeList = [0.0] * BarCount
        self.GetiriFiyatYuzdeList = [0.0] * BarCount
        self.BakiyePuanNetList = [0.0] * BarCount
        self.BakiyeFiyatNetList = [0.0] * BarCount
        self.GetiriPuanNetList = [0.0] * BarCount
        self.GetiriFiyatNetList = [0.0] * BarCount
        self.GetiriPuanYuzdeNetList = [0.0] * BarCount
        self.GetiriFiyatYuzdeNetList = [0.0] * BarCount
        self.GetiriKz = [0.0] * BarCount
        self.GetiriKzNet = [0.0] * BarCount
        self.GetiriKzSistem = [0.0] * BarCount
        self.GetiriKzNetSistem = [0.0] * BarCount
        self.EmirKomutList = [0.0] * BarCount
        self.EmirStatusList = [0.0] * BarCount
        return self
