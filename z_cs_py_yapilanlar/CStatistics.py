import datetime
from CTimeUtils import CTimeUtils
from CTxtFileWriter import CTxtFileWriter

class CStatistics:
    def __init__(self):
        self.Trader = None
        self.IstatistiklerNew = {}
        self.SistemId = 0
        self.SistemName = ""
        self.GrafikSembol = ""
        self.GrafikPeriyot = ""
        self.LastBarIndex = 0
        self.LastExecutionId = 0
        self.LastExecutionTime = ""
        self.LastExecutionTimeStart = ""
        self.LastExecutionTimeStop = ""
        self.ExecutionTimeInMSec = 0
        self.LastResetTime = ""
        self.LastStatisticsCalculationTime = ""
        self.ToplamGecenSureAy = 0.0
        self.ToplamGecenSureGun = 0.0
        self.ToplamGecenSureSaat = 0.0
        self.ToplamGecenSureDakika = 0.0
        self.ToplamBarSayisi = 0
        self.BarBegIndex = 0
        self.BarEndIndex = 0
        self.SecilenBarTarihi = None
        self.SecilenBarNumarasi = 0
        self.SecilenBarAcilisFiyati = 0.0
        self.SecilenBarYuksekFiyati = 0.0
        self.SecilenBarDusukFiyati = 0.0
        self.SecilenBarKapanisFiyati = 0.0
        self.IlkBarTarihi = None
        self.SonBarTarihi = None
        self.IlkBarIndex = 0
        self.SonBarIndex = 0
        self.SonBarAcilisFiyati = 0.0
        self.SonBarYuksekFiyati = 0.0
        self.SonBarDusukFiyati = 0.0
        self.SonBarKapanisFiyati = 0.0
        self.IlkBakiyeFiyat = 0.0
        self.IlkBakiyePuan = 0.0
        self.BakiyeFiyat = 0.0
        self.BakiyePuan = 0.0
        self.GetiriFiyat = 0.0
        self.GetiriPuan = 0.0
        self.GetiriFiyatYuzde = 0.0
        self.GetiriPuanYuzde = 0.0
        self.BakiyeFiyatNet = 0.0
        self.BakiyePuanNet = 0.0
        self.GetiriFiyatNet = 0.0
        self.GetiriPuanNet = 0.0
        self.GetiriFiyatYuzdeNet = 0.0
        self.GetiriPuanYuzdeNet = 0.0
        self.GetiriKz = 0.0
        self.GetiriKzNet = 0.0
        self.MinBakiyeFiyat = 0.0
        self.MaxBakiyeFiyat = 0.0
        self.MinBakiyePuan = 0.0
        self.MaxBakiyePuan = 0.0
        self.MinBakiyeFiyatYuzde = 0.0
        self.MaxBakiyeFiyatYuzde = 0.0
        self.MinBakiyeFiyatIndex = 0
        self.MaxBakiyeFiyatIndex = 0
        self.MinBakiyePuanIndex = 0
        self.MaxBakiyePuanIndex = 0
        self.MinBakiyeFiyatNet = 0.0
        self.MaxBakiyeFiyatNet = 0.0
        self.MinBakiyeFiyatNetIndex = 0
        self.MaxBakiyeFiyatNetIndex = 0
        self.MinBakiyeFiyatNetYuzde = 0.0
        self.MaxBakiyeFiyatNetYuzde = 0.0
        self.GetiriKzSistem = 0.0
        self.GetiriKzNetSistem = 0.0
        self.GetiriKzSistemYuzde = 0.0
        self.GetiriKzNetSistemYuzde = 0.0
        self.GetiriFiyatTipi = ""
        self.IslemSayisi = 0
        self.AlisSayisi = 0
        self.SatisSayisi = 0
        self.FlatSayisi = 0
        self.PassSayisi = 0
        self.KarAlSayisi = 0
        self.ZararKesSayisi = 0
        self.KazandiranIslemSayisi = 0
        self.KaybettirenIslemSayisi = 0
        self.NotrIslemSayisi = 0
        self.KazandiranAlisSayisi = 0
        self.KaybettirenAlisSayisi = 0
        self.NotrAlisSayisi = 0
        self.KazandiranSatisSayisi = 0
        self.KaybettirenSatisSayisi = 0
        self.NotrSatisSayisi = 0
        self.AlKomutSayisi = 0
        self.SatKomutSayisi = 0
        self.PasGecKomutSayisi = 0
        self.KarAlKomutSayisi = 0
        self.ZararKesKomutSayisi = 0
        self.FlatOlKomutSayisi = 0
        self.KomisyonIslemSayisi = 0
        self.KomisyonVarlikAdedSayisi = 1
        self.KomisyonCarpan = 0.0
        self.KomisyonFiyat = 0.0
        self.KomisyonFiyatYuzde = 0.0
        self.KomisyonuDahilEt = False
        self.KarZararFiyat = 0.0
        self.KarZararPuan = 0.0
        self.KarZararFiyatYuzde = 0.0
        self.NetKarFiyat = 0.0
        self.ToplamKarFiyat = 0.0
        self.ToplamZararFiyat = 0.0
        self.NetKarPuan = 0.0
        self.ToplamKarPuan = 0.0
        self.ToplamZararPuan = 0.0
        self.MaxKarFiyat = 0.0
        self.MaxZararFiyat = 0.0
        self.MaxKarPuan = 0.0
        self.MaxZararPuan = 0.0
        self.MaxZararFiyatIndex = 0
        self.MaxKarFiyatIndex = 0
        self.MaxZararPuanIndex = 0
        self.MaxKarPuanIndex = 0
        self.KardaBarSayisi = 0
        self.ZarardaBarSayisi = 0
        self.KarliIslemOrani = 0.0
        self.GetiriMaxDD = 0.0
        self.GetiriMaxDDTarih = None
        self.GetiriMaxKayip = 0.0
        self.ProfitFactor = 0.0
        self.ProfitFactorSistem = 0.0
        self.OrtAylikIslemSayisi = 0.0
        self.OrtHaftalikIslemSayisi = 0.0
        self.OrtGunlukIslemSayisi = 0.0
        self.OrtSaatlikIslemSayisi = 0.0
        self.Sinyal = ""
        self.SonYon = ""
        self.PrevYon = ""
        self.SonFiyat = 0.0
        self.SonAFiyat = 0.0
        self.SonSFiyat = 0.0
        self.SonFFiyat = 0.0
        self.SonPFiyat = 0.0
        self.PrevFiyat = 0.0
        self.PrevAFiyat = 0.0
        self.PrevSFiyat = 0.0
        self.PrevFFiyat = 0.0
        self.PrevPFiyat = 0.0
        self.SonBarNo = 0
        self.SonABarNo = 0
        self.SonSBarNo = 0
        self.SonFBarNo = 0
        self.SonPBarNo = 0
        self.PrevBarNo = 0
        self.PrevABarNo = 0
        self.PrevSBarNo = 0
        self.PrevFBarNo = 0
        self.PrevPBarNo = 0
        self.EmirKomut = 0.0
        self.EmirStatus = 0.0
        self.HisseSayisi = 0
        self.KontratSayisi = 0
        self.VarlikAdedCarpani = 0
        self.VarlikAdedSayisi = 0
        self.KaymaMiktari = 0.0
        self.KaymayiDahilEt = False
        self.GetiriFiyatBuAy = 0.0
        self.GetiriFiyatAy1 = 0.0
        self.GetiriFiyatAy2 = 0.0
        self.GetiriFiyatAy3 = 0.0
        self.GetiriFiyatAy4 = 0.0
        self.GetiriFiyatAy5 = 0.0
        self.GetiriFiyatBuHafta = 0.0
        self.GetiriFiyatHafta1 = 0.0
        self.GetiriFiyatHafta2 = 0.0
        self.GetiriFiyatHafta3 = 0.0
        self.GetiriFiyatHafta4 = 0.0
        self.GetiriFiyatHafta5 = 0.0
        self.GetiriFiyatBuGun = 0.0
        self.GetiriFiyatGun1 = 0.0
        self.GetiriFiyatGun2 = 0.0
        self.GetiriFiyatGun3 = 0.0
        self.GetiriFiyatGun4 = 0.0
        self.GetiriFiyatGun5 = 0.0
        self.GetiriFiyatBuSaat = 0.0
        self.GetiriFiyatSaat1 = 0.0
        self.GetiriFiyatSaat2 = 0.0
        self.GetiriFiyatSaat3 = 0.0
        self.GetiriFiyatSaat4 = 0.0
        self.GetiriFiyatSaat5 = 0.0
        self.GetiriPuanBuAy = 0.0
        self.GetiriPuanAy1 = 0.0
        self.GetiriPuanAy2 = 0.0
        self.GetiriPuanAy3 = 0.0
        self.GetiriPuanAy4 = 0.0
        self.GetiriPuanAy5 = 0.0
        self.GetiriPuanBuHafta = 0.0
        self.GetiriPuanHafta1 = 0.0
        self.GetiriPuanHafta2 = 0.0
        self.GetiriPuanHafta3 = 0.0
        self.GetiriPuanHafta4 = 0.0
        self.GetiriPuanHafta5 = 0.0
        self.GetiriPuanBuGun = 0.0
        self.GetiriPuanGun1 = 0.0
        self.GetiriPuanGun2 = 0.0
        self.GetiriPuanGun3 = 0.0
        self.GetiriPuanGun4 = 0.0
        self.GetiriPuanGun5 = 0.0
        self.GetiriPuanBuSaat = 0.0
        self.GetiriPuanSaat1 = 0.0
        self.GetiriPuanSaat2 = 0.0
        self.GetiriPuanSaat3 = 0.0
        self.GetiriPuanSaat4 = 0.0
        self.GetiriPuanSaat5 = 0.0
        self.colNums = []
        self.rowNums = []
        self.PanelNo = 0
        self.ColNum = 0
        self.RowNum = 0
        self.FontSize = 9
        self.col = 0
        self.row = 0
        self.idx = 0

    def initialize(self, Sistem, Trader):
        self.Trader = Trader
        return self

    def reset(self, Sistem):
        return self

    def hesapla(self, Sistem):
        result = 0
        V = self.Trader.V
        self.Trader.LastStatisticsCalculationTime = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
        self.read_values(Sistem)
        self.ToplamBarSayisi = self.Trader.BarCount
        self.IlkBarTarihi = V[0].Date
        self.SonBarTarihi = V[-1].Date
        LastBar = len(V) - 1
        self.SecilenBarNumarasi = Sistem.SelectBarNo
        self.SecilenBarTarihi = Sistem.SelectTarih
        if self.SecilenBarNumarasi <= LastBar:
            self.SecilenBarAcilisFiyati = V[self.SecilenBarNumarasi].Open
            self.SecilenBarYuksekFiyati = V[self.SecilenBarNumarasi].High
            self.SecilenBarDusukFiyati = V[self.SecilenBarNumarasi].Low
            self.SecilenBarKapanisFiyati = V[self.SecilenBarNumarasi].Close
        self.SonBarTarihi = V[LastBar].Date
        self.SonBarAcilisFiyati = V[LastBar].Open
        self.SonBarYuksekFiyati = V[LastBar].High
        self.SonBarDusukFiyati = V[LastBar].Low
        self.SonBarKapanisFiyati = V[LastBar].Close
        self.SonBarIndex = len(V) - 1
        sure_dakika = (datetime.datetime.now() - V[0].Date).total_seconds() / 60
        sure_saat = (datetime.datetime.now() - V[0].Date).total_seconds() / 3600
        sure_gun = (datetime.datetime.now() - V[0].Date).days
        sure_ay = sure_gun / 30.4
        self.ToplamGecenSureAy = float(sure_ay)
        self.ToplamGecenSureGun = int(sure_gun)
        self.ToplamGecenSureSaat = int(sure_saat)
        self.ToplamGecenSureDakika = int(sure_dakika)
        self.OrtAylikIslemSayisi = 1.0 * self.Trader.Status.IslemSayisi / self.ToplamGecenSureAy
        self.OrtHaftalikIslemSayisi = 0.0
        self.OrtGunlukIslemSayisi = 1.0 * self.Trader.Status.IslemSayisi / self.ToplamGecenSureGun
        self.OrtSaatlikIslemSayisi = 1.0 * self.Trader.Status.IslemSayisi / self.ToplamGecenSureSaat
        self.GetiriMaxDD = -1.0 * float(Sistem.GetiriMaxDD)
        self.GetiriMaxDDTarih = Sistem.GetiriMaxDDTarih
        self.GetiriMaxKayip = self.Trader.Status.VarlikAdedSayisi * -1 * self.GetiriMaxDD
        self.MaxKarFiyat = 0.0
        self.MaxZararFiyat = 0.0
        self.MaxKarPuan = 0.0
        self.MaxZararPuan = 0.0
        self.MinBakiyeFiyat = self.Trader.Status.IlkBakiyeFiyat
        self.MaxBakiyeFiyat = self.Trader.Status.IlkBakiyeFiyat
        self.MinBakiyeFiyatNet = self.Trader.Status.IlkBakiyeFiyat
        self.MaxBakiyeFiyatNet = self.Trader.Status.IlkBakiyeFiyat
        self.MinBakiyePuan = self.Trader.Status.IlkBakiyePuan
        self.MaxBakiyePuan = self.Trader.Status.IlkBakiyePuan
        for i in range(1, self.Trader.BarCount):
            if self.Trader.Lists.KarZararFiyatList[i] < self.MaxZararFiyat:
                self.MaxZararFiyat = self.Trader.Lists.KarZararFiyatList[i]
                self.MaxZararFiyatIndex = i
            if self.Trader.Lists.KarZararFiyatList[i] > self.MaxKarFiyat:
                self.MaxKarFiyat = self.Trader.Lists.KarZararFiyatList[i]
                self.MaxKarFiyatIndex = i
            if self.Trader.Lists.KarZararPuanList[i] < self.MaxZararPuan:
                self.MaxZararPuan = self.Trader.Lists.KarZararPuanList[i]
                self.MaxZararPuanIndex = i
            if self.Trader.Lists.KarZararPuanList[i] > self.MaxKarPuan:
                self.MaxKarPuan = self.Trader.Lists.KarZararPuanList[i]
                self.MaxKarPuanIndex = i
            if self.Trader.Lists.BakiyeFiyatList[i] < self.MinBakiyeFiyat:
                self.MinBakiyeFiyat = self.Trader.Lists.BakiyeFiyatList[i]
                self.MinBakiyeFiyatIndex = i
            if self.Trader.Lists.BakiyeFiyatList[i] > self.MaxBakiyeFiyat:
                self.MaxBakiyeFiyat = self.Trader.Lists.BakiyeFiyatList[i]
                self.MaxBakiyeFiyatIndex = i
            if self.Trader.Lists.BakiyeFiyatNetList[i] < self.MinBakiyeFiyatNet:
                self.MinBakiyeFiyatNet = self.Trader.Lists.BakiyeFiyatNetList[i]
                self.MinBakiyeFiyatNetIndex = i
            if self.Trader.Lists.BakiyeFiyatNetList[i] > self.MaxBakiyeFiyatNet:
                self.MaxBakiyeFiyatNet = self.Trader.Lists.BakiyeFiyatNetList[i]
                self.MaxBakiyeFiyatNetIndex = i
            if self.Trader.Lists.BakiyePuanList[i] < self.MinBakiyePuan:
                self.MinBakiyePuan = self.Trader.Lists.BakiyePuanList[i]
                self.MinBakiyePuanIndex = i
            if self.Trader.Lists.BakiyePuanList[i] > self.MaxBakiyePuan:
                self.MaxBakiyePuan = self.Trader.Lists.BakiyePuanList[i]
                self.MaxBakiyePuanIndex = i
        self.ProfitFactor = self.ToplamKarPuan / abs(self.ToplamZararPuan)
        self.ProfitFactorSistem = float(Sistem.ProfitFactor)
        self.KarliIslemOrani = (1.0 * self.KazandiranIslemSayisi) / (1.0 * self.IslemSayisi) * 100.0
        self.MinBakiyeFiyatYuzde = (self.MinBakiyeFiyat - self.IlkBakiyeFiyat) * 100.0 / self.IlkBakiyeFiyat
        self.MaxBakiyeFiyatYuzde = (self.MaxBakiyeFiyat - self.IlkBakiyeFiyat) * 100.0 / self.IlkBakiyeFiyat
        self.MinBakiyeFiyatNetYuzde = (self.MinBakiyeFiyatNet - self.IlkBakiyeFiyat) * 100.0 / self.IlkBakiyeFiyat
        self.MaxBakiyeFiyatNetYuzde = (self.MaxBakiyeFiyatNet - self.IlkBakiyeFiyat) * 100.0 / self.IlkBakiyeFiyat
        self.KomisyonFiyatYuzde = self.GetiriFiyatYuzde - self.GetiriFiyatYuzdeNet
        self.GetiriKzSistemYuzde = 0.0
        self.GetiriKzNetSistemYuzde = 0.0
        self.getiri_istatiskleri_hesapla(Sistem)
        self.assign_to_map(Sistem)
        return result

    def read_values(self, Sistem):
        self.SistemId = 0
        self.SistemName = Sistem.Name
        self.GrafikSembol = Sistem.Sembol
        self.GrafikPeriyot = Sistem.Periyot
        self.LastExecutionId = 0
        self.LastExecutionTime = self.Trader.LastExecutionTime
        self.LastExecutionTimeStart = self.Trader.LastExecutionTimeStart
        self.LastExecutionTimeStop = self.Trader.LastExecutionTimeStop
        self.ExecutionTimeInMSec = self.Trader.ExecutionTimeInMSec
        self.LastResetTime = self.Trader.LastResetTime
        self.LastStatisticsCalculationTime = self.Trader.LastStatisticsCalculationTime
        self.IslemSayisi = self.Trader.Status.IslemSayisi
        self.KazandiranIslemSayisi = self.Trader.Status.KazandiranIslemSayisi
        self.KaybettirenIslemSayisi = self.Trader.Status.KaybettirenIslemSayisi
        self.NotrIslemSayisi = self.Trader.Status.NotrIslemSayisi
        self.KazandiranAlisSayisi = self.Trader.Status.KazandiranAlisSayisi
        self.KaybettirenAlisSayisi = self.Trader.Status.KaybettirenAlisSayisi
        self.NotrAlisSayisi = self.Trader.Status.NotrAlisSayisi
        self.KazandiranSatisSayisi = self.Trader.Status.KazandiranSatisSayisi
        self.KaybettirenSatisSayisi = self.Trader.Status.KaybettirenSatisSayisi
        self.NotrSatisSayisi = self.Trader.Status.NotrSatisSayisi
        self.AlisSayisi = self.Trader.Status.AlisSayisi
        self.SatisSayisi = self.Trader.Status.SatisSayisi
        self.FlatSayisi = self.Trader.Status.FlatSayisi
        self.PassSayisi = self.Trader.Status.PassSayisi
        self.KarAlSayisi = self.Trader.Status.KarAlSayisi
        self.ZararKesSayisi = self.Trader.Status.ZararKesSayisi
        self.AlKomutSayisi = self.Trader.Status.AlKomutSayisi
        self.SatKomutSayisi = self.Trader.Status.SatKomutSayisi
        self.PasGecKomutSayisi = self.Trader.Status.PasGecKomutSayisi
        self.KarAlKomutSayisi = self.Trader.Status.KarAlKomutSayisi
        self.ZararKesKomutSayisi = self.Trader.Status.ZararKesKomutSayisi
        self.FlatOlKomutSayisi = self.Trader.Status.FlatOlKomutSayisi
        self.KardaBarSayisi = self.Trader.Status.KardaBarSayisi
        self.ZarardaBarSayisi = self.Trader.Status.ZarardaBarSayisi
        self.KarZararFiyat = self.Trader.Status.KarZararFiyat
        self.KarZararPuan = self.Trader.Status.KarZararPuan
        self.KarZararFiyatYuzde = self.Trader.Status.KarZararFiyatYuzde
        self.KomisyonIslemSayisi = self.Trader.Status.KomisyonIslemSayisi
        self.KomisyonVarlikAdedSayisi = self.Trader.Status.KomisyonVarlikAdedSayisi
        self.KomisyonCarpan = self.Trader.Status.KomisyonCarpan
        self.KomisyonFiyat = self.Trader.Status.KomisyonFiyat
        self.KomisyonuDahilEt = self.Trader.Flags.KomisyonuDahilEt
        self.HisseSayisi = self.Trader.Status.HisseSayisi
        self.KontratSayisi = self.Trader.Status.KontratSayisi
        self.VarlikAdedCarpani = self.Trader.Status.VarlikAdedCarpani
        self.VarlikAdedSayisi = self.Trader.Status.VarlikAdedSayisi
        self.KaymaMiktari = self.Trader.Status.KaymaMiktari
        self.KaymayiDahilEt = self.Trader.Flags.KaymayiDahilEt
        self.IlkBakiyeFiyat = self.Trader.Status.IlkBakiyeFiyat
        self.IlkBakiyePuan = self.Trader.Status.IlkBakiyePuan
        self.BakiyeFiyat = self.Trader.Status.BakiyeFiyat
        self.BakiyePuan = self.Trader.Status.BakiyePuan
        self.GetiriFiyat = self.Trader.Status.GetiriFiyat
        self.GetiriPuan = self.Trader.Status.GetiriPuan
        self.GetiriFiyatYuzde = self.Trader.Status.GetiriFiyatYuzde
        self.GetiriPuanYuzde = self.Trader.Status.GetiriPuanYuzde
        self.BakiyeFiyatNet = self.Trader.Status.BakiyeFiyatNet
        self.BakiyePuanNet = self.Trader.Status.BakiyePuanNet
        self.GetiriFiyatNet = self.Trader.Status.GetiriFiyatNet
        self.GetiriPuanNet = self.Trader.Status.GetiriPuanNet
        self.GetiriFiyatYuzdeNet = self.Trader.Status.GetiriFiyatYuzdeNet
        self.GetiriPuanYuzdeNet = self.Trader.Status.GetiriPuanYuzdeNet
        self.GetiriKz = self.Trader.Status.GetiriKz
        self.GetiriKzNet = self.Trader.Status.GetiriKzNet
        self.GetiriKzSistem = self.Trader.Status.GetiriKzSistem
        self.GetiriKzNetSistem = self.Trader.Status.GetiriKzNetSistem
        self.GetiriFiyatTipi = self.Trader.Status.GetiriFiyatTipi
        self.NetKarPuan = self.Trader.Status.NetKarPuan
        self.ToplamKarPuan = self.Trader.Status.ToplamKarPuan
        self.ToplamZararPuan = self.Trader.Status.ToplamZararPuan
        self.NetKarFiyat = self.Trader.Status.NetKarFiyat
        self.ToplamKarFiyat = self.Trader.Status.ToplamKarFiyat
        self.ToplamZararFiyat = self.Trader.Status.ToplamZararFiyat
        self.ToplamZararPuan = self.ToplamZararPuan if self.ToplamZararPuan != 0 else 1e-12
        self.Sinyal = self.Trader.Signals.Sinyal
        self.SonYon = self.Trader.Signals.SonYon
        self.PrevYon = self.Trader.Signals.PrevYon
        self.SonFiyat = self.Trader.Signals.SonFiyat
        self.SonAFiyat = self.Trader.Signals.SonAFiyat
        self.SonSFiyat = self.Trader.Signals.SonSFiyat
        self.SonFFiyat = self.Trader.Signals.SonFFiyat
        self.SonPFiyat = self.Trader.Signals.SonPFiyat
        self.PrevFiyat = self.Trader.Signals.PrevFiyat
        self.PrevAFiyat = self.Trader.Signals.PrevAFiyat
        self.PrevSFiyat = self.Trader.Signals.PrevSFiyat
        self.PrevFFiyat = self.Trader.Signals.PrevFFiyat
        self.PrevPFiyat = self.Trader.Signals.PrevPFiyat
        self.SonBarNo = self.Trader.Signals.SonBarNo
        self.SonABarNo = self.Trader.Signals.SonABarNo
        self.SonSBarNo = self.Trader.Signals.SonSBarNo
        self.SonFBarNo = self.Trader.Signals.SonFBarNo
        self.SonPBarNo = self.Trader.Signals.SonPBarNo
        self.PrevBarNo = self.Trader.Signals.PrevBarNo
        self.PrevABarNo = self.Trader.Signals.PrevABarNo
        self.PrevSBarNo = self.Trader.Signals.PrevSBarNo
        self.PrevFBarNo = self.Trader.Signals.PrevFBarNo
        self.PrevPBarNo = self.Trader.Signals.PrevPBarNo
        self.EmirKomut = self.Trader.Signals.EmirKomut
        self.EmirStatus = self.Trader.Signals.EmirStatus

    def assign_to_map(self, Sistem):
        self.IstatistiklerNew.clear()
        self.IstatistiklerNew["GrafikSembol"] = self.GrafikSembol
        self.IstatistiklerNew["GrafikPeriyot"] = self.GrafikPeriyot
        self.IstatistiklerNew["SistemId"] = str(self.SistemId)
        self.IstatistiklerNew["SistemName"] = self.SistemName
        self.IstatistiklerNew["LastExecutionTime"] = self.LastExecutionTime
        self.IstatistiklerNew["LastExecutionTimeStart"] = self.LastExecutionTimeStart
        self.IstatistiklerNew["LastExecutionTimeStop"] = self.LastExecutionTimeStop
        self.IstatistiklerNew["ExecutionTimeInMSec"] = str(self.ExecutionTimeInMSec)
        self.IstatistiklerNew["LastExecutionId"] = self.LastExecutionId
        self.IstatistiklerNew["LastResetTime"] = self.LastResetTime
        self.IstatistiklerNew["LastStatisticsCalculationTime"] = self.LastStatisticsCalculationTime
        self.IstatistiklerNew["ToplamGecenSureAy"] = f'{self.ToplamGecenSureAy:.1f}'
        self.IstatistiklerNew["ToplamGecenSureGun"] = str(self.ToplamGecenSureGun)
        self.IstatistiklerNew["ToplamGecenSureSaat"] = str(self.ToplamGecenSureSaat)
        self.IstatistiklerNew["ToplamGecenSureDakika"] = str(self.ToplamGecenSureDakika)
        self.IstatistiklerNew["ToplamBarSayisi"] = str(self.ToplamBarSayisi)
        self.IstatistiklerNew["SecilenBarNumarasi"] = str(self.SecilenBarNumarasi)
        self.IstatistiklerNew["SecilenBarTarihi"] = self.SecilenBarTarihi.strftime("%d.%m.%Y")
        self.IstatistiklerNew["SecilenBarSaati"] = str(self.SecilenBarTarihi.time())
        self.IstatistiklerNew["IlkBarTarihi"] = self.IlkBarTarihi.strftime("%d.%m.%Y")
        self.IstatistiklerNew["IlkBarSaati"] = str(self.IlkBarTarihi.time())
        self.IstatistiklerNew["SonBarTarihi"] = self.SonBarTarihi.strftime("%d.%m.%Y")
        self.IstatistiklerNew["SonBarSaati"] = str(self.SonBarTarihi.time())
        self.IstatistiklerNew["IlkBarIndex"] = str(self.IlkBarIndex)
        self.IstatistiklerNew["SonBarIndex"] = str(self.SonBarIndex)
        self.IstatistiklerNew["SonBarAcilisFiyati"] = str(self.SonBarAcilisFiyati)
        self.IstatistiklerNew["SonBarYuksekFiyati"] = str(self.SonBarYuksekFiyati)
        self.IstatistiklerNew["SonBarDusukFiyati"] = str(self.SonBarDusukFiyati)
        self.IstatistiklerNew["SonBarKapanisFiyati"] = str(self.SonBarKapanisFiyati)
        self.IstatistiklerNew["IlkBakiyeFiyat"] = str(self.IlkBakiyeFiyat)
        self.IstatistiklerNew["IlkBakiyePuan"] = str(self.IlkBakiyePuan)
        self.IstatistiklerNew["BakiyeFiyat"] = str(self.BakiyeFiyat)
        self.IstatistiklerNew["BakiyePuan"] = str(self.BakiyePuan)
        self.IstatistiklerNew["GetiriFiyat"] = str(self.GetiriFiyat)
        self.IstatistiklerNew["GetiriPuan"] = str(self.GetiriPuan)
        self.IstatistiklerNew["GetiriFiyatYuzde"] = str(self.GetiriFiyatYuzde)
        self.IstatistiklerNew["GetiriPuanYuzde"] = str(self.GetiriPuanYuzde)
        self.IstatistiklerNew["BakiyeFiyatNet"] = str(self.BakiyeFiyatNet)
        self.IstatistiklerNew["BakiyePuanNet"] = str(self.BakiyePuanNet)
        self.IstatistiklerNew["GetiriFiyatNet"] = str(self.GetiriFiyatNet)
        self.IstatistiklerNew["GetiriPuanNet"] = str(self.GetiriPuanNet)
        self.IstatistiklerNew["GetiriFiyatYuzdeNet"] = str(self.GetiriFiyatYuzdeNet)
        self.IstatistiklerNew["GetiriPuanYuzdeNet"] = str(self.GetiriPuanYuzdeNet)
        self.IstatistiklerNew["GetiriKz"] = str(self.GetiriKz)
        self.IstatistiklerNew["GetiriKzNet"] = str(self.GetiriKzNet)
        self.IstatistiklerNew["MinBakiyeFiyat"] = str(self.MinBakiyeFiyat)
        self.IstatistiklerNew["MaxBakiyeFiyat"] = str(self.MaxBakiyeFiyat)
        self.IstatistiklerNew["MinBakiyePuan"] = str(self.MinBakiyePuan)
        self.IstatistiklerNew["MaxBakiyePuan"] = str(self.MaxBakiyePuan)
        self.IstatistiklerNew["MinBakiyeFiyatYuzde"] = str(self.MinBakiyeFiyatYuzde)
        self.IstatistiklerNew["MaxBakiyeFiyatYuzde"] = str(self.MaxBakiyeFiyatYuzde)
        self.IstatistiklerNew["MinBakiyeFiyatIndex"] = str(self.MinBakiyeFiyatIndex)
        self.IstatistiklerNew["MaxBakiyeFiyatIndex"] = str(self.MaxBakiyeFiyatIndex)
        self.IstatistiklerNew["MinBakiyePuanIndex"] = str(self.MinBakiyePuanIndex)
        self.IstatistiklerNew["MaxBakiyePuanIndex"] = str(self.MaxBakiyePuanIndex)
        self.IstatistiklerNew["MinBakiyeFiyatNet"] = str(self.MinBakiyeFiyatNet)
        self.IstatistiklerNew["MaxBakiyeFiyatNet"] = str(self.MaxBakiyeFiyatNet)
        self.IstatistiklerNew["MinBakiyeFiyatNetIndex"] = str(self.MinBakiyeFiyatNetIndex)
        self.IstatistiklerNew["MaxBakiyeFiyatNetIndex"] = str(self.MaxBakiyeFiyatNetIndex)
        self.IstatistiklerNew["MinBakiyeFiyatNetYuzde"] = str(self.MinBakiyeFiyatNetYuzde)
        self.IstatistiklerNew["MaxBakiyeFiyatNetYuzde"] = str(self.MaxBakiyeFiyatNetYuzde)
        self.IstatistiklerNew["GetiriKzSistem"] = f'{self.GetiriKzSistem:.2f}'
        self.IstatistiklerNew["GetiriKzSistemYuzde"] = f'{self.GetiriKzSistemYuzde:.2f}'
        self.IstatistiklerNew["GetiriKzNetSistem"] = f'{self.GetiriKzNetSistem:.2f}'
        self.IstatistiklerNew["GetiriKzNetSistemYuzde"] = f'{self.GetiriKzNetSistemYuzde:.2f}'
        self.IstatistiklerNew["IslemSayisi"] = str(self.IslemSayisi)
        self.IstatistiklerNew["AlisSayisi"] = str(self.AlisSayisi)
        self.IstatistiklerNew["SatisSayisi"] = str(self.SatisSayisi)
        self.IstatistiklerNew["FlatSayisi"] = str(self.FlatSayisi)
        self.IstatistiklerNew["PassSayisi"] = str(self.PassSayisi)
        self.IstatistiklerNew["KarAlSayisi"] = str(self.KarAlSayisi)
        self.IstatistiklerNew["ZararKesSayisi"] = str(self.ZararKesSayisi)
        self.IstatistiklerNew["KazandiranIslemSayisi"] = str(self.KazandiranIslemSayisi)
        self.IstatistiklerNew["KaybettirenIslemSayisi"] = str(self.KaybettirenIslemSayisi)
        self.IstatistiklerNew["NotrIslemSayisi"] = str(self.NotrIslemSayisi)
        self.IstatistiklerNew["KazandiranAlisSayisi"] = str(self.KazandiranAlisSayisi)
        self.IstatistiklerNew["KaybettirenAlisSayisi"] = str(self.KaybettirenAlisSayisi)
        self.IstatistiklerNew["NotrAlisSayisi"] = str(self.NotrAlisSayisi)
        self.IstatistiklerNew["KazandiranSatisSayisi"] = str(self.KazandiranSatisSayisi)
        self.IstatistiklerNew["KaybettirenSatisSayisi"] = str(self.KaybettirenSatisSayisi)
        self.IstatistiklerNew["NotrSatisSayisi"] = str(self.NotrSatisSayisi)
        self.IstatistiklerNew["AlKomutSayisi"] = str(self.AlKomutSayisi)
        self.IstatistiklerNew["SatKomutSayisi"] = str(self.SatKomutSayisi)
        self.IstatistiklerNew["PasGecKomutSayisi"] = str(self.PasGecKomutSayisi)
        self.IstatistiklerNew["KarAlKomutSayisi"] = str(self.KarAlKomutSayisi)
        self.IstatistiklerNew["ZararKesKomutSayisi"] = str(self.ZararKesKomutSayisi)
        self.IstatistiklerNew["FlatOlKomutSayisi"] = str(self.FlatOlKomutSayisi)
        self.IstatistiklerNew["KomisyonIslemSayisi"] = str(self.KomisyonIslemSayisi)
        self.IstatistiklerNew["KomisyonVarlikAdedSayisi"] = str(self.KomisyonVarlikAdedSayisi)
        self.IstatistiklerNew["KomisyonCarpan"] = str(self.KomisyonCarpan)
        self.IstatistiklerNew["KomisyonFiyat"] = str(self.KomisyonFiyat)
        self.IstatistiklerNew["KomisyonFiyatYuzde"] = str(self.KomisyonFiyatYuzde)
        self.IstatistiklerNew["KomisyonuDahilEt"] = str(self.KomisyonuDahilEt)
        self.IstatistiklerNew["KarZararFiyat"] = str(self.KarZararFiyat)
        self.IstatistiklerNew["KarZararFiyatYuzde"] = str(self.KarZararFiyatYuzde)
        self.IstatistiklerNew["KarZararPuan"] = str(self.KarZararPuan)
        self.IstatistiklerNew["ToplamKarFiyat"] = str(self.ToplamKarFiyat)
        self.IstatistiklerNew["ToplamZararFiyat"] = str(self.ToplamZararFiyat)
        self.IstatistiklerNew["NetKarFiyat"] = str(self.NetKarFiyat)
        self.IstatistiklerNew["ToplamKarPuan"] = str(self.ToplamKarPuan)
        self.IstatistiklerNew["ToplamZararPuan"] = str(self.ToplamZararPuan)
        self.IstatistiklerNew["NetKarPuan"] = str(self.NetKarPuan)
        self.IstatistiklerNew["MaxKarFiyat"] = str(self.MaxKarFiyat)
        self.IstatistiklerNew["MaxZararFiyat"] = str(self.MaxZararFiyat)
        self.IstatistiklerNew["MaxKarPuan"] = str(self.MaxKarPuan)
        self.IstatistiklerNew["MaxZararPuan"] = str(self.MaxZararPuan)
        self.IstatistiklerNew["MaxZararFiyatIndex"] = str(self.MaxZararFiyatIndex)
        self.IstatistiklerNew["MaxKarFiyatIndex"] = str(self.MaxKarFiyatIndex)
        self.IstatistiklerNew["MaxZararPuanIndex"] = str(self.MaxZararPuanIndex)
        self.IstatistiklerNew["MaxKarPuanIndex"] = str(self.MaxKarPuanIndex)
        self.IstatistiklerNew["KardaBarSayisi"] = str(self.KardaBarSayisi)
        self.IstatistiklerNew["ZarardaBarSayisi"] = str(self.ZarardaBarSayisi)
        self.IstatistiklerNew["KarliIslemOrani"] = f'{self.KarliIslemOrani:.2f}'
        self.IstatistiklerNew["GetiriMaxDD"] = str(self.GetiriMaxDD)
        self.IstatistiklerNew["GetiriMaxDDTarih"] = self.GetiriMaxDDTarih.strftime("%d.%m.%Y")
        self.IstatistiklerNew["GetiriMaxDDSaat"] = str(self.GetiriMaxDDTarih.time())
        self.IstatistiklerNew["GetiriMaxKayip"] = str(self.GetiriMaxKayip)
        self.IstatistiklerNew["ProfitFactor"] = f'{self.ProfitFactor:.2f}'
        self.IstatistiklerNew["ProfitFactorSistem"] = f'{self.ProfitFactorSistem:.2f}'
        self.IstatistiklerNew["OrtAylikIslemSayisi"] = f'{self.OrtAylikIslemSayisi:.2f}'
        self.IstatistiklerNew["OrtHaftalikIslemSayisi"] = f'{self.OrtHaftalikIslemSayisi:.2f}'
        self.IstatistiklerNew["OrtGunlukIslemSayisi"] = f'{self.OrtGunlukIslemSayisi:.2f}'
        self.IstatistiklerNew["OrtSaatlikIslemSayisi"] = f'{self.OrtSaatlikIslemSayisi:.2f}'
        self.IstatistiklerNew["Sinyal"] = str(self.Sinyal)
        self.IstatistiklerNew["SonYon"] = str(self.SonYon)
        self.IstatistiklerNew["PrevYon"] = str(self.PrevYon)
        self.IstatistiklerNew["SonFiyat"] = str(self.SonFiyat)
        self.IstatistiklerNew["SonAFiyat"] = str(self.SonAFiyat)
        self.IstatistiklerNew["SonSFiyat"] = str(self.SonSFiyat)
        self.IstatistiklerNew["SonFFiyat"] = str(self.SonFFiyat)
        self.IstatistiklerNew["SonPFiyat"] = str(self.SonPFiyat)
        self.IstatistiklerNew["PrevFiyat"] = str(self.PrevFiyat)
        self.IstatistiklerNew["PrevAFiyat"] = str(self.PrevAFiyat)
        self.IstatistiklerNew["PrevSFiyat"] = str(self.PrevSFiyat)
        self.IstatistiklerNew["PrevFFiyat"] = str(self.PrevFFiyat)
        self.IstatistiklerNew["PrevPFiyat"] = str(self.PrevPFiyat)
        self.IstatistiklerNew["SonBarNo"] = str(self.SonBarNo)
        self.IstatistiklerNew["SonABarNo"] = str(self.SonABarNo)
        self.IstatistiklerNew["SonSBarNo"] = str(self.SonSBarNo)
        self.IstatistiklerNew["SonFBarNo"] = str(self.SonFBarNo)
        self.IstatistiklerNew["SonPBarNo"] = str(self.SonPBarNo)
        self.IstatistiklerNew["PrevBarNo"] = str(self.PrevBarNo)
        self.IstatistiklerNew["PrevABarNo"] = str(self.PrevABarNo)
        self.IstatistiklerNew["PrevSBarNo"] = str(self.PrevSBarNo)
        self.IstatistiklerNew["PrevFBarNo"] = str(self.PrevFBarNo)
        self.IstatistiklerNew["PrevPBarNo"] = str(self.PrevPBarNo)
        self.IstatistiklerNew["EmirKomut"] = str(self.EmirKomut)
        self.IstatistiklerNew["EmirStatus"] = str(self.EmirStatus)
        self.IstatistiklerNew["HisseSayisi"] = str(self.HisseSayisi)
        self.IstatistiklerNew["KontratSayisi"] = str(self.KontratSayisi)
        self.IstatistiklerNew["VarlikAdedCarpani"] = str(self.VarlikAdedCarpani)
        self.IstatistiklerNew["VarlikAdedSayisi"] = str(self.VarlikAdedSayisi)
        self.IstatistiklerNew["KaymaMiktari"] = str(self.KaymaMiktari)
        self.IstatistiklerNew["KaymayiDahilEt"] = str(self.KaymayiDahilEt)
        self.IstatistiklerNew["GetiriFiyatBuAy"] = f'{self.GetiriFiyatBuAy:.2f}'
        self.IstatistiklerNew["GetiriFiyatAy1"] = f'{self.GetiriFiyatAy1:.2f}'
        self.IstatistiklerNew["GetiriFiyatAy2"] = f'{self.GetiriFiyatAy2:.2f}'
        self.IstatistiklerNew["GetiriFiyatAy3"] = f'{self.GetiriFiyatAy3:.2f}'
        self.IstatistiklerNew["GetiriFiyatAy4"] = f'{self.GetiriFiyatAy4:.2f}'
        self.IstatistiklerNew["GetiriFiyatAy5"] = f'{self.GetiriFiyatAy5:.2f}'
        self.IstatistiklerNew["GetiriFiyatBuHafta"] = f'{self.GetiriFiyatBuHafta:.2f}'
        self.IstatistiklerNew["GetiriFiyatHafta1"] = f'{self.GetiriFiyatHafta1:.2f}'
        self.IstatistiklerNew["GetiriFiyatHafta2"] = f'{self.GetiriFiyatHafta2:.2f}'
        self.IstatistiklerNew["GetiriFiyatHafta3"] = f'{self.GetiriFiyatHafta3:.2f}'
        self.IstatistiklerNew["GetiriFiyatHafta4"] = f'{self.GetiriFiyatHafta4:.2f}'
        self.IstatistiklerNew["GetiriFiyatHafta5"] = f'{self.GetiriFiyatHafta5:.2f}'
        self.IstatistiklerNew["GetiriFiyatBuGun"] = f'{self.GetiriFiyatBuGun:.2f}'
        self.IstatistiklerNew["GetiriFiyatGun1"] = f'{self.GetiriFiyatGun1:.2f}'
        self.IstatistiklerNew["GetiriFiyatGun2"] = f'{self.GetiriFiyatGun2:.2f}'
        self.IstatistiklerNew["GetiriFiyatGun3"] = f'{self.GetiriFiyatGun3:.2f}'
        self.IstatistiklerNew["GetiriFiyatGun4"] = f'{self.GetiriFiyatGun4:.2f}'
        self.IstatistiklerNew["GetiriFiyatGun5"] = f'{self.GetiriFiyatGun5:.2f}'
        self.IstatistiklerNew["GetiriFiyatBuSaat"] = f'{self.GetiriFiyatBuSaat:.2f}'
        self.IstatistiklerNew["GetiriFiyatSaat1"] = f'{self.GetiriFiyatSaat1:.2f}'
        self.IstatistiklerNew["GetiriFiyatSaat2"] = f'{self.GetiriFiyatSaat2:.2f}'
        self.IstatistiklerNew["GetiriFiyatSaat3"] = f'{self.GetiriFiyatSaat3:.2f}'
        self.IstatistiklerNew["GetiriFiyatSaat4"] = f'{self.GetiriFiyatSaat4:.2f}'
        self.IstatistiklerNew["GetiriFiyatSaat5"] = f'{self.GetiriFiyatSaat5:.2f}'
        self.IstatistiklerNew["GetiriPuanBuAy"] = f'{self.GetiriPuanBuAy:.2f}'
        self.IstatistiklerNew["GetiriPuanAy1"] = f'{self.GetiriPuanAy1:.2f}'
        self.IstatistiklerNew["GetiriPuanAy2"] = f'{self.GetiriPuanAy2:.2f}'
        self.IstatistiklerNew["GetiriPuanAy3"] = f'{self.GetiriPuanAy3:.2f}'
        self.IstatistiklerNew["GetiriPuanAy4"] = f'{self.GetiriPuanAy4:.2f}'
        self.IstatistiklerNew["GetiriPuanAy5"] = f'{self.GetiriPuanAy5:.2f}'
        self.IstatistiklerNew["GetiriPuanBuHafta"] = f'{self.GetiriPuanBuHafta:.2f}'
        self.IstatistiklerNew["GetiriPuanHafta1"] = f'{self.GetiriPuanHafta1:.2f}'
        self.IstatistiklerNew["GetiriPuanHafta2"] = f'{self.GetiriPuanHafta2:.2f}'
        self.IstatistiklerNew["GetiriPuanHafta3"] = f'{self.GetiriPuanHafta3:.2f}'
        self.IstatistiklerNew["GetiriPuanHafta4"] = f'{self.GetiriPuanHafta4:.2f}'
        self.IstatistiklerNew["GetiriPuanHafta5"] = f'{self.GetiriPuanHafta5:.2f}'
        self.IstatistiklerNew["GetiriPuanBuGun"] = f'{self.GetiriPuanBuGun:.2f}'
        self.IstatistiklerNew["GetiriPuanGun1"] = f'{self.GetiriPuanGun1:.2f}'
        self.IstatistiklerNew["GetiriPuanGun2"] = f'{self.GetiriPuanGun2:.2f}'
        self.IstatistiklerNew["GetiriPuanGun3"] = f'{self.GetiriPuanGun3:.2f}'
        self.IstatistiklerNew["GetiriPuanGun4"] = f'{self.GetiriPuanGun4:.2f}'
        self.IstatistiklerNew["GetiriPuanGun5"] = f'{self.GetiriPuanGun5:.2f}'
        self.IstatistiklerNew["GetiriPuanBuSaat"] = f'{self.GetiriPuanBuSaat:.2f}'
        self.IstatistiklerNew["GetiriPuanSaat1"] = f'{self.GetiriPuanSaat1:.2f}'
        self.IstatistiklerNew["GetiriPuanSaat2"] = f'{self.GetiriPuanSaat2:.2f}'
        self.IstatistiklerNew["GetiriPuanSaat3"] = f'{self.GetiriPuanSaat3:.2f}'
        self.IstatistiklerNew["GetiriPuanSaat4"] = f'{self.GetiriPuanSaat4:.2f}'
        self.IstatistiklerNew["GetiriPuanSaat5"] = f'{self.GetiriPuanSaat5:.2f}'

    def panel(self, PanelNo):
        self.PanelNo = PanelNo
        return self

    def column(self, ColNum):
        self.ColNum = 160 + 80 * ColNum
        if self.PanelNo != 1:
            self.ColNum = self.ColNum - 150
        return self

    def row(self, RowNum):
        self.RowNum = 50 + 25 * RowNum
        if self.PanelNo != 1:
            self.RowNum = self.RowNum - 25
        return self

    def text_message(self, Sistem, Message, MessageColor):
        Sistem.ZeminYazisiEkle(Message, self.PanelNo, self.ColNum, self.RowNum, MessageColor, "Tahoma", self.FontSize)
        return self

    def get_formatted_string(self, Alias, Tab, Key):
        return f'{Alias}{Tab}: {self.IstatistiklerNew[Key]}'

    def ekran_koordianatlarini_goster(self, Sistem, PanelNo=1):
        for col in range(20):
            for row in range(10):
                s = f'{col}, {row}'
                self.panel(PanelNo).column(col).row(row)
                s = f'{self.ColNum}, {self.RowNum}'
                self.text_message(Sistem, s, (255, 215, 0))  # Gold

    def istatistikleri_ekrana_yaz(self, Sistem, PanelNo=1):
        str_list = []
        tab0 = "  "
        tab1 = "\t"
        tab2 = "\t\t"
        tab3 = "\t\t\t"
        seperator = "-" * 40

        str_list.append(self.get_formatted_string("Profit Factor", tab2, "ProfitFactor"))
        str_list.append(self.get_formatted_string("Profit Factor Sistem", tab1, "ProfitFactorSistem"))
        str_list.append(self.get_formatted_string("Karli Islem Orani", tab1, "KarliIslemOrani"))
        str_list.append(self.get_formatted_string("GetiriKz Sistem", tab2, "GetiriKzSistem"))
        str_list.append(self.get_formatted_string("GetiriKz Sistem  (%)", tab1, "GetiriKzSistemYuzde"))
        str_list.append(seperator)
        str_list.append(self.get_formatted_string("Son Çalışma Zamani", tab1, "LastExecutionTime"))
        str_list.append(self.get_formatted_string("Çalışma Süresi (ms)", tab1, "ExecutionTimeInMSec"))
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(0).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        str_list.clear()
        str_list.append(self.get_formatted_string("IlkBakiyeFiyat", tab2, "IlkBakiyeFiyat"))
        str_list.append(self.get_formatted_string("BakiyeFiyat", tab2, "BakiyeFiyat"))
        str_list.append(seperator)
        str_list.append(self.get_formatted_string("GetiriFiyat", tab2, "GetiriFiyat"))
        str_list.append(self.get_formatted_string("GetiriFiyatYuzde (%)", tab1, "GetiriFiyatYuzde"))
        str_list.append(self.get_formatted_string("GetiriFiyatTipi", tab2, "GetiriFiyatTipi"))
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(3).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        str_list.clear()
        str_list.append(self.get_formatted_string("Komisyon", tab2, "KomisyonFiyat"))
        str_list.append(self.get_formatted_string("KomisyonYuzde (%)", tab1, "KomisyonFiyatYuzde"))
        str_list.append(seperator)
        str_list.append(self.get_formatted_string("GetiriFiyatNet", tab2, "GetiriFiyatNet"))
        str_list.append(self.get_formatted_string("GetiriFiyatYuzdeNet (%)", tab0, "GetiriFiyatYuzdeNet"))
        str_list.append(seperator)
        str_list.append(self.get_formatted_string("HisseSayisi", tab2, "HisseSayisi"))
        str_list.append(self.get_formatted_string("KontratSayisi", tab2, "KontratSayisi"))
        str_list.append(self.get_formatted_string("KomisyonVarlikSayisi", tab1, "KomisyonVarlikAdedSayisi"))
        str_list.append(self.get_formatted_string("VarlikAdedCarpani", tab1, "VarlikAdedCarpani"))
        str_list.append(self.get_formatted_string("VarlikAdedSayisi", tab1, "VarlikAdedSayisi"))
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(6).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        str_list.clear()
        str_list.append(self.get_formatted_string("KomisyonIslemSayisi", tab1, "KomisyonIslemSayisi"))
        str_list.append(self.get_formatted_string("ToplamIslemSayisi", tab1, "IslemSayisi"))
        str_list.append(self.get_formatted_string("AlisSayisi", tab2, "AlisSayisi"))
        str_list.append(self.get_formatted_string("SatisSayisi", tab2, "SatisSayisi"))
        str_list.append(self.get_formatted_string("FlatSayisi", tab2, "FlatSayisi"))
        str_list.append(self.get_formatted_string("PassSayisi", tab2, "PassSayisi"))
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(9).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        str_list.clear()
        str_list.append(self.get_formatted_string("OrtAylikİslemSayisi", tab1, "OrtAylikIslemSayisi"))
        str_list.append(self.get_formatted_string("OrtGunlukİslemSayisi", tab1, "OrtGunlukIslemSayisi"))
        str_list.append(seperator)
        str_list.append(self.get_formatted_string("KazandiranIslemSayisi", tab1, "KazandiranIslemSayisi"))
        str_list.append(self.get_formatted_string("KaybettirenIslemSayisi", tab0, "KaybettirenIslemSayisi"))
        str_list.append(self.get_formatted_string("NotrIslemSayisi", tab1, "NotrIslemSayisi"))
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(12).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        str_list.clear()
        str_list.append(self.get_formatted_string("ToplamKarFiyat", tab1, "ToplamKarFiyat"))
        str_list.append(self.get_formatted_string("ToplamZararFiyat", tab1, "ToplamZararFiyat"))
        str_list.append(self.get_formatted_string("NetKarFiyat", tab2, "NetKarFiyat"))
        str_list.append(seperator)
        str_list.append(self.get_formatted_string("MaxKarFiyat", tab2, "MaxKarFiyat"))
        str_list.append(self.get_formatted_string("MaxZararFiyat", tab2, "MaxZararFiyat"))
        str_list.append(self.get_formatted_string("MaxKarFiyatIndex", tab1, "MaxKarFiyatIndex"))
        str_list.append(self.get_formatted_string("MaxZararFiyatIndex", tab1, "MaxZararFiyatIndex"))
        str_list.append(seperator)
        str_list.append(self.get_formatted_string("MinBakiyeFiyat", tab2, "MinBakiyeFiyat"))
        str_list.append(self.get_formatted_string("MaxBakiyeFiyat", tab1, "MaxBakiyeFiyat"))
        str_list.append(self.get_formatted_string("MinBakiyeFiyatIndex", tab1, "MinBakiyeFiyatIndex"))
        str_list.append(self.get_formatted_string("MaxBakiyeFiyatIndex", tab1, "MaxBakiyeFiyatIndex"))
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(15).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        str_list.clear()
        str_list.append(self.get_formatted_string("GetiriMaxDD (Puan)", tab1, "GetiriMaxDD"))
        str_list.append(self.get_formatted_string("GetiriMaxDDTarih", tab1, "GetiriMaxDDTarih"))
        str_list.append(self.get_formatted_string("GetiriMaxDDSaat", tab1, "GetiriMaxDDSaat"))
        str_list.append(self.get_formatted_string("GetiriMaxKayip", tab2, "GetiriMaxKayip"))
        str_list.append(seperator)
        str_list.append(self.get_formatted_string("Toplam Bar Sayisi", tab1, "ToplamBarSayisi"))
        str_list.append(self.get_formatted_string("IlkBar Tarihi", tab2, "IlkBarTarihi"))
        str_list.append(self.get_formatted_string("IlkBar Saati", tab2, "IlkBarSaati"))
        str_list.append(seperator)
        str_list.append(self.get_formatted_string("SonBar Tarihi", tab2, "SonBarTarihi"))
        str_list.append(self.get_formatted_string("SonBar Saati", tab2, "SonBarSaati"))
        str_list.append(self.get_formatted_string("Toplam Sure Ay", tab1, "ToplamGecenSureAy"))
        str_list.append(self.get_formatted_string("Toplam Sure Gun", tab1, "ToplamGecenSureGun"))
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(18).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

    def istatistikleri_dosyaya_yaz(self, Sistem, FileName):
        delimiter = ";"
        myFileUtils = CFileUtils()
        logFileFullName = FileName.strip()
        myFileUtils.reset(Sistem).enable_logging(Sistem).open_log_file(Sistem, logFileFullName, False, False)
        logMessage = f'Kayit Zamani {delimiter} {datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")}'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        for key, value in self.IstatistiklerNew.items():
            logMessage = f'{key.strip()} ; {value} ; \t'
            myFileUtils.write_to_log_file(Sistem, logMessage)
        myFileUtils.close_log_file(Sistem)

    def getiri_istatiskleri_hesapla(self, Sistem):
        V = Sistem.GrafikVerileri
        DateAyBarNoList = []
        DateHaftaBarNoList = []
        DateGunBarNoList = []
        DateSaatBarNoList = []
        timeUtils = CTimeUtils()
        for i in range(1, len(V)):
            yeni_saat = V[i - 1].Date.hour != V[i].Date.hour
            if yeni_saat:
                DateSaatBarNoList.append(i)
            yeni_gun = V[i - 1].Date.day != V[i].Date.day
            if yeni_gun:
                DateGunBarNoList.append(i)
            yeni_hafta = yeni_gun and timeUtils.get_week_number(Sistem, V[i - 1].Date) != timeUtils.get_week_number(Sistem, V[i].Date)
            if yeni_hafta:
                DateHaftaBarNoList.append(i)
            yeni_ay = V[i - 1].Date.month != V[i].Date.month
            if yeni_ay:
                DateAyBarNoList.append(i)

        KZList = Sistem.Liste(0)
        for i in range(len(V)):
            KZList[i] = Sistem.GetiriKZ[i]

        self.GetiriPuanBuAy = 0.0
        self.GetiriPuanAy1 = 0.0
        self.GetiriPuanAy2 = 0.0
        self.GetiriPuanAy3 = 0.0
        self.GetiriPuanAy4 = 0.0
        self.GetiriPuanAy5 = 0.0
        self.GetiriPuanBuHafta = 0.0
        self.GetiriPuanHafta1 = 0.0
        self.GetiriPuanHafta2 = 0.0
        self.GetiriPuanHafta3 = 0.0
        self.GetiriPuanHafta4 = 0.0
        self.GetiriPuanHafta5 = 0.0
        self.GetiriPuanBuGun = 0.0
        self.GetiriPuanGun1 = 0.0
        self.GetiriPuanGun2 = 0.0
        self.GetiriPuanGun3 = 0.0
        self.GetiriPuanGun4 = 0.0
        self.GetiriPuanGun5 = 0.0
        self.GetiriPuanBuSaat = 0.0
        self.GetiriPuanSaat1 = 0.0
        self.GetiriPuanSaat2 = 0.0
        self.GetiriPuanSaat3 = 0.0
        self.GetiriPuanSaat4 = 0.0
        self.GetiriPuanSaat5 = 0.0

        aykz1 = 0.0
        aykz2 = 0.0
        aykz3 = 0.0
        aykz4 = 0.0
        aykz5 = 0.0
        if len(DateAyBarNoList) > 5:
            aykz1 = KZList[-1] - 1 * KZList[DateAyBarNoList[-1]]
            aykz2 = KZList[-1] - 1 * KZList[DateAyBarNoList[-2]]
            aykz3 = KZList[-1] - 1 * KZList[DateAyBarNoList[-3]]
            aykz4 = KZList[-1] - 1 * KZList[DateAyBarNoList[-4]]
            aykz5 = KZList[-1] - 1 * KZList[DateAyBarNoList[-5]]

        haftakz1 = 0.0
        haftakz2 = 0.0
        haftakz3 = 0.0
        haftakz4 = 0.0
        haftakz5 = 0.0
        if len(DateHaftaBarNoList) > 5:
            haftakz1 = KZList[-1] - 1 * KZList[DateHaftaBarNoList[-1]]
            haftakz2 = KZList[-1] - 1 * KZList[DateHaftaBarNoList[-2]]
            haftakz3 = KZList[-1] - 1 * KZList[DateHaftaBarNoList[-3]]
            haftakz4 = KZList[-1] - 1 * KZList[DateHaftaBarNoList[-4]]
            haftakz5 = KZList[-1] - 1 * KZList[DateHaftaBarNoList[-5]]

        gunkz1 = 0.0
        gunkz2 = 0.0
        gunkz3 = 0.0
        gunkz4 = 0.0
        gunkz5 = 0.0
        if len(DateGunBarNoList) > 5:
            gunkz1 = KZList[-1] - 1 * KZList[DateGunBarNoList[-1]]
            gunkz2 = KZList[-1] - 1 * KZList[DateGunBarNoList[-2]]
            gunkz3 = KZList[-1] - 1 * KZList[DateGunBarNoList[-3]]
            gunkz4 = KZList[-1] - 1 * KZList[DateGunBarNoList[-4]]
            gunkz5 = KZList[-1] - 1 * KZList[DateGunBarNoList[-5]]

        saatkz1 = 0.0
        saatkz2 = 0.0
        saatkz3 = 0.0
        saatkz4 = 0.0
        saatkz5 = 0.0
        if len(DateSaatBarNoList) > 5:
            saatkz1 = KZList[-1] - 1 * KZList[DateSaatBarNoList[-2]]
            saatkz2 = KZList[-1] - 1 * KZList[DateSaatBarNoList[-3]]
            saatkz3 = KZList[-1] - 1 * KZList[DateSaatBarNoList[-4]]
            saatkz4 = KZList[-1] - 1 * KZList[DateSaatBarNoList[-5]]
            saatkz5 = KZList[-1] - 1 * KZList[DateSaatBarNoList[-6]]

        self.GetiriPuanBuAy = float(aykz1)
        self.GetiriPuanAy1 = float(aykz1)
        self.GetiriPuanAy2 = float(aykz2)
        self.GetiriPuanAy3 = float(aykz3)
        self.GetiriPuanAy4 = float(aykz4)
        self.GetiriPuanAy5 = float(aykz5)
        self.GetiriPuanBuHafta = float(haftakz1)
        self.GetiriPuanHafta1 = float(haftakz1)
        self.GetiriPuanHafta2 = float(haftakz2)
        self.GetiriPuanHafta3 = float(haftakz3)
        self.GetiriPuanHafta4 = float(haftakz4)
        self.GetiriPuanHafta5 = float(haftakz5)
        self.GetiriPuanBuGun = float(gunkz1)
        self.GetiriPuanGun1 = float(gunkz1)
        self.GetiriPuanGun2 = float(gunkz2)
        self.GetiriPuanGun3 = float(gunkz3)
        self.GetiriPuanGun4 = float(gunkz4)
        self.GetiriPuanGun5 = float(gunkz5)
        self.GetiriPuanBuSaat = float(saatkz1)
        self.GetiriPuanSaat1 = float(saatkz1)
        self.GetiriPuanSaat2 = float(saatkz2)
        self.GetiriPuanSaat3 = float(saatkz3)
        self.GetiriPuanSaat4 = float(saatkz4)
        self.GetiriPuanSaat5 = float(saatkz5)

        for i in range(len(V)):
            KZList[i] = self.Trader.Lists.GetiriFiyatList[i]

        self.GetiriFiyatBuAy = 0.0
        self.GetiriFiyatAy1 = 0.0
        self.GetiriFiyatAy2 = 0.0
        self.GetiriFiyatAy3 = 0.0
        self.GetiriFiyatAy4 = 0.0
        self.GetiriFiyatAy5 = 0.0
        self.GetiriFiyatBuHafta = 0.0
        self.GetiriFiyatHafta1 = 0.0
        self.GetiriFiyatHafta2 = 0.0
        self.GetiriFiyatHafta3 = 0.0
        self.GetiriFiyatHafta4 = 0.0
        self.GetiriFiyatHafta5 = 0.0
        self.GetiriFiyatBuGun = 0.0
        self.GetiriFiyatGun1 = 0.0
        self.GetiriFiyatGun2 = 0.0
        self.GetiriFiyatGun3 = 0.0
        self.GetiriFiyatGun4 = 0.0
        self.GetiriFiyatGun5 = 0.0
        self.GetiriFiyatBuSaat = 0.0
        self.GetiriFiyatSaat1 = 0.0
        self.GetiriFiyatSaat2 = 0.0
        self.GetiriFiyatSaat3 = 0.0
        self.GetiriFiyatSaat4 = 0.0
        self.GetiriFiyatSaat5 = 0.0

        aykz1 = 0.0
        aykz2 = 0.0
        aykz3 = 0.0
        aykz4 = 0.0
        aykz5 = 0.0
        if len(DateAyBarNoList) > 5:
            aykz1 = KZList[-1] - 1 * KZList[DateAyBarNoList[-1]]
            aykz2 = KZList[-1] - 1 * KZList[DateAyBarNoList[-2]]
            aykz3 = KZList[-1] - 1 * KZList[DateAyBarNoList[-3]]
            aykz4 = KZList[-1] - 1 * KZList[DateAyBarNoList[-4]]
            aykz5 = KZList[-1] - 1 * KZList[DateAyBarNoList[-5]]

        haftakz1 = 0.0
        haftakz2 = 0.0
        haftakz3 = 0.0
        haftakz4 = 0.0
        haftakz5 = 0.0
        if len(DateHaftaBarNoList) > 5:
            haftakz1 = KZList[-1] - 1 * KZList[DateHaftaBarNoList[-1]]
            haftakz2 = KZList[-1] - 1 * KZList[DateHaftaBarNoList[-2]]
            haftakz3 = KZList[-1] - 1 * KZList[DateHaftaBarNoList[-3]]
            haftakz4 = KZList[-1] - 1 * KZList[DateHaftaBarNoList[-4]]
            haftakz5 = KZList[-1] - 1 * KZList[DateHaftaBarNoList[-5]]

        gunkz1 = 0.0
        gunkz2 = 0.0
        gunkz3 = 0.0
        gunkz4 = 0.0
        gunkz5 = 0.0
        if len(DateGunBarNoList) > 5:
            gunkz1 = KZList[-1] - 1 * KZList[DateGunBarNoList[-1]]
            gunkz2 = KZList[-1] - 1 * KZList[DateGunBarNoList[-2]]
            gunkz3 = KZList[-1] - 1 * KZList[DateGunBarNoList[-3]]
            gunkz4 = KZList[-1] - 1 * KZList[DateGunBarNoList[-4]]
            gunkz5 = KZList[-1] - 1 * KZList[DateGunBarNoList[-5]]

        saatkz1 = 0.0
        saatkz2 = 0.0
        saatkz3 = 0.0
        saatkz4 = 0.0
        saatkz5 = 0.0
        if len(DateSaatBarNoList) > 5:
            saatkz1 = KZList[-1] - 1 * KZList[DateSaatBarNoList[-2]]
            saatkz2 = KZList[-1] - 1 * KZList[DateSaatBarNoList[-3]]
            saatkz3 = KZList[-1] - 1 * KZList[DateSaatBarNoList[-4]]
            saatkz4 = KZList[-1] - 1 * KZList[DateSaatBarNoList[-5]]
            saatkz5 = KZList[-1] - 1 * KZList[DateSaatBarNoList[-6]]

        self.GetiriFiyatBuAy = float(aykz1)
        self.GetiriFiyatAy1 = float(aykz1)
        self.GetiriFiyatAy2 = float(aykz2)
        self.GetiriFiyatAy3 = float(aykz3)
        self.GetiriFiyatAy4 = float(aykz4)
        self.GetiriFiyatAy5 = float(aykz5)
        self.GetiriFiyatBuHafta = float(haftakz1)
        self.GetiriFiyatHafta1 = float(haftakz1)
        self.GetiriFiyatHafta2 = float(haftakz2)
        self.GetiriFiyatHafta3 = float(haftakz3)
        self.GetiriFiyatHafta4 = float(haftakz4)
        self.GetiriFiyatHafta5 = float(haftakz5)
        self.GetiriFiyatBuGun = float(gunkz1)
        self.GetiriFiyatGun1 = float(gunkz1)
        self.GetiriFiyatGun2 = float(gunkz2)
        self.GetiriFiyatGun3 = float(gunkz3)
        self.GetiriFiyatGun4 = float(gunkz4)
        self.GetiriFiyatGun5 = float(gunkz5)
        self.GetiriFiyatBuSaat = float(saatkz1)
        self.GetiriFiyatSaat1 = float(saatkz1)
        self.GetiriFiyatSaat2 = float(saatkz2)
        self.GetiriFiyatSaat3 = float(saatkz3)
        self.GetiriFiyatSaat4 = float(saatkz4)
        self.GetiriFiyatSaat5 = float(saatkz5)

    def getiri_istatistikleri_ekrana_yaz(self, Sistem, PanelNo=2):
        str_list = []
        tab0 = "  "
        tab1 = "\t"
        tab2 = "\t\t"
        tab3 = "\t\t\t"
        seperator = "-" * 40

        str_list.clear()
        s = f'HisseSayisi : {self.IstatistiklerNew["HisseSayisi"]} / KontratSayisi : {self.IstatistiklerNew["KontratSayisi"]}'
        str_list.append(s)
        s = f'KomisyonVarlikAdedSayisi : {self.IstatistiklerNew["KomisyonVarlikAdedSayisi"]}'
        str_list.append(s)
        s = f'VarlikAdedCarpani : {self.IstatistiklerNew["VarlikAdedCarpani"]}'
        str_list.append(s)
        s = f'VarlikAdedSayisi : {self.IstatistiklerNew["VarlikAdedSayisi"]}'
        str_list.append(s)
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(0).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        ColumnOffset = 12
        str_list.clear()
        s = f'Puan'
        str_list.append(s)
        s = f'----------'
        str_list.append(s)
        s = f'Bu'
        str_list.append(s)
        s = f'Son 2'
        str_list.append(s)
        s = f'Son 3'
        str_list.append(s)
        s = f'Son 4'
        str_list.append(s)
        s = f'Son 5'
        str_list.append(s)
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(ColumnOffset + 1).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        str_list.clear()
        s = f'Ay'
        str_list.append(s)
        s = f'----------'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanBuAy"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanAy2"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanAy3"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanAy4"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanAy5"]}'
        str_list.append(s)
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(ColumnOffset + 2).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        str_list.clear()
        s = f'Hafta'
        str_list.append(s)
        s = f'----------'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanBuHafta"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanHafta2"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanHafta3"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanHafta4"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanHafta5"]}'
        str_list.append(s)
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(ColumnOffset + 3).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        str_list.clear()
        s = f'Gün'
        str_list.append(s)
        s = f'----------'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanBuGun"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanGun2"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanGun3"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanGun4"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanGun5"]}'
        str_list.append(s)
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(ColumnOffset + 4).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        str_list.clear()
        s = f'Saat'
        str_list.append(s)
        s = f'----------'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanBuSaat"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanSaat2"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanSaat3"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanSaat4"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriPuanSaat5"]}'
        str_list.append(s)
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(ColumnOffset + 5).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        str_list.clear()
        s = f'Fiyat'
        str_list.append(s)
        s = f'----------'
        str_list.append(s)
        s = f'Bu'
        str_list.append(s)
        s = f'Son 2'
        str_list.append(s)
        s = f'Son 3'
        str_list.append(s)
        s = f'Son 4'
        str_list.append(s)
        s = f'Son 5'
        str_list.append(s)
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(ColumnOffset + 6).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        str_list.clear()
        s = f'Ay'
        str_list.append(s)
        s = f'----------'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatBuAy"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatAy2"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatAy3"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatAy4"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatAy5"]}'
        str_list.append(s)
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(ColumnOffset + 7).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        str_list.clear()
        s = f'Hafta'
        str_list.append(s)
        s = f'----------'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatBuHafta"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatHafta2"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatHafta3"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatHafta4"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatHafta5"]}'
        str_list.append(s)
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(ColumnOffset + 8).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        str_list.clear()
        s = f'Gün'
        str_list.append(s)
        s = f'----------'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatBuGun"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatGun2"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatGun3"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatGun4"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatGun5"]}'
        str_list.append(s)
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(ColumnOffset + 9).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

        str_list.clear()
        s = f'Saat'
        str_list.append(s)
        s = f'----------'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatBuSaat"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatSaat2"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatSaat3"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatSaat4"]}'
        str_list.append(s)
        s = f'{self.IstatistiklerNew["GetiriFiyatSaat5"]}'
        str_list.append(s)
        for i, s in enumerate(str_list):
            self.panel(PanelNo).column(ColumnOffset + 10).row(i).text_message(Sistem, s, (255, 215, 0))  # Gold

    def optimizasyon_istatistiklerinin_basliklarini_dosyaya_yaz(self, Sistem, FileName):
        fileWriter = CTxtFileWriter()
        fileWriter.open_file(FileName, True)
        if not fileWriter.is_opened():
            return
        delimiter = ";"
        str_list = []
        str_list.append(f'Index {delimiter} ')
        str_list.append(f'TotalCount {delimiter} ')
        str_list.append(f'GetiriKzSistem {delimiter} ')
        str_list.append(f'GetiriKz {delimiter} ')
        str_list.append(f'GetiriKzNet {delimiter} ')
        str_list.append(f'ProfitFactor {delimiter} ')
        str_list.append(f'ProfitFactorSistem {delimiter} ')
        str_list.append(f'KarliIslemOrani {delimiter} ')
        str_list.append(f'IslemSayisi {delimiter} ')
        str_list.append(f'KomisyonIslemSayisi {delimiter} ')
        str_list.append(f'KomisyonFiyat {delimiter} ')
        str_list.append(f'KomisyonFiyatYuzde {delimiter} ')
        str_list.append(f'IlkBakiyeFiyat {delimiter} ')
        str_list.append(f'BakiyeFiyat {delimiter} ')
        str_list.append(f'GetiriFiyat {delimiter} ')
        str_list.append(f'GetiriFiyatYuzde {delimiter} ')
        str_list.append(f'BakiyeFiyatNet {delimiter} ')
        str_list.append(f'GetiriFiyatNet {delimiter} ')
        str_list.append(f'GetiriFiyatYuzdeNet {delimiter} ')
        str_list.append(f'MinBakiyeFiyat {delimiter} ')
        str_list.append(f'MaxBakiyeFiyat {delimiter} ')
        str_list.append(f'MinBakiyeFiyatYuzde {delimiter} ')
        str_list.append(f'MaxBakiyeFiyatYuzde {delimiter} ')
        str_list.append(f'MaxKarFiyat {delimiter} ')
        str_list.append(f'MaxZararFiyat {delimiter} ')
        str_list.append(f'MaxKarPuan {delimiter} ')
        str_list.append(f'MaxZararPuan {delimiter} ')
        str_list.append(f'KazandiranIslemSayisi {delimiter} ')
        str_list.append(f'KaybettirenIslemSayisi {delimiter} ')
        str_list.append(f'ToplamKarFiyat {delimiter} ')
        str_list.append(f'ToplamZararFiyat {delimiter} ')
        str_list.append(f'NetKarFiyat {delimiter} ')
        str_list.append(f'ToplamKarPuan {delimiter} ')
        str_list.append(f'ToplamZararPuan {delimiter} ')
        str_list.append(f'NetKarPuan {delimiter} ')
        str_list.append(f'GetiriMaxDD {delimiter} ')
        str_list.append(f'GetiriMaxDDTarih {delimiter} ')
        str_list.append(f'GetiriMaxDDSaat {delimiter} ')
        str_list.append(f'GetiriMaxKayip {delimiter} ')
        str_list.append(f'OrtAylikIslemSayisi {delimiter} ')
        str_list.append(f'OrtHaftalikIslemSayisi {delimiter} ')
        str_list.append(f'OrtGunlukIslemSayisi {delimiter} ')
        str_list.append(f'Eklenme Zamani {delimiter} ')
        for s in str_list:
            fileWriter.write(s)
        fileWriter.write("\n")
        fileWriter.close_file()

    def optimizasyon_istatistiklerini_dosyaya_yaz(self, Sistem, FileName, Index, TotalCount):
        fileWriter = CTxtFileWriter()
        fileWriter.open_file(FileName, True)
        if not fileWriter.is_opened():
            return
        delimiter = ";"
        str_list = []
        str_list.append(f'{Index} {delimiter} ')
        str_list.append(f'{TotalCount} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["GetiriKzSistem"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["GetiriKz"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["GetiriKzNet"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["ProfitFactor"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["ProfitFactorSistem"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["KarliIslemOrani"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["IslemSayisi"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["KomisyonIslemSayisi"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["KomisyonFiyat"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["KomisyonFiyatYuzde"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["IlkBakiyeFiyat"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["BakiyeFiyat"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["GetiriFiyat"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["GetiriFiyatYuzde"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["BakiyeFiyatNet"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["GetiriFiyatNet"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["GetiriFiyatYuzdeNet"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["MinBakiyeFiyat"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["MaxBakiyeFiyat"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["MinBakiyeFiyatYuzde"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["MaxBakiyeFiyatYuzde"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["MaxKarFiyat"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["MaxZararFiyat"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["MaxKarPuan"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["MaxZararPuan"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["KazandiranIslemSayisi"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["KaybettirenIslemSayisi"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["ToplamKarFiyat"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["ToplamZararFiyat"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["NetKarFiyat"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["ToplamKarPuan"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["ToplamZararPuan"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["NetKarPuan"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["GetiriMaxDD"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["GetiriMaxDDTarih"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["GetiriMaxDDSaat"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["GetiriMaxKayip"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["OrtAylikIslemSayisi"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["OrtHaftalikIslemSayisi"]} {delimiter} ')
        str_list.append(f'{self.IstatistiklerNew["OrtGunlukIslemSayisi"]} {delimiter} ')
        str_list.append(f'{datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")} {delimiter} ')
        for s in str_list:
            fileWriter.write(s)
        fileWriter.write("\n")
        fileWriter.close_file()