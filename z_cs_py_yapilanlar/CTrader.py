from CBase import CBase
from CSignals import CSignals
from CStatus import CStatus
from CFlags import CFlags
from CLists import CLists
from CKarZarar import CKarZarar
from CKomisyon import CKomisyon
from CBakiye import CBakiye
from CVarlikManager import CVarlikManager
from CStatistics import CStatistics
from CKarAlZararKes import CKarAlZararKes
from CTimeFilter import CTimeFilter
from CTimeUtils import CTimeUtils
from CFileUtils import CFileUtils
import datetime

class CTrader(CBase):
    def __init__(self, Sistem, Id=0):
        super().__init__()
        self.Id = Id
        self.Signals = CSignals()
        self.Status = CStatus()
        self.Flags = CFlags()
        self.Lists = CLists()
        self.KarZarar = CKarZarar()
        self.Komisyon = CKomisyon()
        self.Bakiye = CBakiye()
        self.VarlikManager = None
        self.Statistics = CStatistics()
        self.KarAlZararKes = CKarAlZararKes()
        self.TimeFilter = CTimeFilter()
        self.BakiyeInitialized = False
        self.ExecutionStepNumber = 0
        self.LastResetTime = None
        self.LastExecutionTime = None
        self.LastExecutionTimeStart = None
        self.LastExecutionTimeStop = None
        self.LastStatisticsCalculationTime = None
        self.TimeUtils = CTimeUtils()
        self.ExecutionTimeInMSec = 0
        self.DateTimeStringFormat = "%d.%m.%Y %H:%M:%S"
        self.DateStringFormat = "%d.%m.%Y"
        self.TimeStringFormat = "%H:%M:%S"
        self.StartDateTime = None
        self.StopDateTime = None
        self.StartDate = None
        self.StopDate = None
        self.StartTime = None
        self.StopTime = None
        self.StartDateTimeStr = None
        self.StopDateTimeStr = None
        self.StartDateStr = None
        self.StopDateStr = None
        self.StartTimeStr = None
        self.StopTimeStr = None
        self.EnableDateTime = None
        self.DisableDateTime = None
        self.EnableDate = None
        self.DisableDate = None
        self.EnableTime = None
        self.DisableTime = None
        self.EnableDateTimeStr = None
        self.DisableDateTimeStr = None
        self.EnableDateStr = None
        self.DisableDateStr = None
        self.EnableTimeStr = None
        self.DisableTimeStr = None

    def initialize(self, Sistem, V, Open, High, Low, Close, Volume, Lot, VarlikManager):
        self.VarlikManager = VarlikManager
        self.set_data(Sistem, V, Open, High, Low, Close, Volume, Lot)
        self.Signals.initialize(Sistem)
        self.Status.initialize(Sistem)
        self.Flags.initialize(Sistem)
        self.Lists.initialize(Sistem)
        self.Lists.create_lists(Sistem, self.BarCount)
        self.KarZarar.initialize(Sistem, self)
        self.Komisyon.initialize(Sistem, self)
        self.Bakiye.initialize(Sistem, self)
        self.Statistics.initialize(Sistem, self)
        self.KarAlZararKes.initialize(Sistem, self)
        self.TimeFilter.initialize(Sistem, self)
        self.TimeUtils.initialize(Sistem, V, Open, High, Low, Close, Volume, Lot)
        self.reset(Sistem)
        self.BakiyeInitialized = False
        return self

    def reset(self, Sistem):
        self.Signals.reset(Sistem)
        self.Status.reset(Sistem)
        self.Flags.reset(Sistem)
        self.Lists.reset(Sistem)
        self.KarZarar.reset(Sistem)
        self.Komisyon.reset(Sistem)
        self.Bakiye.reset(Sistem)
        self.Statistics.reset(Sistem)
        self.KarAlZararKes.reset(Sistem)
        self.TimeFilter.reset(Sistem)
        self.TimeUtils.reset(Sistem)
        self.ExecutionStepNumber = 0
        self.LastResetTime = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
        self.LastExecutionTime = ""
        self.LastExecutionTimeStart = ""
        self.LastExecutionTimeStop = ""
        self.LastStatisticsCalculationTime = ""
        self.ExecutionTimeInMSec = 0
        self.reset_date_times(Sistem)
        return self

    def start(self, Sistem):
        self.LastExecutionTimeStart = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
        self.LastExecutionTime = self.LastExecutionTimeStart
        self.TimeUtils.start_timer(Sistem)

    def stop(self, Sistem):
        self.LastExecutionTimeStop = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
        self.TimeUtils.stop_timer(Sistem)
        self.ExecutionTimeInMSec = self.TimeUtils.get_execution_time_in_msec(Sistem)

    def dongu_basi_degiskenleri_resetle(self, Sistem, BarIndex):
        result = 0
        i = BarIndex
        self.Lists.BarIndexList[i] = i
        self.Lists.YonList[i] = ""
        self.Lists.SeviyeList[i] = 0.0
        self.Lists.SinyalList[i] = 0.0
        self.Lists.KarZararPuanList[i] = 0.0
        self.Lists.KarZararFiyatList[i] = 0.0
        self.Lists.KarZararFiyatYuzdeList[i] = 0.0
        self.Status.KarZararFiyat = 0.0
        self.Status.KarZararPuan = 0.0
        self.Status.KarZararFiyatYuzde = 0.0
        self.Lists.KarAlList[i] = 0.0
        self.Lists.IzleyenStopList[i] = 0.0
        self.Lists.IslemSayisiList[i] = 0.0
        self.Lists.AlisSayisiList[i] = 0.0
        self.Lists.SatisSayisiList[i] = 0.0
        self.Lists.FlatSayisiList[i] = 0.0
        self.Lists.PassSayisiList[i] = 0.0
        self.Lists.KontratSayisiList[i] = 0.0
        self.Lists.VarlikAdedSayisiList[i] = 0.0
        self.Lists.KomisyonVarlikAdedSayisiList[i] = 0.0
        self.Lists.KomisyonIslemSayisiList[i] = 0.0
        self.Lists.KomisyonFiyatList[i] = 0.0
        self.Lists.KardaBarSayisiList[i] = 0.0
        self.Lists.ZarardaBarSayisiList[i] = 0.0
        self.Lists.BakiyeFiyatList[i] = self.Status.BakiyeFiyat
        self.Lists.GetiriFiyatList[i] = self.Lists.BakiyeFiyatList[i] - self.Status.BakiyeFiyat
        self.Lists.BakiyePuanList[i] = self.Status.BakiyePuan
        self.Lists.GetiriPuanList[i] = self.Lists.BakiyePuanList[i] - self.Status.BakiyePuan
        self.Lists.EmirKomutList[i] = 0.0
        self.Lists.EmirStatusList[i] = 0.0
        if self.ExecutionStepNumber == 0:
            pass
        self.ExecutionStepNumber += 1
        return result

    def dongu_basi_degiskenleri_guncelle(self, Sistem, BarIndex):
        result = 0
        i = BarIndex
        self.Status.KomisyonVarlikAdedSayisi = self.VarlikManager.KomisyonVarlikAdedSayisi
        self.Status.KomisyonCarpan = self.VarlikManager.KomisyonCarpan
        self.Flags.KomisyonuDahilEt = self.VarlikManager.KomisyonuDahilEt
        self.Status.KaymaMiktari = self.VarlikManager.KaymaMiktari
        self.Flags.KaymayiDahilEt = self.VarlikManager.KaymayiDahilEt
        self.Status.VarlikAdedSayisi = self.VarlikManager.VarlikAdedSayisi
        self.Status.VarlikAdedCarpani = self.VarlikManager.VarlikAdedCarpani
        self.Status.KontratSayisi = self.VarlikManager.KontratSayisi
        self.Status.HisseSayisi = self.VarlikManager.HisseSayisi
        self.Status.IlkBakiyeFiyat = self.VarlikManager.IlkBakiyeFiyat
        self.Status.IlkBakiyePuan = self.VarlikManager.IlkBakiyePuan
        self.Status.GetiriFiyatTipi = self.VarlikManager.GetiriFiyatTipi
        if not self.BakiyeInitialized:
            self.BakiyeInitialized = True
            self.Status.BakiyeFiyat = self.Status.IlkBakiyeFiyat
            self.Status.BakiyePuan = self.Status.IlkBakiyePuan
        return result

    def anlik_kar_zarar_hesapla(self, Sistem, BarIndex, Type="C"):
        return self.KarZarar.anlik_kar_zarar_hesapla(Sistem, BarIndex, Type)

    def emirleri_resetle(self, Sistem, BarIndex):
        result = 0
        i = BarIndex
        self.Signals.Al = False
        self.Signals.Sat = False
        self.Signals.FlatOl = False
        self.Signals.PasGec = False
        self.Signals.KarAl = False
        self.Signals.ZararKes = False
        return result

    def emirleri_setle(self, Sistem, BarIndex, Al, Sat, FlatOl=False, PasGec=False, KarAl=False, ZararKes=False):
        result = 0
        i = BarIndex
        self.Signals.Al = Al
        self.Signals.Sat = Sat
        self.Signals.FlatOl = FlatOl
        self.Signals.PasGec = PasGec
        self.Signals.KarAl = KarAl
        self.Signals.ZararKes = ZararKes
        return result

    def emirleri_uygula(self, Sistem, BarIndex):
        result = 0
        i = BarIndex
        self.Flags.AGerceklesti = False
        self.Flags.SGerceklesti = False
        self.Flags.FGerceklesti = False
        self.Flags.PGerceklesti = False
        AnlikKapanisFiyati = self.V[i].Close
        AnlikYuksekFiyati = self.V[i].High
        AnlikDusukFiyati = self.V[i].Low
        if self.Signals.Al:
            self.Signals.Sinyal = "A"
            self.Signals.EmirKomut = 1
            self.Status.AlKomutSayisi += 1
        if self.Signals.Sat:
            self.Signals.Sinyal = "S"
            self.Signals.EmirKomut = 2
            self.Status.SatKomutSayisi += 1
        if self.Signals.PasGec:
            self.Signals.Sinyal = "P"
            self.Signals.EmirKomut = 3
            self.Status.PasGecKomutSayisi += 1
        if self.Signals.KarAl:
            self.Signals.Sinyal = "F"
            self.Signals.EmirKomut = 4
            self.Status.KarAlKomutSayisi += 1
        if self.Signals.ZararKes:
            self.Signals.Sinyal = "F"
            self.Signals.EmirKomut = 5
            self.Status.ZararKesKomutSayisi += 1
        if self.Signals.FlatOl:
            self.Signals.Sinyal = "F"
            self.Signals.EmirKomut = 6
            self.Status.FlatOlKomutSayisi += 1
        self.Status.KarAlSayisi = self.Status.KarAlKomutSayisi
        self.Status.ZararKesSayisi = self.Status.ZararKesKomutSayisi
        if self.Signals.Sinyal == "A" and self.Signals.SonYon != "A":
            self.Signals.PrevAFiyat = self.Signals.SonAFiyat
            self.Signals.PrevABarNo = self.Signals.SonABarNo
            self.Signals.PrevYon = self.Signals.SonYon
            self.Signals.PrevFiyat = self.Signals.SonFiyat
            self.Signals.PrevBarNo = self.Signals.SonBarNo
            if self.Signals.PrevYon == "F":
                pass
            if self.Signals.PrevYon == "S":
                pass
            self.Lists.YonList[i] = "A"
            self.Signals.SonYon = self.Lists.YonList[i]
            self.Signals.SonFiyat = AnlikKapanisFiyati
            if self.Flags.KaymayiDahilEt:
                self.Signals.SonFiyat = AnlikYuksekFiyati
            self.Lists.SeviyeList[i] = self.Signals.SonFiyat
            self.Signals.SonBarNo = i
            self.Signals.SonAFiyat = self.Signals.SonFiyat
            self.Signals.SonABarNo = self.Signals.SonBarNo
            if self.Signals.PrevYon == "F":
                self.Status.KomisyonIslemSayisi += 1
                self.Signals.EmirStatus = 1
            if self.Signals.PrevYon == "S":
                fark = self.Signals.SonFiyat - self.Signals.SonSFiyat
                if fark < 0:
                    self.Status.KazandiranSatisSayisi += 1
                elif fark > 0:
                    self.Status.KaybettirenSatisSayisi += 1
                else:
                    self.Status.NotrSatisSayisi += 1
                self.Status.KomisyonIslemSayisi += 2
                self.Signals.EmirStatus = 2
            self.Flags.BakiyeGuncelle = True
            self.Flags.KomisyonGuncelle = True
            self.Flags.DonguSonuIstatistikGuncelle = True
            self.Status.IslemSayisi += 1
            self.Status.AlisSayisi += 1
            self.Flags.AGerceklesti = True
        elif self.Signals.Sinyal == "S" and self.Signals.SonYon != "S":
            self.Signals.PrevSFiyat = self.Signals.SonSFiyat
            self.Signals.PrevSBarNo = self.Signals.SonSBarNo
            self.Signals.PrevYon = self.Signals.SonYon
            self.Signals.PrevFiyat = self.Signals.SonFiyat
            self.Signals.PrevBarNo = self.Signals.SonBarNo
            if self.Signals.PrevYon == "F":
                pass
            if self.Signals.PrevYon == "A":
                pass
            self.Lists.YonList[i] = "S"
            self.Signals.SonYon = self.Lists.YonList[i]
            self.Signals.SonFiyat = AnlikKapanisFiyati
            if self.Flags.KaymayiDahilEt:
                self.Signals.SonFiyat = AnlikDusukFiyati
            self.Lists.SeviyeList[i] = self.Signals.SonFiyat
            self.Signals.SonBarNo = i
            self.Signals.SonSFiyat = self.Signals.SonFiyat
            self.Signals.SonSBarNo = self.Signals.SonSBarNo
            if self.Signals.PrevYon == "F":
                self.Status.KomisyonIslemSayisi += 1
                self.Signals.EmirStatus = 3
            if self.Signals.PrevYon == "A":
                fark = self.Signals.SonFiyat - self.Signals.SonAFiyat
                if fark > 0:
                    self.Status.KazandiranAlisSayisi += 1
                elif fark < 0:
                    self.Status.KaybettirenAlisSayisi += 1
                else:
                    self.Status.NotrAlisSayisi += 1
                self.Status.KomisyonIslemSayisi += 2
                self.Signals.EmirStatus = 4
            self.Flags.BakiyeGuncelle = True
            self.Flags.KomisyonGuncelle = True
            self.Flags.DonguSonuIstatistikGuncelle = True
            self.Status.IslemSayisi += 1
            self.Status.SatisSayisi += 1
            self.Flags.SGerceklesti = True
        elif self.Signals.Sinyal == "F" and self.Signals.SonYon != "F":
            self.Signals.PrevFFiyat = self.Signals.SonFFiyat
            self.Signals.PrevFBarNo = self.Signals.SonFBarNo
            self.Signals.PrevYon = self.Signals.SonYon
            self.Signals.PrevFiyat = self.Signals.SonFiyat
            self.Signals.PrevBarNo = self.Signals.SonBarNo
            if self.Signals.PrevYon == "A":
                pass
            if self.Signals.PrevYon == "S":
                pass
            self.Lists.YonList[i] = "F"
            self.Signals.SonYon = self.Lists.YonList[i]
            self.Signals.SonFiyat = AnlikKapanisFiyati
            if self.Flags.KaymayiDahilEt:
                if self.Signals.PrevYon == "A":
                    self.Signals.SonFiyat = AnlikDusukFiyati
                if self.Signals.PrevYon == "S":
                    self.Signals.SonFiyat = AnlikYuksekFiyati
            self.Lists.SeviyeList[i] = self.Signals.SonFiyat
            self.Signals.SonBarNo = i
            self.Signals.SonFFiyat = self.Signals.SonFiyat
            self.Signals.SonFBarNo = self.Signals.SonFBarNo
            if self.Signals.PrevYon == "A":
                fark = self.Signals.SonFiyat - self.Signals.SonAFiyat
                if fark > 0:
                    self.Status.KazandiranAlisSayisi += 1
                elif fark < 0:
                    self.Status.KaybettirenAlisSayisi += 1
                else:
                    self.Status.NotrAlisSayisi += 1
                self.Status.KomisyonIslemSayisi += 1
                self.Signals.EmirStatus = 5
            if self.Signals.PrevYon == "S":
                fark = self.Signals.SonFiyat - self.Signals.SonSFiyat
                if fark < 0:
                    self.Status.KazandiranSatisSayisi += 1
                elif fark > 0:
                    self.Status.KaybettirenSatisSayisi += 1
                else:
                    self.Status.NotrSatisSayisi += 1
                self.Status.KomisyonIslemSayisi += 1
                self.Signals.EmirStatus = 6
            self.Flags.BakiyeGuncelle = True
            self.Flags.KomisyonGuncelle = True
            self.Flags.DonguSonuIstatistikGuncelle = True
            self.Status.IslemSayisi += 1
            self.Status.FlatSayisi += 1
            self.Flags.FGerceklesti = True
        elif self.Signals.Sinyal == "P" or self.Signals.Sinyal == "":
            self.Signals.PrevPFiyat = self.Signals.SonPFiyat
            self.Signals.PrevPBarNo = self.Signals.SonPBarNo
            self.Signals.SonPFiyat = AnlikKapanisFiyati
            self.Signals.SonPBarNo = i
            if self.Signals.SonYon == "A":
                self.Signals.EmirStatus = 7
            if self.Signals.SonYon == "S":
                self.Signals.EmirStatus = 8
            if self.Signals.SonYon == "F":
                self.Signals.EmirStatus = 9
            self.Flags.BakiyeGuncelle = True
            self.Flags.KomisyonGuncelle = True
            self.Flags.DonguSonuIstatistikGuncelle = True
            self.Status.PassSayisi += 1
            self.Flags.PGerceklesti = True
        self.Status.KazandiranIslemSayisi = self.Status.KazandiranAlisSayisi + self.Status.KazandiranSatisSayisi
        self.Status.KaybettirenIslemSayisi = self.Status.KaybettirenAlisSayisi + self.Status.KaybettirenSatisSayisi
        self.Status.NotrIslemSayisi = self.Status.NotrAlisSayisi + self.Status.NotrSatisSayisi
        if self.Flags.AGerceklesti or self.Flags.SGerceklesti or self.Flags.FGerceklesti:
            self.Status.KardaBarSayisi = 0
            self.Status.ZarardaBarSayisi = 0
        if self.Status.IslemSayisi > 0:
            self.Flags.AnlikKarZararHesaplaEnabled = True
            self.Flags.KarAlYuzdeHesaplaEnabled = True
            self.Flags.IzleyenStopYuzdeHesaplaEnabled = True
            self.Flags.ZararKesYuzdeHesaplaEnabled = True
            self.Flags.KarAlSeviyeHesaplaEnabled = True
            self.Flags.ZararKesSeviyeHesaplaEnabled = True
        self.Flags.AGerceklesti = False
        self.Flags.SGerceklesti = False
        self.Flags.FGerceklesti = False
        self.Flags.PGerceklesti = False
        self.Lists.EmirKomutList[i] = self.Signals.EmirKomut
        self.Lists.EmirStatusList[i] = self.Signals.EmirStatus
        return result

    def sistem_yon_listesini_guncelle(self, Sistem, BarIndex):
        result = 0
        i = BarIndex
        Sistem.Yon[i] = self.Lists.YonList[i]
        return result

    def sistem_seviye_listesini_guncelle(self, Sistem, BarIndex):
        result = 0
        i = BarIndex
        Sistem.Seviye[i] = self.Lists.SeviyeList[i]
        return result

    def sinyal_listesini_guncelle(self, Sistem, BarIndex):
        result = 0
        i = BarIndex
        if self.Signals.SonYon == "A":
            self.Lists.SinyalList[i] = 1.0
        elif self.Signals.SonYon == "S":
            self.Lists.SinyalList[i] = -1.0
        elif self.Signals.SonYon == "F":
            self.Lists.SinyalList[i] = 0.0
        return result

    def islem_listesini_guncelle(self, Sistem, BarIndex):
        result = 0
        i = BarIndex
        self.Lists.IslemSayisiList[i] = self.Status.IslemSayisi
        self.Lists.AlisSayisiList[i] = self.Status.AlisSayisi
        self.Lists.SatisSayisiList[i] = self.Status.SatisSayisi
        self.Lists.FlatSayisiList[i] = self.Status.FlatSayisi
        self.Lists.PassSayisiList[i] = self.Status.PassSayisi
        self.Lists.VarlikAdedSayisiList[i] = self.Status.VarlikAdedSayisi
        self.Lists.KontratSayisiList[i] = self.Status.KontratSayisi
        self.Lists.KomisyonVarlikAdedSayisiList[i] = self.Status.KomisyonVarlikAdedSayisi
        self.Lists.KomisyonIslemSayisiList[i] = self.Status.KomisyonIslemSayisi
        self.Lists.KomisyonFiyatList[i] = self.Status.KomisyonFiyat
        self.Lists.KardaBarSayisiList[i] = self.Status.KardaBarSayisi
        self.Lists.ZarardaBarSayisiList[i] = self.Status.ZarardaBarSayisi
        return result

    def komisyon_listesini_guncelle(self, Sistem, BarIndex):
        result = 0
        i = BarIndex
        self.Komisyon.hesapla(Sistem, i)
        if self.Flags.KomisyonGuncelle:
            self.Flags.KomisyonGuncelle = False
        return result

    def bakiye_listesini_guncelle(self, Sistem, BarIndex):
        result = 0
        i = BarIndex
        self.Bakiye.hesapla(Sistem, i)
        if self.Flags.BakiyeGuncelle:
            self.Flags.BakiyeGuncelle = False
        return result

    def dongu_sonu_degiskenleri_setle(self, Sistem, BarIndex):
        result = 0
        i = BarIndex
        return result

    def istatistikleri_hesapla(self, Sistem):
        result = 0
        self.Statistics.hesapla(Sistem)
        return result

    def istatistikleri_ekrana_yaz(self, Sistem, PanelNo=1):
        result = 0
        self.Statistics.istatistikleri_ekrana_yaz(Sistem, PanelNo)
        return result

    def getiri_istatistikleri_ekrana_yaz(self, Sistem, PanelNo=2):
        result = 0
        self.Statistics.getiri_istatistikleri_ekrana_yaz(Sistem, PanelNo)
        return result

    def istatistikleri_dosyaya_yaz(self, Sistem, FileName):
        self.Statistics.istatistikleri_dosyaya_yaz(Sistem, FileName)

    def optimizasyon_istatistiklerinin_basliklarini_dosyaya_yaz(self, Sistem, FileName):
        self.Statistics.optimizasyon_istatistiklerinin_basliklarini_dosyaya_yaz(Sistem, FileName)

    def optimizasyon_istatistiklerini_dosyaya_yaz(self, Sistem, FileName, Index, TotalCount):
        self.Statistics.optimizasyon_istatistiklerini_dosyaya_yaz(Sistem, FileName, Index, TotalCount)

    def is_son_yon_a(self, Sistem):
        return self.Signals.SonYon == "A"

    def is_son_yon_s(self, Sistem):
        return self.Signals.SonYon == "S"

    def is_son_yon_f(self, Sistem):
        return self.Signals.SonYon == "F"

    def ideal_getiri_hesapla(self, Sistem, KaymaMiktari=0.0, BaslangicTarihi="01/01/1900", BitisTarihi="01/01/2100"):
        if not self.Flags.IdealGetiriHesapla:
            return
        self.Flags.IdealGetiriHesaplandi = True
        Sistem.GetiriHesapla(BaslangicTarihi, KaymaMiktari)
        Sistem.GetiriMaxDDHesapla(BaslangicTarihi, BitisTarihi)
        for i in range(len(Sistem.GetiriKZ)):
            self.Lists.GetiriKzSistem[i] = Sistem.GetiriKZ[i]
            self.Lists.GetiriKzNetSistem[i] = 0.0
        self.Status.GetiriKzSistem = self.Lists.GetiriKzSistem[-1]
        self.Status.GetiriKzNetSistem = 0.0

    def gun_sonu_poz_kapat(self, Sistem, BarIndex, GunSonuPozKapatEnabled=True):
        i = BarIndex
        GunSonuPozKapatildi = False
        if GunSonuPozKapatEnabled:
            if i < self.BarCount - 1 and self.V[i].Date.day != self.V[i + 1].Date.day:
                self.Signals.FlatOl = True
                GunSonuPozKapatildi = True
        return GunSonuPozKapatildi

    def gun_sonu_poz_kapat2(self, Sistem, BarIndex, GunSonuPozKapatEnabled=True, Hour=18, Minute=0):
        i = BarIndex
        GunSonuPozKapatildi = False
        if GunSonuPozKapatEnabled:
            if self.V[i].Date.hour == 18 and self.V[i].Date.minute >= 0:
                self.Signals.FlatOl = True
                GunSonuPozKapatildi = True
        return GunSonuPozKapatildi

    def sinyalleri_ekrana_ciz(self, Sistem, k, SistemGetiriKZDahilEt=False):
        myTrader = self
        BarIndexList = myTrader.Lists.BarIndexList
        SeviyeList = myTrader.Lists.SeviyeList
        SinyalList = myTrader.Lists.SinyalList
        KarZararPuanList = myTrader.Lists.KarZararPuanList
        KarZararFiyatList = myTrader.Lists.KarZararFiyatList
        KarZararFiyatYuzdeList = myTrader.Lists.KarZararFiyatYuzdeList
        IslemSayisiList = myTrader.Lists.IslemSayisiList
        AlisSayisiList = myTrader.Lists.AlisSayisiList
        SatisSayisiList = myTrader.Lists.SatisSayisiList
        FlatSayisiList = myTrader.Lists.FlatSayisiList
        PassSayisiList = myTrader.Lists.PassSayisiList
        KontratSayisiList = myTrader.Lists.KontratSayisiList
        VarlikAdedSayisiList = myTrader.Lists.VarlikAdedSayisiList
        KomisyonVarlikAdedSayisiList = myTrader.Lists.KomisyonVarlikAdedSayisiList
        KomisyonIslemSayisiList = myTrader.Lists.KomisyonIslemSayisiList
        KomisyonFiyatList = myTrader.Lists.KomisyonFiyatList
        KardaBarSayisiList = myTrader.Lists.KardaBarSayisiList
        ZarardaBarSayisiList = myTrader.Lists.ZarardaBarSayisiList
        BakiyePuanList = myTrader.Lists.BakiyePuanList
        BakiyeFiyatList = myTrader.Lists.BakiyeFiyatList
        GetiriPuanList = myTrader.Lists.GetiriPuanList
        GetiriFiyatList = myTrader.Lists.GetiriFiyatList
        GetiriPuanYuzdeList = myTrader.Lists.GetiriPuanYuzdeList
        GetiriFiyatYuzdeList = myTrader.Lists.GetiriFiyatYuzdeList
        BakiyePuanNetList = myTrader.Lists.BakiyePuanNetList
        BakiyeFiyatNetList = myTrader.Lists.BakiyeFiyatNetList
        GetiriPuanNetList = myTrader.Lists.GetiriPuanNetList
        GetiriFiyatNetList = myTrader.Lists.GetiriFiyatNetList
        GetiriPuanYuzdeNetList = myTrader.Lists.GetiriPuanYuzdeNetList
        GetiriFiyatYuzdeNetList = myTrader.Lists.GetiriFiyatYuzdeNetList
        GetiriKz = myTrader.Lists.GetiriKz
        GetiriKzNet = myTrader.Lists.GetiriKzNet
        EmirKomutList = myTrader.Lists.EmirKomutList
        EmirStatusList = myTrader.Lists.EmirStatusList
        SistemGetiriKZ = Sistem.GetiriKZ if SistemGetiriKZDahilEt else Sistem.Liste(0)
        Sistem.Cizgiler[k].Deger = Sistem.GetiriKZ
        Sistem.Cizgiler[k].Aciklama = "Sistem.GetiriKZ "
        k += 1
        Sistem.Cizgiler[k].Deger = KarZararFiyatList
        Sistem.Cizgiler[k].Aciklama = "KarZararFiyat"
        k += 1
        Sistem.Cizgiler[k].Deger = KarZararFiyatYuzdeList
        Sistem.Cizgiler[k].Aciklama = "KarZararFiyatYuzdeList"
        k += 1
        Sistem.Cizgiler[k].Deger = GetiriKz
        Sistem.Cizgiler[k].Aciklama = "GetiriKz"
        k += 1
        Sistem.Cizgiler[k].Deger = GetiriKzNet
        Sistem.Cizgiler[k].Aciklama = "GetiriKzNet"
        k += 1
        Sistem.Cizgiler[k].Deger = GetiriFiyatList
        Sistem.Cizgiler[k].Aciklama = "GetiriFiyat"
        k += 1
        Sistem.Cizgiler[k].Deger = BakiyeFiyatList
        Sistem.Cizgiler[k].Aciklama = "BakiyeFiyat"
        k += 1
        Sistem.Cizgiler[k].Deger = GetiriFiyatNetList
        Sistem.Cizgiler[k].Aciklama = "GetiriFiyatNet"
        k += 1
        Sistem.Cizgiler[k].Deger = BakiyeFiyatNetList
        Sistem.Cizgiler[k].Aciklama = "BakiyeFiyatNet"
        k += 1
        Sistem.Cizgiler[k].Deger = KomisyonFiyatList
        Sistem.Cizgiler[k].Aciklama = "KomisyonFiyatList"
        k += 1
        Sistem.Cizgiler[k].Deger = KomisyonIslemSayisiList
        Sistem.Cizgiler[k].Aciklama = "KomisyonIslemSayisiList"
        k += 1
        Sistem.Cizgiler[k].Deger = IslemSayisiList
        Sistem.Cizgiler[k].Aciklama = "IslemSayisiList "
        k += 1
        Sistem.Cizgiler[k].Deger = EmirKomutList
        Sistem.Cizgiler[k].Aciklama = "EmirKomutList "
        k += 1
        Sistem.Cizgiler[k].Deger = EmirStatusList
        Sistem.Cizgiler[k].Aciklama = "EmirStatusList "
        k += 1
        Sistem.Cizgiler[k].Deger = BarIndexList
        Sistem.Cizgiler[k].Aciklama = "BarIndexList"
        k += 1
        return k

    def get_bar_values_description(self, Sistem):
        delimiter = ";"
        LogMessage = ""
        mode = 1
        if mode == 0:
            LogMessage = f'#  {"Column Names"},{No},{Date},{Time},{Open},{High},{Low},{Close},{Vol},{Size(Lot)}'
        else:
            LogMessage = f'#  {"Sutunlar":<12} {delimiter} {"No"} {delimiter} {"Date"} {delimiter} {"Time"} {delimiter} {"Open"} {delimiter} {"High"} {delimiter} {"Low"} {delimiter} {"Close"} {delimiter} {"Vol"} {delimiter} {"Size(Lot)"}'
        return LogMessage

    def get_bar_values_as_string(self, Sistem, BarIndex):
        i = BarIndex
        delimiter = ";"
        LogMessage = ""
        mode = 1
        if mode == 0:
            LogMessage = f'{i:<5} 	 {self.V[i].Date.strftime("%Y.%m.%d %H:%M:%S"):<20} 	 {self.V[i].Open:5.2f} 	 {self.V[i].High:5.2f} 	 {self.V[i].Low:5.2f} 	 {self.V[i].Close:5.2f} 	 {self.V[i].Vol:10.0f} 	 {self.V[i].Size:5.0f}'
        else:
            LogMessage = f'{delimiter} {i:<5} {delimiter} {self.V[i].Date.strftime("%Y.%m.%d"):>10} {delimiter} {self.V[i].Date.strftime("%H:%M:%S"):>10} {delimiter} {self.V[i].Open:10.2f} {delimiter} {self.V[i].High:10.2f} {delimiter} {self.V[i].Low:10.2f} {delimiter} {self.V[i].Close:10.2f} {delimiter} {self.V[i].Vol:10.0f} {delimiter} {self.V[i].Size:10.0f} {delimiter}'
        return LogMessage

    def write_data_to_file_ohlc(self, Sistem, FileName):
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
        myFileUtils.reset(Sistem).enable_logging(Sistem).open_log_file(Sistem, logFileFullName, False, False)
        logMessage = f'#  {"Aciklama (1)":<14}   ; {aciklama1.strip()}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Aciklama (2)":<14}   ; {aciklama2.strip()}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Aciklama (3)":<14}   ; {aciklama3.strip()}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Aciklama (4)":<14}   ; {aciklama4.strip()}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Aciklama (5)":<14}   ; {aciklama5.strip()}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Aciklama (6)":<14}   ; {aciklama6.strip()}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Log Zamani":<14}   ; {datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Sembol":<14}   ; {Sistem.Sembol}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Periyod":<14}   ; {Sistem.Periyot}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Bar Sayisi":<14}   ; {self.BarCount}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Ilk Bar Zamani":<10}   ; {self.V[0].Date.strftime("%Y.%m.%d %H:%M:%S")}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Son Bar Zamani":<10}   ; {self.V[-1].Date.strftime("%Y.%m.%d %H:%M:%S")}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Gecen Sure (A)":<10}   ; {myTimeUtils.gecen_sure(Sistem, "A"):0.1f}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Gecen Sure (G)":<10}   ; {myTimeUtils.gecen_sure(Sistem, "G"):0.0f}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = self.get_bar_values_description(Sistem)
        myFileUtils.write_to_log_file(Sistem, logMessage)
        for i in range(self.BarCount):
            logMessage = f'{self.get_bar_values_as_string(Sistem, i)}'
            myFileUtils.write_to_log_file(Sistem, logMessage)
        myFileUtils.close_log_file(Sistem)

    def write_data_to_file_custom(self, Sistem, FileName, DataLists, CaptionList):
        delimiter = ";"
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
        myFileUtils.reset(Sistem).enable_logging(Sistem).open_log_file(Sistem, logFileFullName, False, False)
        logMessage = f'#  {"Aciklama (1)":<14}   ; {aciklama1.strip()}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Aciklama (2)":<14}   ; {aciklama2.strip()}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Aciklama (3)":<14}   ; {aciklama3.strip()}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Aciklama (4)":<14}   ; {aciklama4.strip()}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Aciklama (5)":<14}   ; {aciklama5.strip()}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Aciklama (6)":<14}   ; {aciklama6.strip()}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Log Zamani":<14}   ; {datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Sembol":<14}   ; {Sistem.Sembol}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Periyod":<14}   ; {Sistem.Periyot}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Bar Sayisi":<14}   ; {self.BarCount}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Ilk Bar Zamani":<10}   ; {self.V[0].Date.strftime("%Y.%m.%d %H:%M:%S")}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Son Bar Zamani":<10}   ; {self.V[-1].Date.strftime("%Y.%m.%d %H:%M:%S")}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Gecen Sure (A)":<10}   ; {myTimeUtils.gecen_sure(Sistem, "A"):0.1f}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Gecen Sure (G)":<10}   ; {myTimeUtils.gecen_sure(Sistem, "G"):0.0f}\t'
        myFileUtils.write_to_log_file(Sistem, logMessage)
        logMessage = f'#  {"Sutunlar":<12} {delimiter} {"No"}'
        for j in range(len(CaptionList)):
            logMessage = logMessage + delimiter + CaptionList[j]
        myFileUtils.write_to_log_file(Sistem, logMessage)
        for i in range(self.BarCount):
            logMessage = f'{delimiter} {i:<5}'
            for j in range(len(DataLists)):
                column = DataLists[j]
                logMessage = logMessage + delimiter + str(column[i])
            myFileUtils.write_to_log_file(Sistem, logMessage)
        myFileUtils.close_log_file(Sistem)

    def reset_date_times(self, Sistem):
        useLastBarDateTime = True
        self.StartDateTime = self.V[0].Date
        self.StopDateTime = self.V[Sistem.BarSayisi - 1].Date if useLastBarDateTime else datetime.datetime.now()
        self.StartDate = self.StartDateTime
        self.StopDate = self.StopDateTime
        self.StartTime = self.StartDateTime
        self.StopTime = self.StopDateTime
        self.StartDateTimeStr = self.StartDateTime.strftime(self.DateTimeStringFormat)
        self.StopDateTimeStr = self.StopDateTime.strftime(self.DateTimeStringFormat)
        self.StartDateStr = self.StartDate.strftime(self.DateStringFormat)
        self.StopDateStr = self.StopDate.strftime(self.DateStringFormat)
        self.StartTimeStr = self.StartTime.strftime(self.TimeStringFormat)
        self.StopTimeStr = self.StopTime.strftime(self.TimeStringFormat)
        return self

    def set_date_times(self, Sistem, StartDate, StartTime, StopDate, StopTime):
        date1 = StartDate.strip()
        time1 = StartTime.strip()
        date2 = StopDate.strip()
        time2 = StopTime.strip()
        dateTime1 = date1 + " " + time1
        dateTime2 = date2 + " " + time2
        suffixDate = "09:30:00"
        prefixTime = "01.01.1900"
        self.StartDateTime = self.TimeUtils.get_date_time(date1, time1)
        self.StopDateTime = self.TimeUtils.get_date_time(date2, time2)
        self.StartDate = self.TimeUtils.get_date_time(date1 + " " + suffixDate)
        self.StopDate = self.TimeUtils.get_date_time(date2 + " " + suffixDate)
        self.StartTime = self.TimeUtils.get_date_time(prefixTime + " " + time1)
        self.StopTime = self.TimeUtils.get_date_time(prefixTime + " " + time2)
        self.StartDateTimeStr = self.StartDateTime.strftime(self.DateTimeStringFormat)
        self.StopDateTimeStr = self.StopDateTime.strftime(self.DateTimeStringFormat)
        self.StartDateStr = self.StartDate.strftime(self.DateStringFormat)
        self.StopDateStr = self.StopDate.strftime(self.DateStringFormat)
        self.StartTimeStr = self.StartTime.strftime(self.TimeStringFormat)
        self.StopTimeStr = self.StopTime.strftime(self.TimeStringFormat)
        return self

    def set_date_times_from_strings(self, Sistem, StartDateTime, StopDateTime):
        dateTime1 = StartDateTime.strip()
        dateTime2 = StopDateTime.strip()
        date1 = dateTime1[0:10]
        time1 = dateTime1[11:]
        date2 = dateTime2[0:10]
        time2 = dateTime2[11:]
        suffixDate = "09:30:00"
        prefixTime = "01.01.1900"
        self.StartDateTime = self.TimeUtils.get_date_time(dateTime1)
        self.StopDateTime = self.TimeUtils.get_date_time(dateTime2)
        self.StartDate = self.TimeUtils.get_date_time(date1 + " " + suffixDate)
        self.StopDate = self.TimeUtils.get_date_time(date2 + " " + suffixDate)
        self.StartTime = self.TimeUtils.get_date_time(prefixTime + " " + time1)
        self.StopTime = self.TimeUtils.get_date_time(prefixTime + " " + time2)
        self.StartDateTimeStr = self.StartDateTime.strftime(self.DateTimeStringFormat)
        self.StopDateTimeStr = self.StopDateTime.strftime(self.DateTimeStringFormat)
        self.StartDateStr = self.StartDate.strftime(self.DateStringFormat)
        self.StopDateStr = self.StopDate.strftime(self.DateStringFormat)
        self.StartTimeStr = self.StartTime.strftime(self.TimeStringFormat)
        self.StopTimeStr = self.StopTime.strftime(self.TimeStringFormat)
        return self

    def set_date_time(self, Sistem, StartDate, StartTime):
        StartDateTime = StartDate + " " + StartTime
        return self.set_date_time_from_string(Sistem, StartDateTime)

    def set_date_time_from_string(self, Sistem, StartDateTime):
        return self.set_date_times_from_strings(Sistem, StartDateTime, datetime.datetime.now().strftime(self.DateTimeStringFormat))

    def islem_zaman_filtresi_uygula(self, Sistem, BarIndex, FilterMode, IsTradeEnabled, IsPozKapatEnabled, CheckResult):
        i = BarIndex
        BarDateTime = self.V[i].Date
        startDateTime = self.StartDateTimeStr
        stopDateTime = self.StopDateTimeStr
        startDate = self.StartDateStr
        stopDate = self.StopDateStr
        startTime = self.StartTimeStr
        stopTime = self.StopTimeStr
        nowDateTime = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        nowDate = datetime.datetime.now().strftime("%d.%m.%Y")
        nowTime = datetime.datetime.now().strftime("%H:%M:%S")
        useTimeFiltering = self.Signals.TimeFilteringEnabled
        if useTimeFiltering:
            if i == Sistem.BarSayisi - 1:
                s = ""
                s += f'  {startDateTime}\n'
                s += f'  {stopDateTime}\n'
                s += f'  {startDate}\n'
                s += f'  {stopDate}\n'
                s += f'  {startTime}\n'
                s += f'  {stopTime}\n'
                s += f'  {nowDateTime}\n'
                s += f'  {nowDate}\n'
                s += f'  {nowTime}\n'
                s += f'  FilterMode = {FilterMode}\n'
                s += '  CTrader::IslemZamanFiltresiUygula\n'
            if FilterMode == 0:
                IsTradeEnabled.value = True
                CheckResult.value = 0
            elif FilterMode == 1:
                if self.TimeUtils.check_bar_time_with(Sistem, i, startTime) >= 0 and self.TimeUtils.check_bar_time_with(Sistem, i, stopTime) < 0:
                    IsTradeEnabled.value = True
                    CheckResult.value = 0
                elif self.TimeUtils.check_bar_time_with(Sistem, i, startTime) < 0:
                    if not self.is_son_yon_f(Sistem):
                        IsPozKapatEnabled.value = True
                    CheckResult.value = -1
                elif self.TimeUtils.check_bar_time_with(Sistem, i, stopTime) >= 0:
                    if not self.is_son_yon_f(Sistem):
                        IsPozKapatEnabled.value = True
                    CheckResult.value = 1
            elif FilterMode == 2:
                if self.TimeUtils.check_bar_date_with(Sistem, i, startDate) >= 0 and self.TimeUtils.check_bar_date_with(Sistem, i, stopDate) < 0:
                    IsTradeEnabled.value = True
                    CheckResult.value = 0
                elif self.TimeUtils.check_bar_date_with(Sistem, i, startDate) < 0:
                    if not self.is_son_yon_f(Sistem):
                        IsPozKapatEnabled.value = True
                    CheckResult.value = -1
                elif self.TimeUtils.check_bar_date_with(Sistem, i, stopDate) >= 0:
                    if not self.is_son_yon_f(Sistem):
                        IsPozKapatEnabled.value = True
                    CheckResult.value = 1
            elif FilterMode == 3:
                if self.TimeUtils.check_bar_date_time_with(Sistem, i, startDateTime) >= 0 and self.TimeUtils.check_bar_date_time_with(Sistem, i, stopDateTime) < 0:
                    IsTradeEnabled.value = True
                    CheckResult.value = 0
                elif self.TimeUtils.check_bar_date_time_with(Sistem, i, startDateTime) < 0:
                    if not self.is_son_yon_f(Sistem):
                        IsPozKapatEnabled.value = True
                    CheckResult.value = -1
                elif self.TimeUtils.check_bar_date_time_with(Sistem, i, stopDateTime) >= 0:
                    if not self.is_son_yon_f(Sistem):
                        IsPozKapatEnabled.value = True
                    CheckResult.value = 1
            elif FilterMode == 4:
                if self.TimeUtils.check_bar_time_with(Sistem, i, startTime) >= 0:
                    IsTradeEnabled.value = True
                    CheckResult.value = 0
                elif self.TimeUtils.check_bar_time_with(Sistem, i, startTime) < 0:
                    if not self.is_son_yon_f(Sistem):
                        IsPozKapatEnabled.value = True
                    CheckResult.value = -1
            elif FilterMode == 5:
                if self.TimeUtils.check_bar_date_with(Sistem, i, startDate) >= 0:
                    IsTradeEnabled.value = True
                    CheckResult.value = 0
                elif self.TimeUtils.check_bar_date_with(Sistem, i, startDate) < 0:
                    if not self.is_son_yon_f(Sistem):
                        IsPozKapatEnabled.value = True
                    CheckResult.value = -1
            elif FilterMode == 6:
                if self.TimeUtils.check_bar_date_time_with(Sistem, i, startDateTime) >= 0:
                    IsTradeEnabled.value = True
                    CheckResult.value = 0
                elif self.TimeUtils.check_bar_date_time_with(Sistem, i, startDateTime) < 0:
                    if not self.is_son_yon_f(Sistem):
                        IsPozKapatEnabled.value = True
                    CheckResult.value = -1
        return 0
