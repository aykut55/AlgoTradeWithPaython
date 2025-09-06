from CBase import CBase
import random
import datetime
import os
from collections import OrderedDict

class CSystemWrapper(CBase):
    def __init__(self):
        self.GrafikSembol = ""
        self.GrafikPeriyot = ""
        self.SistemAdi = ""

        self.myVarlik = None
        self.myTrader = None
        self.myUtils = None
        self.myTimeUtils = None
        self.myBarUtils = None
        self.myFileUtils = None
        self.myExcelUtils = None
        self.mySharedMemory = None
        self.myConfig = None
        self.myIndicators = None

        self.HisseSayisi = 0
        self.KontratSayisi = 10
        self.KomisyonCarpan = 0.0
        self.VarlikAdedCarpani = 1

        self.InputsDir = ""
        self.OutputsDir = ""
        self.ParamsInputFileName = ""
        self.IstatistiklerOutputFileName = ""
        self.IstatistiklerOptOutputFileName = ""

        self.bUseParamsFromInputFile = False
        self.bOptEnabled = False
        self.bIdealGetiriHesapla = True
        self.bIstatistikleriHesapla = True
        self.bIstatistikleriEkranaYaz = True
        self.bGetiriIstatistikleriEkranaYaz = True
        self.bIstatistikleriDosyayaYaz = True
        self.bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz = True
        self.bOptimizasyonIstatistikleriniDosyayaYaz = True
        self.bSinyalleriEkranaCiz = True
        self.CurrentRunIndex = 0
        self.TotalRunCount = 1

        self.Al = False
        self.Sat = False
        self.FlatOl = False
        self.PasGec = False
        self.KarAl = False
        self.ZararKes = False

        self.InputParamsCount = 50
        self.InputParams = ["" for _ in range(50)]

    def __del__(self):
        pass

    def create_modules(self, Sistem, Lib):
        self.myVarlik = Lib.GetVarlik(Sistem)
        self.myTrader = Lib.GetTrader(Sistem)
        self.myUtils = Lib.GetUtils(Sistem)
        self.myTimeUtils = Lib.GetTimeUtils(Sistem)
        self.myBarUtils = Lib.GetBarUtils(Sistem)
        self.myFileUtils = Lib.GetFileUtils(Sistem)
        self.myExcelUtils = Lib.GetExcelFileHandler(Sistem)
        self.mySharedMemory = Lib.GetSharedMemory(Sistem)
        self.myConfig = Lib.GetConfig(Sistem)
        self.myIndicators = Lib.GetIndicatorManager(Sistem)
        return self

    def initialize(self, Sistem, V, Open, High, Low, Close, Volume, Lot):
        self.GrafikSembol = Sistem.Sembol
        self.GrafikPeriyot = Sistem.Periyot
        self.SistemAdi = Sistem.Name

        self.set_data(Sistem, V, Open, High, Low, Close, Volume, Lot)

        self.myVarlik.initialize(Sistem)
        self.myTrader.initialize(Sistem, V, Open, High, Low, Close, Volume, Lot, self.myVarlik)
        self.myUtils.initialize(Sistem)
        self.myTimeUtils.initialize(Sistem, V, Open, High, Low, Close, Volume, Lot)
        self.myBarUtils.initialize(Sistem, V, Open, High, Low, Close, Volume, Lot)
        self.myIndicators.initialize(Sistem, V, Open, High, Low, Close, Volume, Lot)

        return self

    def reset(self, Sistem):
        self.myVarlik.reset(Sistem)
        self.myTrader.reset(Sistem)
        self.myUtils.reset(Sistem)
        self.myTimeUtils.reset(Sistem)
        self.myBarUtils.reset(Sistem)
        self.myIndicators.reset(Sistem)

        for i in range(self.InputParamsCount):
            self.InputParams[i] = ""

        return self

    def get_varlik(self, Sistem):
        return self.myVarlik

    def get_trader(self, Sistem):
        return self.myTrader

    def get_utils(self, Sistem):
        return self.myUtils

    def get_time_utils(self, Sistem):
        return self.myTimeUtils

    def get_bar_utils(self, Sistem):
        return self.myBarUtils

    def get_file_utils(self, Sistem):
        return self.myFileUtils

    def get_excel_file_handler(self, Sistem):
        return self.myExcelUtils

    def get_shared_memory(self, Sistem):
        return self.mySharedMemory

    def get_config(self, Sistem):
        return self.myConfig

    def get_indicator_manager(self, Sistem):
        return self.myIndicators

    def initialize_params_with_defaults(self, Sistem):
        self.HisseSayisi = 0
        self.KontratSayisi = 10
        self.KomisyonCarpan = 0.0
        self.VarlikAdedCarpani = 1

        self.InputsDir = "Aykut/Exports/"
        self.OutputsDir = "Aykut/Exports/"
        self.ParamsInputFileName = self.InputsDir + self.SistemAdi + "_params.txt"
        self.IstatistiklerOutputFileName = self.OutputsDir + "Istatistikler.csv"
        self.IstatistiklerOptOutputFileName = self.OutputsDir + "IstatistiklerOpt.csv"

        self.bUseParamsFromInputFile = False
        self.bOptEnabled = False
        self.bIdealGetiriHesapla = True
        self.bIstatistikleriHesapla = True
        self.bIstatistikleriEkranaYaz = True
        self.bGetiriIstatistikleriEkranaYaz = True
        self.bIstatistikleriDosyayaYaz = True
        self.bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz = False
        self.bOptimizasyonIstatistikleriniDosyayaYaz = False
        self.bSinyalleriEkranaCiz = True

        self.CurrentRunIndex = 0
        self.TotalRunCount = 1

        self.myVarlik.set_kontrat_params_fx_ons_altin_micro(Sistem, KontratSayisi=1, VarlikAdedCarpani=1).set_komisyon_params(Sistem, KomisyonCarpan=0.0)

        self.myTrader.Signals.KarAlEnabled = False
        self.myTrader.Signals.ZararKesEnabled = False
        self.myTrader.Signals.KarAlindi = False
        self.myTrader.Signals.ZararKesildi = False
        self.myTrader.Signals.FlatOlundu = False
        self.myTrader.Signals.PozAcilabilir = False
        self.myTrader.Signals.PozAcildi = False
        self.myTrader.Signals.PozKapatilabilir = False
        self.myTrader.Signals.PozKapatildi = False
        self.myTrader.Signals.PozAcilabilirAlis = False
        self.myTrader.Signals.PozAcilabilirSatis = False
        self.myTrader.Signals.PozAcildiAlis = False
        self.myTrader.Signals.PozAcildiSatis = False
        self.myTrader.Signals.GunSonuPozKapatEnabled = False
        self.myTrader.Signals.GunSonuPozKapatildi = False
        self.myTrader.Signals.TimeFilteringEnabled = False

        return self

    def set_params_for_single_run(self, Sistem, IdealGetiriHesapla=True, IstatistikleriHesapla=True,
                                 IstatistikleriEkranaYaz=True, GetiriIstatistikleriEkranaYaz=True,
                                 IstatistikleriDosyayaYaz=True, SinyalleriEkranaCiz=True):
        self.bIdealGetiriHesapla = IdealGetiriHesapla
        self.bIstatistikleriHesapla = IstatistikleriHesapla
        self.bIstatistikleriEkranaYaz = IstatistikleriEkranaYaz
        self.bGetiriIstatistikleriEkranaYaz = GetiriIstatistikleriEkranaYaz
        self.bIstatistikleriDosyayaYaz = IstatistikleriDosyayaYaz
        self.bSinyalleriEkranaCiz = SinyalleriEkranaCiz

        return self

    def start(self, Sistem):
        self.myTimeUtils.start_timer(Sistem)
        self.myTrader.start(Sistem)

    def emirleri_resetle(self, Sistem, BarIndex):
        self.Al = False
        self.Sat = False
        self.FlatOl = False
        self.PasGec = False
        self.KarAl = False
        self.ZararKes = False

    def emir_oncesi_dongu_foksiyonlarini_calistir(self, Sistem, BarIndex):
        i = BarIndex
        self.myTrader.dongu_basi_degiskenleri_resetle(Sistem, i)
        self.myTrader.dongu_basi_degiskenleri_guncelle(Sistem, i)
        if i < 1:
            return
        self.myTrader.anlik_kar_zarar_hesapla(Sistem, i)
        self.myTrader.emirleri_resetle(Sistem, i)
        is_yeni_gun = self.V[i].Date.day != self.V[i - 1].Date.day
        if is_yeni_gun:
            Sistem.DikeyCizgiEkle(i, "DimGray", 2, 2)
        is_yeni_saat = self.V[i].Date.hour != self.V[i - 1].Date.hour
        if self.myTrader.Signals.GunSonuPozKapatildi:
            self.myTrader.Signals.GunSonuPozKapatildi = False
        if self.myTrader.Signals.KarAlindi or self.myTrader.Signals.ZararKesildi or self.myTrader.Signals.FlatOlundu:
            self.myTrader.Signals.KarAlindi = False
            self.myTrader.Signals.ZararKesildi = False
            self.myTrader.Signals.FlatOlundu = False
            self.myTrader.Signals.PozAcilabilir = False
        if not self.myTrader.Signals.PozAcilabilir:
            self.myTrader.Signals.PozAcilabilir = True
            self.myTrader.Signals.PozAcildi = False

    def emirleri_setle(self, Sistem, BarIndex, Al, Sat, FlatOl=False, PasGec=False, KarAl=False, ZararKes=False):
        i = BarIndex
        self.Al = Al
        self.Sat = Sat
        self.FlatOl = FlatOl
        self.PasGec = PasGec
        self.KarAl = KarAl
        self.ZararKes = ZararKes

    def emir_sonrasi_dongu_foksiyonlarini_calistir(self, Sistem, BarIndex):
        i = BarIndex
        self.myTrader.emirleri_setle(Sistem, i, self.Al, self.Sat, self.FlatOl, self.PasGec, self.KarAl, self.ZararKes)
        self.myTrader.Signals.GunSonuPozKapatildi = self.myTrader.gun_sonu_poz_kapat(Sistem, i, self.myTrader.Signals.GunSonuPozKapatEnabled)
        self.myTrader.emirleri_uygula(Sistem, i)
        if not self.myTrader.Signals.KarAlindi and self.myTrader.Signals.KarAl:
            self.myTrader.Signals.KarAlindi = True
        if not self.myTrader.Signals.ZararKesildi and self.myTrader.Signals.ZararKes:
            self.myTrader.Signals.ZararKesildi = True
        if not self.myTrader.Signals.FlatOlundu and self.myTrader.Signals.FlatOl:
            self.myTrader.Signals.FlatOlundu = True
        self.myTrader.sistem_yon_listesini_guncelle(Sistem, i)
        self.myTrader.sistem_seviye_listesini_guncelle(Sistem, i)
        self.myTrader.sinyal_listesini_guncelle(Sistem, i)
        self.myTrader.islem_listesini_guncelle(Sistem, i)
        self.myTrader.komisyon_listesini_guncelle(Sistem, i)
        self.myTrader.bakiye_listesini_guncelle(Sistem, i)
        self.myTrader.dongu_sonu_degiskenleri_setle(Sistem, i)

    def stop(self, Sistem):
        self.myTrader.stop(Sistem)
        self.myTimeUtils.stop_timer(Sistem)

    def hesaplamalari_yap(self, Sistem):
        if self.bIdealGetiriHesapla:
            self.myTrader.ideal_getiri_hesapla(Sistem)
        if self.bIstatistikleriHesapla:
            self.myTrader.istatistikleri_hesapla(Sistem)

    def sonuclari_ekranda_goster(self, Sistem):
        if self.bIstatistikleriEkranaYaz:
            self.myTrader.istatistikleri_ekrana_yaz(Sistem, 1)
        if self.bGetiriIstatistikleriEkranaYaz:
            self.myTrader.getiri_istatistikleri_ekrana_yaz(Sistem, 2)

    def sonuclari_dosyaya_yaz(self, Sistem):
        if self.bIstatistikleriDosyayaYaz:
            self.myTrader.istatistikleri_dosyaya_yaz(Sistem, self.IstatistiklerOutputFileName)

    def sinyalleri_ekrana_ciz(self, Sistem, k):
        k_ = 0
        if self.bSinyalleriEkranaCiz:
            k_ = self.myTrader.sinyalleri_ekrana_ciz(Sistem, k)
        return k_

    def aciklamalari_temizle(self, Sistem, k):
        for i in range(k, 50):
            Sistem.Cizgiler[i].Aciklama = "---"

    def rastgele_renk_ata(self, Sistem, k):
        for i in range(k, 50):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            Sistem.Cizgiler[i].Renk = (255, r, g, b)

    def set_params_for_optimizasyon(self, Sistem, OptEnabled=True, IdealGetiriHesapla=True, IstatistikleriHesapla=True,
                                    IstatistikleriEkranaYaz=False, GetiriIstatistikleriEkranaYaz=False,
                                    IstatistikleriDosyayaYaz=True, SinyalleriEkranaCiz=False,
                                    OptimizasyonIstatistiklerininBasliklariniDosyayaYaz=True,
                                    OptimizasyonIstatistikleriniDosyayaYaz=True):
        self.bOptEnabled = OptEnabled
        self.bIdealGetiriHesapla = IdealGetiriHesapla
        self.bIstatistikleriHesapla = IstatistikleriHesapla
        self.bIstatistikleriEkranaYaz = IstatistikleriEkranaYaz
        self.bGetiriIstatistikleriEkranaYaz = GetiriIstatistikleriEkranaYaz
        self.bIstatistikleriDosyayaYaz = IstatistikleriDosyayaYaz
        self.bSinyalleriEkranaCiz = SinyalleriEkranaCiz
        self.bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz = OptimizasyonIstatistiklerininBasliklariniDosyayaYaz
        self.bOptimizasyonIstatistikleriniDosyayaYaz = OptimizasyonIstatistikleriniDosyayaYaz
        return self

    def optimizasyon_istatistiklerinin_basliklarini_dosyaya_yaz(self, Sistem):
        if self.bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz:
            self.myTrader.optimizasyon_istatistiklerinin_basliklarini_dosyaya_yaz(Sistem, self.IstatistiklerOptOutputFileName)

    def optimizasyon_istatistiklerini_dosyaya_yaz(self, Sistem, CurrentRunIndex, TotalRunCount):
        if self.bOptimizasyonIstatistikleriniDosyayaYaz:
            self.myTrader.optimizasyon_istatistiklerini_dosyaya_yaz(Sistem, self.IstatistiklerOptOutputFileName, CurrentRunIndex, TotalRunCount)

    def set_current_index(self, Sistem, CurrentRunIndex):
        self.CurrentRunIndex = CurrentRunIndex

    def set_total_run_count(self, Sistem, TotalRunCount):
        self.TotalRunCount = TotalRunCount

    def set_input_params(self, Sistem, Index, Value):
        self.InputParams[Index] = Value

    def read_params_from_file(self, Sistem, FileName):
        if os.path.exists(FileName):
            with open(FileName, 'r') as f:
                readLines = f.readlines()
            self.bUseParamsFromInputFile = int(readLines[0]) == 1
            if self.bUseParamsFromInputFile:
                self.CurrentRunIndex = int(readLines[1])
                self.TotalRunCount = int(readLines[2])
                self.bOptEnabled = int(readLines[3]) == 1
                self.bIdealGetiriHesapla = int(readLines[4]) == 1
                self.bIstatistikleriHesapla = int(readLines[5]) == 1
                self.bIstatistikleriEkranaYaz = int(readLines[6]) == 1
                self.bGetiriIstatistikleriEkranaYaz = int(readLines[7]) == 1
                self.bIstatistikleriDosyayaYaz = int(readLines[8]) == 1
                self.bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz = int(readLines[9]) == 1
                self.bOptimizasyonIstatistikleriniDosyayaYaz = int(readLines[10]) == 1
                self.bSinyalleriEkranaCiz = int(readLines[11]) == 1
                self.ParamsInputFileName = readLines[12].strip()
                self.IstatistiklerOutputFileName = readLines[13].strip()
                self.IstatistiklerOptOutputFileName = readLines[14].strip()
            if self.bUseParamsFromInputFile:
                k = 0
                self.InputParams[k] = readLines[15].strip()
                k += 1
                self.InputParams[k] = readLines[16].strip()
                k += 1
                self.InputParams[k] = readLines[17].strip()
                k += 1
                for i in range(k, self.InputParamsCount):
                    self.InputParams[i] = ""
        return self

    def update_sistem_parametreleri(self, Sistem):
        for i in range(20):
            Sistem.Parametreler[i] = self.InputParams[i]
        return self

    def write_params_to_file(self, Sistem, FileName, bUseParamsFromInputFile, CurrentRunIndex, TotalRunCount,
                             bOptEnabled, bIdealGetiriHesapla, bIstatistikleriHesapla, bIstatistikleriEkranaYaz,
                             bGetiriIstatistikleriEkranaYaz, bIstatistikleriDosyayaYaz,
                             bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz,
                             bOptimizasyonIstatistikleriniDosyayaYaz, bSinyalleriEkranaCiz, ParamsInputFileName,
                             IstatistiklerOutputFileName, IstatistiklerOptOutputFileName):
        strOne = "1"
        strZero = "0"
        writeLines = []
        writeLines.append(strOne if bUseParamsFromInputFile else strZero)
        writeLines.append(str(CurrentRunIndex))
        writeLines.append(str(TotalRunCount))
        writeLines.append(strOne if bOptEnabled else strZero)
        writeLines.append(strOne if bIdealGetiriHesapla else strZero)
        writeLines.append(strOne if bIstatistikleriHesapla else strZero)
        writeLines.append(strOne if bIstatistikleriEkranaYaz else strZero)
        writeLines.append(strOne if bGetiriIstatistikleriEkranaYaz else strZero)
        writeLines.append(strOne if bIstatistikleriDosyayaYaz else strZero)
        writeLines.append(strOne if bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz else strZero)
        writeLines.append(strOne if bOptimizasyonIstatistikleriniDosyayaYaz else strZero)
        writeLines.append(strOne if bSinyalleriEkranaCiz else strZero)
        writeLines.append(ParamsInputFileName)
        writeLines.append(IstatistiklerOutputFileName)
        writeLines.append(IstatistiklerOptOutputFileName)
        with open(FileName, 'w', encoding='utf-8') as f:
            f.write('\n'.join(writeLines))
        return self

    def create_optimizasyon_log_string(self, Sistem, Counter, TotalCount, OptParam1=None, OptParam2=None,
                                       OptParam3=None, OptParam4=None, OptParam5=None, OptParam6=None):
        IstatistiklerNew = self.myTrader.Statistics.IstatistiklerNew
        delimiter = ";"
        strList = []
        strList.append(f"{Counter} {delimiter} ")
        strList.append(f"{TotalCount} {delimiter} ")
        if OptParam1 is not None:
            strList.append(f"{OptParam1} {delimiter} ")
        if OptParam2 is not None:
            strList.append(f"{OptParam2} {delimiter} ")
        if OptParam3 is not None:
            strList.append(f"{OptParam3} {delimiter} ")
        if OptParam4 is not None:
            strList.append(f"{OptParam4} {delimiter} ")
        if OptParam5 is not None:
            strList.append(f"{OptParam5} {delimiter} ")
        if OptParam6 is not None:
            strList.append(f"{OptParam6} {delimiter} ")
        strList.append(f"{IstatistiklerNew['GetiriKzSistem']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['GetiriKz']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['GetiriKzNet']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['ProfitFactor']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['ProfitFactorSistem']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['KarliIslemOrani']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['IslemSayisi']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['KomisyonIslemSayisi']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['KomisyonFiyat']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['KomisyonFiyatYuzde']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['IlkBakiyeFiyat']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['BakiyeFiyat']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['GetiriFiyat']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['GetiriFiyatYuzde']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['BakiyeFiyatNet']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['GetiriFiyatNet']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['GetiriFiyatYuzdeNet']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['MinBakiyeFiyat']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['MaxBakiyeFiyat']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['MinBakiyeFiyatYuzde']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['MaxBakiyeFiyatYuzde']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['MaxKarFiyat']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['MaxZararFiyat']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['MaxKarPuan']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['MaxZararPuan']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['KazandiranIslemSayisi']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['KaybettirenIslemSayisi']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['ToplamKarFiyat']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['ToplamZararFiyat']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['NetKarFiyat']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['ToplamKarPuan']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['ToplamZararPuan']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['NetKarPuan']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['GetiriMaxDD']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['GetiriMaxDDTarih']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['GetiriMaxDDSaat']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['GetiriMaxKayip']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['OrtAylikIslemSayisi']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['OrtHaftalikIslemSayisi']} {delimiter} ")
        strList.append(f"{IstatistiklerNew['OrtGunlukIslemSayisi']} {delimiter} ")
        strList.append(f"{datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')} {delimiter} ")
        return "".join(f"  {s}" for s in strList).strip()

    def islem_zaman_filtresi_uygula(self, Sistem, BarIndex, Al, Sat, FlatOl, PasGec, KarAl, ZararKes, FilterMode=3):
        useTimeFiltering = self.myTrader.Signals.TimeFilteringEnabled
        if useTimeFiltering:
            checkResult = 0
            isTradeEnabled = False
            isPozKapatEnabled = False
            self.myTrader.islem_zaman_filtresi_uygula(Sistem, BarIndex, FilterMode, isTradeEnabled, isPozKapatEnabled, checkResult)
            if not isTradeEnabled:
                Al = False
            if not isTradeEnabled:
                Sat = False
            if isPozKapatEnabled:
                FlatOl = True
        return Al, Sat, FlatOl, PasGec, KarAl, ZararKes

    def islem_zaman_filtresi_uygula(self, Sistem, BarIndex, FilterMode=3):
        useTimeFiltering = self.myTrader.Signals.TimeFilteringEnabled
        if useTimeFiltering:
            checkResult = 0
            isTradeEnabled = False
            isPozKapatEnabled = False
            self.myTrader.islem_zaman_filtresi_uygula(Sistem, BarIndex, FilterMode, isTradeEnabled, isPozKapatEnabled, checkResult)
            if not isTradeEnabled:
                self.Al = False
            if not isTradeEnabled:
                self.Sat = False
            if isPozKapatEnabled:
                self.FlatOl = True

    def islem_zaman_filtresi_uygula_bunu_kullanma(self, Sistem, BarIndex, mode1=0):
        i = BarIndex
        mySystem = self
        checkResult = 0
        isTradeEnabled = False
        isPozKapatEnabled = False
        startDateTime = mySystem.get_trader(Sistem).StartDateTimeStr
        stopDateTime = mySystem.get_trader(Sistem).StopDateTimeStr
        startDate = mySystem.get_trader(Sistem).StartDateStr
        stopDate = mySystem.get_trader(Sistem).StopDateStr
        startTime = mySystem.get_trader(Sistem).StartTimeStr
        stopTime = mySystem.get_trader(Sistem).StopTimeStr
        nowDateTime = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        nowDate = datetime.datetime.now().strftime("%d.%m.%Y")
        nowTime = datetime.datetime.now().strftime("%H:%M:%S")
        useTimeFiltering = mySystem.get_trader(Sistem).Signals.TimeFilteringEnabled
        mode = mode1
        if useTimeFiltering and 1 == 0:
            if i == Sistem.BarSayisi - 1:
                s = ""
                s += f"  {startDateTime}\n"
                s += f"  {stopDateTime}\n"
                s += f"  {startDate}\n"
                s += f"  {stopDate}\n"
                s += f"  {startTime}\n"
                s += f"  {stopTime}\n"
                s += f"  {nowDateTime}\n"
                s += f"  {nowDate}\n"
                s += f"  {nowTime}\n"
                s += f"  FilterMode = {mode}\n"
                s += "  CSystemWrapper::IslemZamanFiltresiUygula\n"
                Sistem.Mesaj(s)
            if mode == 0:
                pass
            elif mode == 1:
                if mySystem.get_time_utils(Sistem).check_bar_time_with(Sistem, i, startTime) >= 0 and mySystem.get_time_utils(Sistem).check_bar_time_with(Sistem, i, stopTime) < 0:
                    isTradeEnabled = True
                elif mySystem.get_time_utils(Sistem).check_bar_time_with(Sistem, i, startTime) < 0:
                    if not mySystem.get_trader(Sistem).is_son_yon_f(Sistem):
                        isPozKapatEnabled = True
                elif mySystem.get_time_utils(Sistem).check_bar_time_with(Sistem, i, stopTime) >= 0:
                    if not mySystem.get_trader(Sistem).is_son_yon_f(Sistem):
                        isPozKapatEnabled = True
                if not isTradeEnabled:
                    self.Al = False
                if not isTradeEnabled:
                    self.Sat = False
                if isPozKapatEnabled:
                    self.FlatOl = True
            elif mode == 2:
                if mySystem.get_time_utils(Sistem).check_bar_date_with(Sistem, i, startDate) >= 0 and mySystem.get_time_utils(Sistem).check_bar_date_with(Sistem, i, stopDate) < 0:
                    isTradeEnabled = True
                elif mySystem.get_time_utils(Sistem).check_bar_date_with(Sistem, i, startDate) < 0:
                    if not mySystem.get_trader(Sistem).is_son_yon_f(Sistem):
                        isPozKapatEnabled = True
                elif mySystem.get_time_utils(Sistem).check_bar_date_with(Sistem, i, stopDate) >= 0:
                    if not mySystem.get_trader(Sistem).is_son_yon_f(Sistem):
                        isPozKapatEnabled = True
                if not isTradeEnabled:
                    self.Al = False
                if not isTradeEnabled:
                    self.Sat = False
                if isPozKapatEnabled:
                    self.FlatOl = True
            elif mode == 3:
                if mySystem.get_time_utils(Sistem).check_bar_date_time_with(Sistem, i, startDateTime) >= 0 and mySystem.get_time_utils(Sistem).check_bar_date_time_with(Sistem, i, stopDateTime) < 0:
                    isTradeEnabled = True
                elif mySystem.get_time_utils(Sistem).check_bar_date_time_with(Sistem, i, startDateTime) < 0:
                    if not mySystem.get_trader(Sistem).is_son_yon_f(Sistem):
                        isPozKapatEnabled = True
                elif mySystem.get_time_utils(Sistem).check_bar_date_time_with(Sistem, i, stopDateTime) >= 0:
                    if not mySystem.get_trader(Sistem).is_son_yon_f(Sistem):
                        isPozKapatEnabled = True
                if not isTradeEnabled:
                    self.Al = False
                if not isTradeEnabled:
                    self.Sat = False
                if isPozKapatEnabled:
                    self.FlatOl = True
            elif mode == 4:
                if mySystem.get_time_utils(Sistem).check_bar_time_with(Sistem, i, startTime) >= 0:
                    isTradeEnabled = True
                elif mySystem.get_time_utils(Sistem).check_bar_time_with(Sistem, i, startTime) < 0:
                    if not mySystem.get_trader(Sistem).is_son_yon_f(Sistem):
                        isPozKapatEnabled = True
                if not isTradeEnabled:
                    self.Al = False
                if not isTradeEnabled:
                    self.Sat = False
                if isPozKapatEnabled:
                    self.FlatOl = True
            elif mode == 5:
                if mySystem.get_time_utils(Sistem).check_bar_date_with(Sistem, i, startDate) >= 0:
                    isTradeEnabled = True
                elif mySystem.get_time_utils(Sistem).check_bar_date_with(Sistem, i, startDate) < 0:
                    if not mySystem.get_trader(Sistem).is_son_yon_f(Sistem):
                        isPozKapatEnabled = True
                if not isTradeEnabled:
                    self.Al = False
                if not isTradeEnabled:
                    self.Sat = False
                if isPozKapatEnabled:
                    self.FlatOl = True
            elif mode == 6:
                if mySystem.get_time_utils(Sistem).check_bar_date_time_with(Sistem, i, startDateTime) >= 0:
                    isTradeEnabled = True
                elif mySystem.get_time_utils(Sistem).check_bar_date_time_with(Sistem, i, startDateTime) < 0:
                    if not mySystem.get_trader(Sistem).is_son_yon_f(Sistem):
                        isPozKapatEnabled = True
                if not isTradeEnabled:
                    self.Al = False
                if not isTradeEnabled:
                    self.Sat = False
                if isPozKapatEnabled:
                    self.FlatOl = True
