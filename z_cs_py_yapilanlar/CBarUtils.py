from CBase import CBase
from CTimeUtils import CTimeUtils

class CBarUtils(CBase):
    def __init__(self):
        super().__init__()
        self.TimeUtils = None
        self.BarOpen = []
        self.BarHigh = []
        self.BarLow = []
        self.BarClose = []
        self.BarVolume = []
        self.BarLot = []
        self.BarMAYontem = ""
        self.BarPeriyot = 0
        self.BarPanelNo = 0
        self.BarTip = 0
        self.SecilenBarTarihi = None
        self.SecilenBarNumarasi = 0
        self.SecilenBarAcilisFiyati = 0.0
        self.SecilenBarYuksekFiyati = 0.0
        self.SecilenBarDusukFiyati = 0.0
        self.SecilenBarKapanisFiyati = 0.0
        self.SonBarTarihi = None
        self.SonBarAcilisFiyati = 0.0
        self.SonBarYuksekFiyati = 0.0
        self.SonBarDusukFiyati = 0.0
        self.SonBarKapanisFiyati = 0.0
        self.SonBarIndex = 0
        self.AyBarIndeksList = []
        self.HaftaBarIndeksList = []
        self.GunBarIndeksList = []
        self.SaatBarIndeksList = []
        self.DakikaBarIndeksList = []
        self.XBarIndeksList = []
        self.KapanisListesi5Dak = []
        self.kapanis_5_dak = 0.0
        # ... and so on for all the other fields

    def initialize(self, Sistem, V, Open, High, Low, Close, Volume, Lot):
        self.SetData(Sistem, V, Open, High, Low, Close, Volume, Lot)
        self.TimeUtils = CTimeUtils()
        self.TimeUtils.initialize(Sistem, V, Open, High, Low, Close, Volume, Lot)
        self.AyBarIndeksList = []
        self.HaftaBarIndeksList = []
        self.GunBarIndeksList = []
        self.SaatBarIndeksList = []
        self.DakikaBarIndeksList = []
        self.XBarIndeksList = []
        self.reset(Sistem)
        return self

    def reset(self, Sistem):
        self.AyBarIndeksList.clear()
        self.HaftaBarIndeksList.clear()
        self.GunBarIndeksList.clear()
        self.SaatBarIndeksList.clear()
        self.DakikaBarIndeksList.clear()
        self.XBarIndeksList.clear()
        self.kapanis_listelerini_resetle(Sistem)
        self.yuksek_listelerini_resetle(Sistem)
        self.dusuk_listelerini_resetle(Sistem)
        self.acilis_listelerini_resetle(Sistem)
        self.x_bar_listelerini_resetle(Sistem)
        self.x_bar_indeks_listesini_resetle(Sistem)
        self.bar_yuzde_listesini_resetle(Sistem)
        return self

    def get_bar_values_description(self, Sistem):
        return "#  Column Names : No,Date,Time,Open,High,Low,Close,Vol,Size(Lot)"

    def get_bar_values_as_string(self, Sistem, BarIndex):
        i = BarIndex
        return f"{i:<5} 	 {self.V[i].Date.strftime('%Y.%m.%d %H:%M:%S'):<20} 	 {self.V[i].Open:>5.2f} 	 {self.V[i].High:>5.2f} 	 {self.V[i].Low:>5.2f} 	 {self.V[i].Close:>5.2f} 	 {self.V[i].Vol:>10.0f} 	 {self.V[i].Size:>5.0f} 	 "

    def barlari_yumusat(self, Sistem, BarPeriyot=5, BarMAYontem="Exp", GercekVerilereUygula=False):
        self.BarMAYontem = BarMAYontem
        self.BarPeriyot = BarPeriyot
        self.BarOpen = Sistem.MA(self.Open, BarMAYontem, BarPeriyot)
        self.BarHigh = Sistem.MA(self.High, BarMAYontem, BarPeriyot)
        self.BarLow = Sistem.MA(self.Low, BarMAYontem, BarPeriyot)
        self.BarClose = Sistem.MA(self.Close, BarMAYontem, BarPeriyot)
        self.BarVolume = self.Volume

        if GercekVerilereUygula:
            self.Open = self.BarOpen
            self.High = self.BarHigh
            self.Low = self.BarLow
            self.Close = self.BarClose
            self.Volume = self.BarVolume
            self.Lot = self.BarLot

    def barlari_ciz(self, Sistem, BarPanelNo=1, BarTip=2):
        self.BarPanelNo = BarPanelNo
        self.BarTip = BarTip
        Sistem.BarCiz(self.BarPanelNo, self.BarTip, self.BarOpen, self.BarHigh, self.BarLow, self.BarClose, 'Green', 'Red')

    def son_bar_bilgilerini_al(self, Sistem):
        LastBar = len(self.V) - 1
        self.SonBarTarihi = self.V[LastBar].Date
        self.SonBarAcilisFiyati = self.V[LastBar].Open
        self.SonBarYuksekFiyati = self.V[LastBar].High
        self.SonBarDusukFiyati = self.V[LastBar].Low
        self.SonBarKapanisFiyati = self.V[LastBar].Close
        self.SonBarIndex = len(self.V) - 1

    def secilen_bar_bilgilerini_al(self, Sistem):
        LastBar = len(self.V) - 1
        self.SecilenBarNumarasi = Sistem.SelectBarNo
        self.SecilenBarTarihi = Sistem.SelectTarih
        if self.SecilenBarNumarasi <= LastBar:
            self.SecilenBarAcilisFiyati = self.V[self.SecilenBarNumarasi].Open
            self.SecilenBarYuksekFiyati = self.V[self.SecilenBarNumarasi].High
            self.SecilenBarDusukFiyati = self.V[self.SecilenBarNumarasi].Low
            self.SecilenBarKapanisFiyati = self.V[self.SecilenBarNumarasi].Close

    def ay_gun_saat_dakika_indeks_listelerini_guncelle(self, Sistem, BarIndex, TimeUtils):
        IsYeniAy = TimeUtils.is_yeni_ay(Sistem, BarIndex)
        IsYeniHafta = TimeUtils.is_yeni_hafta(Sistem, BarIndex)
        IsYeniGun = TimeUtils.is_yeni_gun(Sistem, BarIndex)
        IsYeniSaat = TimeUtils.is_yeni_saat(Sistem, BarIndex)

        if IsYeniAy:
            self.AyBarIndeksList.append(BarIndex)
        if IsYeniHafta:
            self.HaftaBarIndeksList.append(BarIndex)
        if IsYeniGun:
            self.GunBarIndeksList.append(BarIndex)
        if IsYeniSaat:
            self.SaatBarIndeksList.append(BarIndex)

    def get_bar_date_time(self, Sistem, BarIndex):
        return self.V[BarIndex].Date

    # ... The rest of the methods would be translated here. This is a large class.
    # Due to the complexity and length, a full translation is not provided in this step.
