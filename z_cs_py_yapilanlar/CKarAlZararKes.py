from CUtils import CUtils

class CKarAlZararKes:
    def __init__(self):
        self.Trader = None

    def __del__(self):
        pass

    def initialize(self, Sistem, Trader):
        self.Trader = Trader
        return self

    def reset(self, Sistem):
        return self

    def kar_al_yuzde_hesapla(self, Sistem, BarIndex, KarAlYuzdesi, Ref=None):
        if Ref is None:
            Ref = self.Trader.Close
        
        result = 0
        i = BarIndex

        if self.Trader.Flags.KarAlYuzdeHesaplaEnabled:
            self.Trader.Lists.KarAlList[i] = Sistem.KarAlYuzde(KarAlYuzdesi, i)
            if self.Trader.Lists.KarAlList[i] == 0:
                self.Trader.Lists.KarAlList[i] = Ref[i]

            if self.Trader.is_son_yon_a(Sistem) and (Ref[i] > self.Trader.Lists.KarAlList[i]):
                result = 1
            elif self.Trader.is_son_yon_s(Sistem) and (Ref[i] < self.Trader.Lists.KarAlList[i]):
                result = -1
        return result

    def izleyen_stop_yuzde_hesapla(self, Sistem, BarIndex, IzleyenStopYuzdesi, Ref=None):
        if Ref is None:
            Ref = self.Trader.Close

        result = 0
        i = BarIndex

        if self.Trader.Flags.IzleyenStopYuzdeHesaplaEnabled:
            self.Trader.Lists.IzleyenStopList[i] = Sistem.IzleyenStopYuzde(IzleyenStopYuzdesi, i)
            if self.Trader.Lists.IzleyenStopList[i] == 0:
                self.Trader.Lists.IzleyenStopList[i] = Ref[i]

            if self.Trader.is_son_yon_a(Sistem) and (Ref[i] < self.Trader.Lists.IzleyenStopList[i]):
                result = 1
            elif self.Trader.is_son_yon_s(Sistem) and (Ref[i] > self.Trader.Lists.IzleyenStopList[i]):
                result = -1
        return result

    def son_fiyata_gore_kar_al_yuzde_hesapla(self, Sistem, BarIndex, KarAlYuzdesi=2.0):
        result = 0
        i = BarIndex
        if self.Trader.Flags.KarAlYuzdeHesaplaEnabled:
            if self.Trader.is_son_yon_a(Sistem) and (self.Trader.Close[i] > self.Trader.Signals.SonFiyat * (1.0 + KarAlYuzdesi * 0.01)):
                result = 1
            elif self.Trader.is_son_yon_s(Sistem) and (self.Trader.Close[i] < self.Trader.Signals.SonFiyat * (1.0 - KarAlYuzdesi * 0.01)):
                result = -1
        return result

    def son_fiyata_gore_zarar_kes_yuzde_hesapla(self, Sistem, BarIndex, ZararKesYuzdesi=-1.0):
        result = 0
        i = BarIndex
        ZararKesYuzdesi_ = -1.0 * ZararKesYuzdesi
        if self.Trader.Flags.ZararKesYuzdeHesaplaEnabled:
            if self.Trader.is_son_yon_a(Sistem) and (self.Trader.Close[i] < self.Trader.Signals.SonFiyat * (1.0 - ZararKesYuzdesi_ * 0.01)):
                result = 1
            elif self.Trader.is_son_yon_s(Sistem) and (self.Trader.Close[i] > self.Trader.Signals.SonFiyat * (1.0 + ZararKesYuzdesi_ * 0.01)):
                result = -1
        return result

    def son_fiyata_gore_kar_al_yuzde_hesapla_seviyeli(self, Sistem, BarIndex, SeviyeBas=2, SeviyeSon=10, Carpan=0.01):
        result = 0
        i = BarIndex
        if self.Trader.Flags.KarAlYuzdeHesaplaEnabled:
            myUtils = CUtils()
            _karAl = False
            if self.Trader.is_son_yon_a(Sistem):
                for m in range(SeviyeBas, SeviyeSon):
                    _karAl = _karAl or myUtils.asagi_kesti(Sistem, i, self.Trader.Close, self.Trader.Signals.SonFiyat * (1.0 + m * Carpan))
                    if _karAl:
                        break
                if _karAl: result = 1
            elif self.Trader.is_son_yon_s(Sistem):
                for m in range(SeviyeBas, SeviyeSon):
                    _karAl = _karAl or myUtils.yukari_kesti(Sistem, i, self.Trader.Close, self.Trader.Signals.SonFiyat * (1.0 - m * Carpan))
                    if _karAl:
                        break
                if _karAl: result = -1
        return result

    def son_fiyata_gore_zarar_kes_yuzde_hesapla_seviyeli(self, Sistem, BarIndex, SeviyeBas=-2, SeviyeSon=-10, Carpan=0.01):
        result = 0
        i = BarIndex
        SeviyeBas_ = -1 * SeviyeBas
        SeviyeSon_ = -1 * SeviyeSon
        if self.Trader.Flags.ZararKesYuzdeHesaplaEnabled:
            myUtils = CUtils()
            _zararKes = False
            if self.Trader.is_son_yon_a(Sistem):
                for m in range(SeviyeBas_, SeviyeSon_):
                    _zararKes = _zararKes or myUtils.asagi_kesti(Sistem, i, self.Trader.Close, self.Trader.Signals.SonFiyat * (1.0 - m * Carpan))
                    if _zararKes:
                        break
                if _zararKes: result = 1
            elif self.Trader.is_son_yon_s(Sistem):
                for m in range(SeviyeBas_, SeviyeSon_):
                    _zararKes = _zararKes or myUtils.yukari_kesti(Sistem, i, self.Trader.Close, self.Trader.Signals.SonFiyat * (1.0 + m * Carpan))
                    if _zararKes:
                        break
                if _zararKes: result = -1
        return result

    def son_fiyata_gore_kar_al_seviye_hesapla(self, Sistem, BarIndex, KarAlSeviyesi=2000.0):
        result = 0
        i = BarIndex
        if self.Trader.Flags.KarAlSeviyeHesaplaEnabled:
            myUtils = CUtils()
            result = 1 if myUtils.yukari_kesti(Sistem, i, self.Trader.Lists.KarZararFiyatList, KarAlSeviyesi) else 0
        return result

    def son_fiyata_gore_zarar_kes_seviye_hesapla(self, Sistem, BarIndex, ZararKesSeviyesi=-1000.0):
        result = 0
        i = BarIndex
        if self.Trader.Flags.ZararKesSeviyeHesaplaEnabled:
            myUtils = CUtils()
            result = 1 if myUtils.asagi_kesti(Sistem, i, self.Trader.Lists.KarZararFiyatList, ZararKesSeviyesi) else 0
        return result

    def son_fiyata_gore_kar_al_seviye_hesapla_seviyeli(self, Sistem, BarIndex, SeviyeBas=5, SeviyeSon=50, Carpan=1000):
        result = 0
        i = BarIndex
        if self.Trader.Flags.KarAlSeviyeHesaplaEnabled:
            myUtils = CUtils()
            _karAl = False
            for m in range(SeviyeBas, SeviyeSon):
                _karAl = _karAl or myUtils.asagi_kesti(Sistem, i, self.Trader.Lists.KarZararFiyatList, m * Carpan)
                if _karAl:
                    break
            if _karAl: result = 1
        return result

    def son_fiyata_gore_zarar_kes_seviye_hesapla_seviyeli(self, Sistem, BarIndex, SeviyeBas=-1, SeviyeSon=-10, Carpan=1000):
        result = 0
        i = BarIndex
        if self.Trader.Flags.ZararKesSeviyeHesaplaEnabled:
            myUtils = CUtils()
            _zararKes = False
            for m in range(SeviyeBas, SeviyeSon, -1):
                _zararKes = _zararKes or myUtils.asagi_kesti(Sistem, i, self.Trader.Lists.KarZararFiyatList, m * Carpan)
                if _zararKes:
                    break
            if _zararKes: result = 1
        return result
