class CFlags:
    def __init__(self):
        self.reset()

    def __del__(self):
        pass

    def initialize(self, Sistem):
        return self

    def reset(self):
        self.BakiyeGuncelle = False
        self.KomisyonGuncelle = False
        self.DonguSonuIstatistikGuncelle = False
        self.KomisyonuDahilEt = True
        self.KaymayiDahilEt = False
        self.AnlikKarZararHesaplaEnabled = False
        self.KarAlYuzdeHesaplaEnabled = False
        self.IzleyenStopYuzdeHesaplaEnabled = False
        self.ZararKesYuzdeHesaplaEnabled = False
        self.KarAlSeviyeHesaplaEnabled = False
        self.ZararKesSeviyeHesaplaEnabled = False
        self.AGerceklesti = False
        self.SGerceklesti = False
        self.FGerceklesti = False
        self.PGerceklesti = False
        self.IdealGetiriHesapla = True
        self.IdealGetiriHesaplandi = False
        return self
