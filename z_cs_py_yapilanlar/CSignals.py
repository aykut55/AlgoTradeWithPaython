class CSignals:
    def __init__(self):
        self.Al = False
        self.Sat = False
        self.FlatOl = False
        self.PasGec = False
        self.KarAl = False
        self.ZararKes = False
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
        self.KarAlEnabled = False
        self.ZararKesEnabled = False
        self.KarAlindi = False
        self.ZararKesildi = False
        self.FlatOlundu = False
        self.PozAcilabilir = False
        self.PozAcildi = False
        self.PozKapatilabilir = False
        self.PozKapatildi = False
        self.PozAcilabilirAlis = False
        self.PozAcilabilirSatis = False
        self.PozAcildiAlis = False
        self.PozAcildiSatis = False
        self.GunSonuPozKapatEnabled = False
        self.GunSonuPozKapatildi = False
        self.TimeFilteringEnabled = False

    def __del__(self):
        pass

    def initialize(self, Sistem):
        return self

    def reset(self, Sistem):
        self.Al = False
        self.Sat = False
        self.FlatOl = False
        self.PasGec = False
        self.KarAl = False
        self.ZararKes = False
        self.Sinyal = ""
        self.SonYon = "F"
        self.PrevYon = "F"
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
        return self
