V = Sistem.GrafikVerileri
Open = Sistem.GrafikFiyatOku(V, "Acilis")
High = Sistem.GrafikFiyatOku(V, "Yuksek")
Low = Sistem.GrafikFiyatOku(V, "Dusuk")
Close = Sistem.GrafikFiyatOku(V, "Kapanis")
Volume = Sistem.GrafikFiyatOku(V, "Hacim")
Lot = Sistem.GrafikFiyatOku(V, "Lot")
BarCount = Sistem.BarSayisi

myIndicators = Lib.GetIndicatorManager(Sistem)
mySystem = Lib.GetSystemWrapper(Sistem)
myUtils = Lib.GetUtils(Sistem)

mySystem.CreateModules(Sistem, Lib).Initialize(Sistem, V, Open, High, Low, Close, Volume, Lot)

mySystem.Reset(Sistem)
mySystem.InitializeParamsWithDefaults(Sistem)
mySystem.SetParamsForSingleRun(Sistem)

Yontem = Sistem.Parametreler[0]
Periyot1 = int(Sistem.Parametreler[1])
Periyot2 = int(Sistem.Parametreler[2])

MA1 = Sistem.MA(Close, Yontem, Periyot1)
MA2 = Sistem.MA(Close, Yontem, Periyot2)

DateTimes = ["25.05.2025 14:30:00", "02.06.2025 14:00:00"]
Dates = ["01.01.1900", "01.01.2100"]
Times = ["09:30:00", "11:59:00"]

mySystem.GetTrader(Sistem).ResetDateTimes(Sistem)
mySystem.GetTrader(Sistem).SetDateTimes(Sistem, DateTimes[0], DateTimes[1])

mySystem.GetTrader(Sistem).Signals.KarAlEnabled = False
mySystem.GetTrader(Sistem).Signals.ZararKesEnabled = False
mySystem.GetTrader(Sistem).Signals.GunSonuPozKapatEnabled = False
mySystem.GetTrader(Sistem).Signals.TimeFilteringEnabled = True

mySystem.Start(Sistem)
for i in range(BarCount):
    Al = False
    Sat = False
    FlatOl = False
    PasGec = False
    KarAl = False
    ZararKes = False
    isTradeEnabled = False
    isPozKapatEnabled = False

    mySystem.EmirleriResetle(Sistem, i)

    mySystem.EmirOncesiDonguFoksiyonlariniCalistir(Sistem, i)

    if i < 1:
        continue

    Al = myUtils.YukariKesti(Sistem, i, MA1, MA2)

    Sat = myUtils.AsagiKesti(Sistem, i, MA1, MA2)

    FlatOl = False

    Al = True
    Al = Al and Ma0[i] > Ma1[i]
    Al = Al and Ma1[i] > Ma2[i]
    Al = Al and Ma2[i] > Ma3[i]

    Sat = True
    Sat = Sat and Ma0[i] < Ma1[i]
    Sat = Sat and Ma1[i] < Ma2[i]
    Sat = Sat and Ma2[i] < Ma3[i]

    Al = myUtils.YukariKesti(Sistem, i, MA1, MA2)

    Sat = myUtils.AsagiKesti(Sistem, i, MA1, MA2)

    Al = myUtils.YukariKesti(Sistem, i, Rsi, 50.0)

    Sat = myUtils.AsagiKesti(Sistem, i, Rsi, 50.0)

    KarAl = mySystem.GetTrader(Sistem).KarAlZararKes.SonFiyataGoreKarAlSeviyeHesapla(Sistem, i, 5, 50, 1000) != 0

    ZararKes = mySystem.GetTrader(Sistem).KarAlZararKes.SonFiyataGoreZararKesSeviyeHesapla(Sistem, i, -1, -10, 1000) != 0

    KarAl = mySystem.GetTrader(Sistem).Signals.KarAlEnabled

    ZararKes = mySystem.GetTrader(Sistem).Signals.ZararKesEnabled

    IsSonYonA = mySystem.GetTrader(Sistem).IsSonYonA(Sistem)

    IsSonYonS = mySystem.GetTrader(Sistem).IsSonYonS(Sistem)

    IsSonYonF = mySystem.GetTrader(Sistem).IsSonYonF(Sistem)

    useTimeFiltering = mySystem.GetTrader(Sistem).Signals.TimeFilteringEnabled

    mySystem.EmirleriSetle(Sistem, i, Al, Sat, FlatOl, PasGec, KarAl, ZararKes)

    mySystem.IslemZamanFiltresiUygula(Sistem, i)

    mySystem.EmirSonrasiDonguFoksiyonlariniCalistir(Sistem, i)

mySystem.Stop(Sistem)

mySystem.HesaplamalariYap(Sistem)

mySystem.SonuclariEkrandaGoster(Sistem)

mySystem.SonuclariDosyayaYaz(Sistem)

k = 0

Lib.DeleteSystemWrapper(mySystem)
