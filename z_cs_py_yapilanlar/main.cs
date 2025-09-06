// --------------------------------------------------------------
var V              = Sistem.GrafikVerileri;
var Open           = Sistem.GrafikFiyatOku(V, "Acilis");   // var Close = Sistem.GrafikFiyatSec("Acilis");
var High           = Sistem.GrafikFiyatOku(V, "Yuksek");   // var Close = Sistem.GrafikFiyatSec("Yuksek");
var Low            = Sistem.GrafikFiyatOku(V, "Dusuk");    // var Close = Sistem.GrafikFiyatSec("Dusuk");
var Close          = Sistem.GrafikFiyatOku(V, "Kapanis");  // var Close = Sistem.GrafikFiyatSec("Kapanis");
var Volume         = Sistem.GrafikFiyatOku(V, "Hacim");    // var Close = Sistem.GrafikFiyatSec("Hacim");
var Lot            = Sistem.GrafikFiyatOku(V, "Lot");      // var Close = Sistem.GrafikFiyatSec("Lot");
var BarCount       = Sistem.BarSayisi;

// --------------------------------------------------------------
var myIndicators   = Lib.GetIndicatorManager(Sistem);
var mySystem       = Lib.GetSystemWrapper(Sistem);
var myUtils        = Lib.GetUtils(Sistem);

// --------------------------------------------------------------
mySystem.CreateModules(Sistem, Lib).Initialize(Sistem, V, Open, High, Low, Close, Volume, Lot);

// --------------------------------------------------------------
mySystem.Reset(Sistem);
mySystem.InitializeParamsWithDefaults(Sistem);
mySystem.SetParamsForSingleRun(Sistem);
//mySystem.SetParamsForOptimizasyon(Sistem);
//mySystem.ReadParamsFromFile(Sistem).UpdateSistemParametreleri(Sistem)

// ----------------- parametreleri al ---------------------------
var Yontem   = Sistem.Parametreler[0];
var Periyot1 = Convert.ToInt32(Sistem.Parametreler[1]);
var Periyot2 = Convert.ToInt32(Sistem.Parametreler[2]);
// ----------------- parametreleri al ---------------------------

// hareketli ortalamaları hesapla
var MA1 = Sistem.MA(Close, Yontem, Periyot1);
var MA2 = Sistem.MA(Close, Yontem, Periyot2);

// --------------------------------------------------------------
string[] DateTimes = { "25.05.2025 14:30:00", "02.06.2025 14:00:00" };
string[] Dates     = { "01.01.1900", "01.01.2100"                   };
string[] Times     = { "09:30:00", "11:59:00"                       };

mySystem.GetTrader(Sistem).ResetDateTimes(Sistem);
mySystem.GetTrader(Sistem).SetDateTimes(Sistem, DateTimes[0], DateTimes[1]);

// --------------------------------------------------------------
mySystem.GetTrader(Sistem).Signals.KarAlEnabled = false;
mySystem.GetTrader(Sistem).Signals.ZararKesEnabled = false;
mySystem.GetTrader(Sistem).Signals.GunSonuPozKapatEnabled = false;
mySystem.GetTrader(Sistem).Signals.TimeFilteringEnabled = true;

// --------------------------------------------------------------
mySystem.Start(Sistem);
for (int i = 0; i < BarCount; i++)
{
    bool Al = false;
    bool Sat = false;
    bool FlatOl = false;
    bool PasGec = false;
    bool KarAl = false;
    bool ZararKes = false;
    bool isTradeEnabled = false;
    bool isPozKapatEnabled = false;

    mySystem.EmirleriResetle(Sistem, i);

    mySystem.EmirOncesiDonguFoksiyonlariniCalistir(Sistem, i);

    if (i < 1) continue;

    Al = myUtils.YukariKesti(Sistem, i, MA1, MA2);

    Sat = myUtils.AsagiKesti(Sistem, i, MA1, MA2);

    FlatOl = false;

    Al = true;
    Al = Al && Ma0[i] > Ma1[i];
    Al = Al && Ma1[i] > Ma2[i];
    Al = Al && Ma2[i] > Ma3[i];/*
    Al = Al && Ma3[i] > Ma4[i];
    Al = Al && Ma4[i] > Ma5[i];
    Al = Al && Ma5[i] > Ma6[i];
    Al = Al && Ma6[i] > Ma7[i];
    Al = Al && Ma7[i] > Ma8[i];*/
    //Al = Al && Ma0[i] > Ma5[i];

    Sat = true;
    Sat = Sat && Ma0[i] < Ma1[i];
    Sat = Sat && Ma1[i] < Ma2[i];
    Sat = Sat && Ma2[i] < Ma3[i];/*
    Sat = Sat && Ma3[i] < Ma4[i];
    Sat = Sat && Ma4[i] < Ma5[i];
    Sat = Sat && Ma5[i] < Ma6[i];
    Sat = Sat && Ma6[i] < Ma7[i];
    Sat = Sat && Ma7[i] < Ma8[i];*/
    //Sat = Sat && Ma0[i] < Ma5[i];

    Al = myUtils.YukariKesti(Sistem, i, MA1, MA2);

    Sat = myUtils.AsagiKesti(Sistem, i, MA1, MA2);


    Al = myUtils.YukariKesti(Sistem, i, Rsi, 50f);

    Sat = myUtils.AsagiKesti(Sistem, i, Rsi, 50f);


    KarAl = mySystem.GetTrader(Sistem).KarAlZararKes.SonFiyataGoreKarAlSeviyeHesapla(Sistem, i, 5, 50, 1000) != 0 ? true : false;

    ZararKes = mySystem.GetTrader(Sistem).KarAlZararKes.SonFiyataGoreZararKesSeviyeHesapla(Sistem, i, -1, -10, 1000) != 0 ? true : false;

    KarAl = mySystem.GetTrader(Sistem).Signals.KarAlEnabled ? KarAl : false;

    ZararKes = mySystem.GetTrader(Sistem).Signals.ZararKesEnabled ? ZararKes : false;

    bool IsSonYonA = mySystem.GetTrader(Sistem).IsSonYonA(Sistem);

    bool IsSonYonS = mySystem.GetTrader(Sistem).IsSonYonS(Sistem);

    bool IsSonYonF = mySystem.GetTrader(Sistem).IsSonYonF(Sistem);

    bool useTimeFiltering = mySystem.GetTrader(Sistem).Signals.TimeFilteringEnabled ? true : false;


    // Siralama Onemli
    mySystem.EmirleriSetle(Sistem, i, Al, Sat, FlatOl, PasGec, KarAl, ZararKes);

    mySystem.IslemZamanFiltresiUygula(Sistem, i);

    mySystem.EmirSonrasiDonguFoksiyonlariniCalistir(Sistem, i);
}
mySystem.Stop(Sistem);

// --------------------------------------------------------------
mySystem.HesaplamalariYap(Sistem);

mySystem.SonuclariEkrandaGoster(Sistem);

mySystem.SonuclariDosyayaYaz(Sistem);

// --------------------------------------------------------------
int k = 0;

// --------------------------------------------------------------
Lib.DeleteSystemWrapper(mySystem);

