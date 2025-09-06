public class CStatistics
{
    CTrader Trader = null;

    public Dictionary<string, dynamic> IstatistiklerNew;

    // Grafik, Sembol, Sistem
    public int SistemId = 0;
    public string SistemName = "";
    public string GrafikSembol = "";
    public string GrafikPeriyot = "";
    public int LastBarIndex= 0;

    // Koşum Süreleri
    public int LastExecutionId = 0;
    public string LastExecutionTime = "";
    public string LastExecutionTimeStart = "";
    public string LastExecutionTimeStop = "";
    public UInt64 ExecutionTimeInMSec = 0;
    public string LastResetTime = "";
    public string LastStatisticsCalculationTime = "";

    // Toplam Geçen Süreler
    public float ToplamGecenSureAy = 0f;
    public float ToplamGecenSureGun = 0f;
    public float ToplamGecenSureSaat = 0f;
    public float ToplamGecenSureDakika = 0f;

    // Toplam Bar Sayısı
    public int ToplamBarSayisi = 0;

    // Others
    public int BarBegIndex = 0;
    public int BarEndIndex = 0;

    // Seçilen Bar
    public DateTime SecilenBarTarihi { get; set; }
    public int SecilenBarNumarasi = 0;
    public float SecilenBarAcilisFiyati = 0f;
    public float SecilenBarYuksekFiyati = 0f;
    public float SecilenBarDusukFiyati = 0f;
    public float SecilenBarKapanisFiyati = 0f;

    // İlk Bar ve Son Bar
    public DateTime IlkBarTarihi { get; set; }
    public DateTime SonBarTarihi { get; set; }
    public int IlkBarIndex = 0;
    public int SonBarIndex = 0;
    public float SonBarAcilisFiyati = 0f;
    public float SonBarYuksekFiyati = 0f;
    public float SonBarDusukFiyati = 0f;
    public float SonBarKapanisFiyati = 0f;

    // Bakiye, Getiri, Min Max
    public float IlkBakiyeFiyat = 0f;
    public float IlkBakiyePuan = 0f;
    public float BakiyeFiyat = 0f;
    public float BakiyePuan = 0f;
    public float GetiriFiyat = 0f;
    public float GetiriPuan = 0f;
    public float GetiriFiyatYuzde = 0f;
    public float GetiriPuanYuzde = 0f;
    public float BakiyeFiyatNet = 0f;
    public float BakiyePuanNet = 0f;
    public float GetiriFiyatNet = 0f;
    public float GetiriPuanNet = 0f;
    public float GetiriFiyatYuzdeNet = 0f;
    public float GetiriPuanYuzdeNet = 0f;
    public float GetiriKz = 0f;
    public float GetiriKzNet = 0f;
    public float MinBakiyeFiyat = 0f;
    public float MaxBakiyeFiyat = 0f;
    public float MinBakiyePuan = 0f;
    public float MaxBakiyePuan = 0f;
    public float MinBakiyeFiyatYuzde = 0f;
    public float MaxBakiyeFiyatYuzde = 0f;
    public int MinBakiyeFiyatIndex = 0;
    public int MaxBakiyeFiyatIndex = 0;
    public int MinBakiyePuanIndex = 0;
    public int MaxBakiyePuanIndex = 0;
    public float MinBakiyeFiyatNet = 0f;
    public float MaxBakiyeFiyatNet = 0f;
    public int MinBakiyeFiyatNetIndex = 0;
    public int MaxBakiyeFiyatNetIndex = 0;
    public float MinBakiyeFiyatNetYuzde = 0f;
    public float MaxBakiyeFiyatNetYuzde = 0f;

    public float GetiriKzSistem = 0f;
    public float GetiriKzNetSistem = 0f;
    public float GetiriKzSistemYuzde = 0f;
    public float GetiriKzNetSistemYuzde = 0f;

    public string GetiriFiyatTipi = "";

    // İslem Sayilari 1
    public int IslemSayisi = 0;
    public int AlisSayisi = 0;
    public int SatisSayisi = 0;
    public int FlatSayisi = 0;
    public int PassSayisi = 0;
    public int KarAlSayisi = 0;
    public int ZararKesSayisi = 0;

    // İslem Sayilari 2
    public int KazandiranIslemSayisi = 0;
    public int KaybettirenIslemSayisi = 0;
    public int NotrIslemSayisi = 0;
    public int KazandiranAlisSayisi = 0;
    public int KaybettirenAlisSayisi = 0;
    public int NotrAlisSayisi = 0;
    public int KazandiranSatisSayisi = 0;
    public int KaybettirenSatisSayisi = 0;
    public int NotrSatisSayisi = 0;

    // Komut Sayilari
    public int AlKomutSayisi = 0;
    public int SatKomutSayisi = 0;
    public int PasGecKomutSayisi = 0;
    public int KarAlKomutSayisi = 0;
    public int ZararKesKomutSayisi = 0;
    public int FlatOlKomutSayisi = 0;

    // Komisyon
    public int KomisyonIslemSayisi = 0;
    public int KomisyonVarlikAdedSayisi = 1;
    public float KomisyonCarpan = 0f;
    public float KomisyonFiyat = 0f;
    public float KomisyonFiyatYuzde = 0f;
    public bool KomisyonuDahilEt = false;

    // Kar Zarar  Min Max
    public float KarZararFiyat = 0f;
    public float KarZararPuan = 0f;
    public float KarZararFiyatYuzde = 0f;
    public float NetKarFiyat = 0f;
    public float ToplamKarFiyat = 0f;
    public float ToplamZararFiyat = 0f;
    public float NetKarPuan = 0f;
    public float ToplamKarPuan = 0f;
    public float ToplamZararPuan = 0f;
    public float MaxKarFiyat = 0f;
    public float MaxZararFiyat = 0f;
    public float MaxKarPuan = 0f;
    public float MaxZararPuan = 0f;
    public int MaxZararFiyatIndex = 0;
    public int MaxKarFiyatIndex = 0;
    public int MaxZararPuanIndex = 0;
    public int MaxKarPuanIndex = 0;
    public int KardaBarSayisi = 0;
    public int ZarardaBarSayisi = 0;
    public float KarliIslemOrani = 0f;

    // GetiriMaxDD
    public float GetiriMaxDD = 0f;
    public DateTime GetiriMaxDDTarih { get; set; }
    public float GetiriMaxKayip = 0f;

    // ProfitFactor
    public float ProfitFactor = 0f;
    public float ProfitFactorSistem = 0f;

    // Ort İslem Sayilari
    public float OrtAylikIslemSayisi = 0f;
    public float OrtHaftalikIslemSayisi = 0f;
    public float OrtGunlukIslemSayisi = 0f;
    public float OrtSaatlikIslemSayisi = 0f;

    // Son Emir İstatistikleri
    public string Sinyal = "";
    public string SonYon = "";
    public string PrevYon = "";
    public float SonFiyat = 0f;
    public float SonAFiyat = 0f;
    public float SonSFiyat = 0f;
    public float SonFFiyat = 0f;
    public float SonPFiyat = 0f;
    public float PrevFiyat = 0f;
    public float PrevAFiyat = 0f;
    public float PrevSFiyat = 0f;
    public float PrevFFiyat = 0f;
    public float PrevPFiyat = 0f;
    public int SonBarNo = 0;
    public int SonABarNo = 0;
    public int SonSBarNo = 0;
    public int SonFBarNo = 0;
    public int SonPBarNo = 0;
    public int PrevBarNo = 0;
    public int PrevABarNo = 0;
    public int PrevSBarNo = 0;
    public int PrevFBarNo = 0;
    public int PrevPBarNo = 0;
    public float EmirKomut = 0f;
    public float EmirStatus = 0f;

    // Varlik
    public int HisseSayisi = 0;
    public int KontratSayisi = 0;
    public int VarlikAdedCarpani = 0;
    public int VarlikAdedSayisi = 0;
    public float KaymaMiktari = 0f;
    public bool KaymayiDahilEt = false;

    // Getiri Fiyat-Puan Ay Hafta Gun Saat
    public float GetiriFiyatBuAy = 0f;
    public float GetiriFiyatAy1 = 0f;
    public float GetiriFiyatAy2 = 0f;
    public float GetiriFiyatAy3 = 0f;
    public float GetiriFiyatAy4 = 0f;
    public float GetiriFiyatAy5 = 0f;
    public float GetiriFiyatBuHafta = 0f;
    public float GetiriFiyatHafta1 = 0f;
    public float GetiriFiyatHafta2 = 0f;
    public float GetiriFiyatHafta3 = 0f;
    public float GetiriFiyatHafta4 = 0f;
    public float GetiriFiyatHafta5 = 0f;
    public float GetiriFiyatBuGun = 0f;
    public float GetiriFiyatGun1 = 0f;
    public float GetiriFiyatGun2 = 0f;
    public float GetiriFiyatGun3 = 0f;
    public float GetiriFiyatGun4 = 0f;
    public float GetiriFiyatGun5 = 0f;
    public float GetiriFiyatBuSaat = 0f;
    public float GetiriFiyatSaat1 = 0f;
    public float GetiriFiyatSaat2 = 0f;
    public float GetiriFiyatSaat3 = 0f;
    public float GetiriFiyatSaat4 = 0f;
    public float GetiriFiyatSaat5 = 0f;
    public float GetiriPuanBuAy = 0f;
    public float GetiriPuanAy1 = 0f;
    public float GetiriPuanAy2 = 0f;
    public float GetiriPuanAy3 = 0f;
    public float GetiriPuanAy4 = 0f;
    public float GetiriPuanAy5 = 0f;
    public float GetiriPuanBuHafta = 0f;
    public float GetiriPuanHafta1 = 0f;
    public float GetiriPuanHafta2 = 0f;
    public float GetiriPuanHafta3 = 0f;
    public float GetiriPuanHafta4 = 0f;
    public float GetiriPuanHafta5 = 0f;
    public float GetiriPuanBuGun = 0f;
    public float GetiriPuanGun1 = 0f;
    public float GetiriPuanGun2 = 0f;
    public float GetiriPuanGun3 = 0f;
    public float GetiriPuanGun4 = 0f;
    public float GetiriPuanGun5 = 0f;
    public float GetiriPuanBuSaat = 0f;
    public float GetiriPuanSaat1 = 0f;
    public float GetiriPuanSaat2 = 0f;
    public float GetiriPuanSaat3 = 0f;
    public float GetiriPuanSaat4 = 0f;
    public float GetiriPuanSaat5 = 0f;

    ~CStatistics()
    {

    }

    public CStatistics()
    {
        IstatistiklerNew = new Dictionary<string, dynamic>();
    }

    public CStatistics Initialize(dynamic Sistem, CTrader Trader)
    {
        this.Trader = Trader;

        return this;
    }

    public CStatistics Reset(dynamic Sistem)
    {
        return this;
    }

    public int Hesapla(dynamic Sistem)
    {
        int result = 0;

        var V = Trader.V;

        Trader.LastStatisticsCalculationTime = DateTime.Now.ToString("yyyy.MM.dd HH:mm:ss");

        // ---------------------------------------------
        readValues(Sistem);

        // ---------------------------------------------
        ToplamBarSayisi = Trader.BarCount;

        // ---------------------------------------------
        IlkBarTarihi = V[0].Date;
        SonBarTarihi = V[V.Count - 1].Date;

        var LastBar = V.Count - 1;
        SecilenBarNumarasi = Sistem.SelectBarNo;
        SecilenBarTarihi = Sistem.SelectTarih;
        if (SecilenBarNumarasi <= LastBar)
        {
            SecilenBarAcilisFiyati = V[SecilenBarNumarasi].Open;
            SecilenBarYuksekFiyati = V[SecilenBarNumarasi].High;
            SecilenBarDusukFiyati = V[SecilenBarNumarasi].Low;
            SecilenBarKapanisFiyati = V[SecilenBarNumarasi].Close;
        }

        SonBarTarihi = V[LastBar].Date;
        SonBarAcilisFiyati = V[LastBar].Open;
        SonBarYuksekFiyati = V[LastBar].High;
        SonBarDusukFiyati = V[LastBar].Low;
        SonBarKapanisFiyati = V[LastBar].Close;
        SonBarIndex = V.Count - 1;

        // ---------------------------------------------
        var sureDakika = (DateTime.Now - V[0].Date).TotalMinutes;
        var sureSaat = (DateTime.Now - V[0].Date).TotalHours;
        var sureGun = (DateTime.Now - V[0].Date).TotalDays;
        var sureAy = sureGun / 30.4f;

        ToplamGecenSureAy = Convert.ToSingle(sureAy);           //ToplamGecenSureAy = AyList.Count;
        ToplamGecenSureGun = Convert.ToInt32(sureGun);         //ToplamGecenSureGun = GunList.Count;
        ToplamGecenSureSaat = Convert.ToInt32(sureSaat);       //ToplamGecenSureSaat = SaatList.Count;
        ToplamGecenSureDakika = Convert.ToInt32(sureDakika);   //ToplamGecenSureDakika = DakikalikList.Count;

        // -------------------------------------------------------------------
        // Aylik İşlem Sayisi - Ort Gunluk İşlem Sayisi - Ort Saatlik İşlem Sayisi
        OrtAylikIslemSayisi = 1.0f * Trader.Status.IslemSayisi / ToplamGecenSureAy;
        OrtHaftalikIslemSayisi = 0f;
        OrtGunlukIslemSayisi = 1.0f * Trader.Status.IslemSayisi / ToplamGecenSureGun;
        OrtSaatlikIslemSayisi = 1.0f * Trader.Status.IslemSayisi / ToplamGecenSureSaat;

        // Aylik bazda ort islem sayisi yukarisi ile check edilecek!!!
        //var ortIslemSayisi = Convert.ToSingle(Sistem.GetiriToplamIslem) / TimeUtils.GecenSure(Sistem);

        // -------------------------------------------------------------------
        // GetiriMaxDD
        GetiriMaxDD = -1.0f * Convert.ToSingle(Sistem.GetiriMaxDD);
        GetiriMaxDDTarih = Sistem.GetiriMaxDDTarih;
        GetiriMaxKayip = Trader.Status.VarlikAdedSayisi * -1 * GetiriMaxDD;

        // ---------------------------------------------
        // MaxKarFiyat - MaxZararFiyat, MaxKarPuan - MaxZararPuan
        MaxKarFiyat = 0f;
        MaxZararFiyat = 0f;
        MaxKarPuan = 0f;
        MaxZararPuan = 0f;

        // MinBakiye - MaxBakiye, MinBakiyeIndex - MaxBakiyeIndex
        MinBakiyeFiyat = Trader.Status.IlkBakiyeFiyat;
        MaxBakiyeFiyat = Trader.Status.IlkBakiyeFiyat;
        MinBakiyeFiyatNet = Trader.Status.IlkBakiyeFiyat;
        MaxBakiyeFiyatNet = Trader.Status.IlkBakiyeFiyat;

        // MinBakiye - MaxBakiye, MinBakiyeIndex - MaxBakiyeIndex
        MinBakiyePuan = Trader.Status.IlkBakiyePuan;
        MaxBakiyePuan = Trader.Status.IlkBakiyePuan;

        // i = 1 den baslamasi gerekiyor, aksi durumda MinBakiyeFiyat 0 geliyor.
        // Bug var!
        for (int i = 1; i < Trader.BarCount; i++)
        {
            // MaxKarFiyat - MaxZararFiyat, MaxKarPuan - MaxZararPuan
            // ---------------------------------------------
            if (Trader.Lists.KarZararFiyatList[i] < MaxZararFiyat)
            {
                MaxZararFiyat = Trader.Lists.KarZararFiyatList[i];
                MaxZararFiyatIndex = i;
            }

            if (Trader.Lists.KarZararFiyatList[i] > MaxKarFiyat)
            {
                MaxKarFiyat = Trader.Lists.KarZararFiyatList[i];
                MaxKarFiyatIndex = i;
            }

            if (Trader.Lists.KarZararPuanList[i] < MaxZararPuan)
            {
                MaxZararPuan = Trader.Lists.KarZararPuanList[i];
                MaxZararPuanIndex = i;
            }

            if (Trader.Lists.KarZararPuanList[i] > MaxKarPuan)
            {
                MaxKarPuan = Trader.Lists.KarZararPuanList[i];
                MaxKarPuanIndex = i;
            }

            // MinBakiye - MaxBakiye, MinBakiyeIndex - MaxBakiyeIndex
            // ---------------------------------------------
            if (Trader.Lists.BakiyeFiyatList[i] < MinBakiyeFiyat)
            {
                MinBakiyeFiyat = Trader.Lists.BakiyeFiyatList[i];
                MinBakiyeFiyatIndex = i;
            }
            if (Trader.Lists.BakiyeFiyatList[i] > MaxBakiyeFiyat)
            {
                MaxBakiyeFiyat = Trader.Lists.BakiyeFiyatList[i];
                MaxBakiyeFiyatIndex = i;
            }

            if (Trader.Lists.BakiyeFiyatNetList[i] < MinBakiyeFiyatNet)
            {
                MinBakiyeFiyatNet = Trader.Lists.BakiyeFiyatNetList[i];
                MinBakiyeFiyatNetIndex = i;
            }
            if (Trader.Lists.BakiyeFiyatNetList[i] > MaxBakiyeFiyatNet)
            {
                MaxBakiyeFiyatNet = Trader.Lists.BakiyeFiyatNetList[i];
                MaxBakiyeFiyatNetIndex = i;
            }

            // MinBakiye - MaxBakiye, MinBakiyeIndex - MaxBakiyeIndex
            // ---------------------------------------------
            if (Trader.Lists.BakiyePuanList[i] < MinBakiyePuan)
            {
                MinBakiyePuan = Trader.Lists.BakiyePuanList[i];
                MinBakiyePuanIndex = i;
            }
            if (Trader.Lists.BakiyePuanList[i] > MaxBakiyePuan)
            {
                MaxBakiyePuan = Trader.Lists.BakiyePuanList[i];
                MaxBakiyePuanIndex = i;
            }
        }

        // ProfitFactor
        ProfitFactor = ToplamKarPuan / Math.Abs(ToplamZararPuan);

        // ProfitFactor (Sistem)
        ProfitFactorSistem = Convert.ToSingle(Sistem.ProfitFactor);

        // ---------------------------------------------
        // Karlı İşlem Oranı
        KarliIslemOrani = (1.0f * KazandiranIslemSayisi) / (1.0f * IslemSayisi) * 100.0f;

        // ---------------------------------------------
        MinBakiyeFiyatYuzde = (MinBakiyeFiyat - IlkBakiyeFiyat) * 100f / IlkBakiyeFiyat;
        MaxBakiyeFiyatYuzde = (MaxBakiyeFiyat - IlkBakiyeFiyat) * 100f / IlkBakiyeFiyat;
        MinBakiyeFiyatNetYuzde = (MinBakiyeFiyatNet - IlkBakiyeFiyat) * 100f / IlkBakiyeFiyat;
        MaxBakiyeFiyatNetYuzde = (MaxBakiyeFiyatNet - IlkBakiyeFiyat) * 100f / IlkBakiyeFiyat;
        KomisyonFiyatYuzde = GetiriFiyatYuzde - GetiriFiyatYuzdeNet;

        GetiriKzSistemYuzde = 0f;
        GetiriKzNetSistemYuzde = 0f;

        // ---------------------------------------------
        GetiriIstatiskleriHesapla(Sistem);

        // ---------------------------------------------
        assignToMap(Sistem);

        return result;
    }

    public void readValues(dynamic Sistem)
    {
        //Trader.Status.KomisyonFiyat = Trader.Lists.KomisyonIslemSayisiList[i] * Convert.ToSingle(Trader.Status.KomisyonCarpan) * Trader.Status.KomisyonVarlikAdedSayisi;
        //Trader.Lists.KomisyonFiyatList[i] = Trader.Status.KomisyonFiyat;

        SistemId = 0;
        SistemName = Sistem.Name;
        GrafikSembol = Sistem.Sembol;
        GrafikPeriyot = Sistem.Periyot;

        LastExecutionId = 0;
        LastExecutionTime = Trader.LastExecutionTime;
        LastExecutionTimeStart = Trader.LastExecutionTimeStart;
        LastExecutionTimeStop = Trader.LastExecutionTimeStop;
        ExecutionTimeInMSec = Trader.ExecutionTimeInMSec;
        LastResetTime = Trader.LastResetTime;
        LastStatisticsCalculationTime = Trader.LastStatisticsCalculationTime;

        // ---------------------------------------------
        //IslemSayisi = AlisSayisi * 1 + SatisSayisi * 1 + FlatSayisi * 0;
        IslemSayisi = Trader.Status.IslemSayisi;

        KazandiranIslemSayisi = Trader.Status.KazandiranIslemSayisi;
        KaybettirenIslemSayisi = Trader.Status.KaybettirenIslemSayisi;
        NotrIslemSayisi = Trader.Status.NotrIslemSayisi;

        KazandiranAlisSayisi = Trader.Status.KazandiranAlisSayisi;
        KaybettirenAlisSayisi = Trader.Status.KaybettirenAlisSayisi;
        NotrAlisSayisi = Trader.Status.NotrAlisSayisi;

        KazandiranSatisSayisi = Trader.Status.KazandiranSatisSayisi;
        KaybettirenSatisSayisi = Trader.Status.KaybettirenSatisSayisi;
        NotrSatisSayisi = Trader.Status.NotrSatisSayisi;

        AlisSayisi = Trader.Status.AlisSayisi;
        SatisSayisi = Trader.Status.SatisSayisi;
        FlatSayisi = Trader.Status.FlatSayisi;
        PassSayisi = Trader.Status.PassSayisi;
        KarAlSayisi = Trader.Status.KarAlSayisi;
        ZararKesSayisi = Trader.Status.ZararKesSayisi;

        AlKomutSayisi = Trader.Status.AlKomutSayisi;
        SatKomutSayisi = Trader.Status.SatKomutSayisi;
        PasGecKomutSayisi = Trader.Status.PasGecKomutSayisi;
        KarAlKomutSayisi = Trader.Status.KarAlKomutSayisi;
        ZararKesKomutSayisi = Trader.Status.ZararKesKomutSayisi;
        FlatOlKomutSayisi = Trader.Status.FlatOlKomutSayisi;

        KardaBarSayisi = Trader.Status.KardaBarSayisi;
        ZarardaBarSayisi = Trader.Status.ZarardaBarSayisi;

        KarZararFiyat = Trader.Status.KarZararFiyat;
        KarZararPuan = Trader.Status.KarZararPuan;
        KarZararFiyatYuzde = Trader.Status.KarZararFiyatYuzde;

        KomisyonIslemSayisi = Trader.Status.KomisyonIslemSayisi;

        KomisyonVarlikAdedSayisi = Trader.Status.KomisyonVarlikAdedSayisi;
        KomisyonCarpan = Trader.Status.KomisyonCarpan;
        KomisyonFiyat = Trader.Status.KomisyonFiyat;
        KomisyonuDahilEt = Trader.Flags.KomisyonuDahilEt;

        HisseSayisi = Trader.Status.HisseSayisi;

        KontratSayisi = Trader.Status.KontratSayisi;
        VarlikAdedCarpani = Trader.Status.VarlikAdedCarpani;
        VarlikAdedSayisi = Trader.Status.VarlikAdedSayisi;

        KaymaMiktari = Trader.Status.KaymaMiktari;
        KaymayiDahilEt = Trader.Flags.KaymayiDahilEt;

        IlkBakiyeFiyat = Trader.Status.IlkBakiyeFiyat;
        IlkBakiyePuan = Trader.Status.IlkBakiyePuan;
        BakiyeFiyat = Trader.Status.BakiyeFiyat;
        BakiyePuan = Trader.Status.BakiyePuan;
        GetiriFiyat = Trader.Status.GetiriFiyat;
        GetiriPuan = Trader.Status.GetiriPuan;
        GetiriFiyatYuzde = Trader.Status.GetiriFiyatYuzde;
        GetiriPuanYuzde = Trader.Status.GetiriPuanYuzde;
        BakiyeFiyatNet = Trader.Status.BakiyeFiyatNet;
        BakiyePuanNet = Trader.Status.BakiyePuanNet;
        GetiriFiyatNet = Trader.Status.GetiriFiyatNet;
        GetiriPuanNet = Trader.Status.GetiriPuanNet;
        GetiriFiyatYuzdeNet = Trader.Status.GetiriFiyatYuzdeNet;
        GetiriPuanYuzdeNet = Trader.Status.GetiriPuanYuzdeNet;
        GetiriKz = Trader.Status.GetiriKz;
        GetiriKzNet = Trader.Status.GetiriKzNet;
        GetiriKzSistem = Trader.Status.GetiriKzSistem;
        GetiriKzNetSistem = Trader.Status.GetiriKzNetSistem;
        GetiriFiyatTipi = Trader.Status.GetiriFiyatTipi;

        NetKarPuan = Trader.Status.NetKarPuan;
        ToplamKarPuan = Trader.Status.ToplamKarPuan;
        ToplamZararPuan = Trader.Status.ToplamZararPuan;

        NetKarFiyat = Trader.Status.NetKarFiyat;
        ToplamKarFiyat = Trader.Status.ToplamKarFiyat;
        ToplamZararFiyat = Trader.Status.ToplamZararFiyat;

        ToplamZararPuan = (ToplamZararPuan == 0) ? 0.00000000001f : ToplamZararPuan;

        Sinyal = Trader.Signals.Sinyal;
        SonYon = Trader.Signals.SonYon;
        PrevYon = Trader.Signals.PrevYon;
        SonFiyat = Trader.Signals.SonFiyat;
        SonAFiyat = Trader.Signals.SonAFiyat;
        SonSFiyat = Trader.Signals.SonSFiyat;
        SonFFiyat = Trader.Signals.SonFFiyat;
        SonPFiyat = Trader.Signals.SonPFiyat;
        PrevFiyat = Trader.Signals.PrevFiyat;
        PrevAFiyat = Trader.Signals.PrevAFiyat;
        PrevSFiyat = Trader.Signals.PrevSFiyat;
        PrevFFiyat = Trader.Signals.PrevFFiyat;
        PrevPFiyat =  Trader.Signals.PrevPFiyat;
        SonBarNo = Trader.Signals.SonBarNo;
        SonABarNo = Trader.Signals.SonABarNo;
        SonSBarNo = Trader.Signals.SonSBarNo;
        SonFBarNo = Trader.Signals.SonFBarNo;
        SonPBarNo = Trader.Signals.SonPBarNo;
        PrevBarNo =  Trader.Signals.PrevBarNo;
        PrevABarNo = Trader.Signals.PrevABarNo;
        PrevSBarNo = Trader.Signals.PrevSBarNo;
        PrevFBarNo = Trader.Signals.PrevFBarNo;
        PrevPBarNo = Trader.Signals.PrevPBarNo;
        EmirKomut =  Trader.Signals.EmirKomut;
        EmirStatus =  Trader.Signals.EmirStatus;
    }

    public void assignToMap(dynamic Sistem)
    {
        IstatistiklerNew.Clear();

        // Grafik, Sembol, Sistem
        IstatistiklerNew["GrafikSembol"] = GrafikSembol;
        IstatistiklerNew["GrafikPeriyot"] = GrafikPeriyot;
        IstatistiklerNew["SistemId"] = SistemId.ToString();
        IstatistiklerNew["SistemName"] = SistemName;

        // Koşum Süreleri
        IstatistiklerNew["LastExecutionTime"] = LastExecutionTime;
        IstatistiklerNew["LastExecutionTimeStart"] = LastExecutionTimeStart;
        IstatistiklerNew["LastExecutionTimeStop"] = LastExecutionTimeStop;
        IstatistiklerNew["ExecutionTimeInMSec"] = ExecutionTimeInMSec.ToString();
        IstatistiklerNew["LastExecutionId"] = LastExecutionId;
        IstatistiklerNew["LastResetTime"] = LastResetTime;
        IstatistiklerNew["LastStatisticsCalculationTime"] = LastStatisticsCalculationTime;

        // Toplam Geçen Süreler
        IstatistiklerNew["ToplamGecenSureAy"] = ToplamGecenSureAy.ToString("0.0");
        IstatistiklerNew["ToplamGecenSureGun"] = ToplamGecenSureGun.ToString();
        IstatistiklerNew["ToplamGecenSureSaat"] = ToplamGecenSureSaat.ToString();
        IstatistiklerNew["ToplamGecenSureDakika"] = ToplamGecenSureDakika.ToString();

        // Toplam Bar Sayısı
        IstatistiklerNew["ToplamBarSayisi"] = ToplamBarSayisi.ToString();

        // Seçilen Bar
        IstatistiklerNew["SecilenBarNumarasi"] = SecilenBarNumarasi.ToString();
        IstatistiklerNew["SecilenBarTarihi"] = SecilenBarTarihi.ToString("dd.MM.yyyy");
        IstatistiklerNew["SecilenBarSaati"] = SecilenBarTarihi.TimeOfDay.ToString();

        // İlk Bar ve Son Bar
        IstatistiklerNew["IlkBarTarihi"] = IlkBarTarihi.ToString("dd.MM.yyyy");
        IstatistiklerNew["IlkBarSaati"] = IlkBarTarihi.TimeOfDay.ToString();
        IstatistiklerNew["SonBarTarihi"] = SonBarTarihi.ToString("dd.MM.yyyy");
        IstatistiklerNew["SonBarSaati"] = SonBarTarihi.TimeOfDay.ToString();
        IstatistiklerNew["IlkBarIndex"] = IlkBarIndex.ToString();
        IstatistiklerNew["SonBarIndex"] = SonBarIndex.ToString();
        IstatistiklerNew["SonBarAcilisFiyati"] = SonBarAcilisFiyati.ToString();
        IstatistiklerNew["SonBarYuksekFiyati"] = SonBarYuksekFiyati.ToString();
        IstatistiklerNew["SonBarDusukFiyati"] = SonBarDusukFiyati.ToString();
        IstatistiklerNew["SonBarKapanisFiyati"] = SonBarKapanisFiyati.ToString();

        // Bakiye, Getiri, Min Max
        IstatistiklerNew["IlkBakiyeFiyat"] = IlkBakiyeFiyat.ToString();
        IstatistiklerNew["IlkBakiyePuan"] = IlkBakiyePuan.ToString();
        IstatistiklerNew["BakiyeFiyat"] = BakiyeFiyat.ToString();
        IstatistiklerNew["BakiyePuan"] = BakiyePuan.ToString();
        IstatistiklerNew["GetiriFiyat"] = GetiriFiyat.ToString();
        IstatistiklerNew["GetiriPuan"] = GetiriPuan.ToString();
        IstatistiklerNew["GetiriFiyatYuzde"] = GetiriFiyatYuzde.ToString();
        IstatistiklerNew["GetiriPuanYuzde"] = GetiriPuanYuzde.ToString();
        IstatistiklerNew["BakiyeFiyatNet"] = BakiyeFiyatNet.ToString();
        IstatistiklerNew["BakiyePuanNet"] = BakiyePuanNet.ToString();
        IstatistiklerNew["GetiriFiyatNet"] = GetiriFiyatNet.ToString();
        IstatistiklerNew["GetiriPuanNet"] = GetiriPuanNet.ToString();
        IstatistiklerNew["GetiriFiyatYuzdeNet"] = GetiriFiyatYuzdeNet.ToString();
        IstatistiklerNew["GetiriPuanYuzdeNet"] = GetiriPuanYuzdeNet.ToString();
        IstatistiklerNew["GetiriKz"] = GetiriKz.ToString();
        IstatistiklerNew["GetiriKzNet"] = GetiriKzNet.ToString();
        IstatistiklerNew["GetiriFiyatTipi"] = GetiriFiyatTipi;
        IstatistiklerNew["MinBakiyeFiyat"] = MinBakiyeFiyat.ToString();
        IstatistiklerNew["MaxBakiyeFiyat"] = MaxBakiyeFiyat.ToString();
        IstatistiklerNew["MinBakiyePuan"] = MinBakiyePuan.ToString();
        IstatistiklerNew["MaxBakiyePuan"] = MaxBakiyePuan.ToString();
        IstatistiklerNew["MinBakiyeFiyatYuzde"] = MinBakiyeFiyatYuzde.ToString();
        IstatistiklerNew["MaxBakiyeFiyatYuzde"] = MaxBakiyeFiyatYuzde.ToString();
        IstatistiklerNew["MinBakiyeFiyatIndex"] = MinBakiyeFiyatIndex.ToString();
        IstatistiklerNew["MaxBakiyeFiyatIndex"] = MaxBakiyeFiyatIndex.ToString();
        IstatistiklerNew["MinBakiyePuanIndex"] = MinBakiyePuanIndex.ToString();
        IstatistiklerNew["MaxBakiyePuanIndex"] = MaxBakiyePuanIndex.ToString();
        IstatistiklerNew["MinBakiyeFiyatNet"] = MinBakiyeFiyatNet.ToString();
        IstatistiklerNew["MaxBakiyeFiyatNet"] = MaxBakiyeFiyatNet.ToString();
        IstatistiklerNew["MinBakiyeFiyatNetIndex"] = MinBakiyeFiyatNetIndex.ToString();
        IstatistiklerNew["MaxBakiyeFiyatNetIndex"] = MaxBakiyeFiyatNetIndex.ToString();
        IstatistiklerNew["MinBakiyeFiyatNetYuzde"] = MinBakiyeFiyatNetYuzde.ToString();
        IstatistiklerNew["MaxBakiyeFiyatNetYuzde"] = MaxBakiyeFiyatNetYuzde.ToString();

        IstatistiklerNew["GetiriKzSistem"] = GetiriKzSistem.ToString("0.00");
        IstatistiklerNew["GetiriKzSistemYuzde"] = GetiriKzSistemYuzde.ToString("0.00");
        IstatistiklerNew["GetiriKzNetSistem"] = GetiriKzNetSistem.ToString("0.00");
        IstatistiklerNew["GetiriKzNetSistemYuzde"] = GetiriKzNetSistemYuzde.ToString("0.00");

        // İslem Sayilari 1
        IstatistiklerNew["IslemSayisi"] = IslemSayisi.ToString();
        IstatistiklerNew["AlisSayisi"] = AlisSayisi.ToString();
        IstatistiklerNew["SatisSayisi"] = SatisSayisi.ToString();
        IstatistiklerNew["FlatSayisi"] = FlatSayisi.ToString();
        IstatistiklerNew["PassSayisi"] = PassSayisi.ToString();
        IstatistiklerNew["KarAlSayisi"] = KarAlSayisi.ToString();
        IstatistiklerNew["ZararKesSayisi"] = ZararKesSayisi.ToString();

        // İslem Sayilari 2
        IstatistiklerNew["KazandiranIslemSayisi"] = KazandiranIslemSayisi.ToString();
        IstatistiklerNew["KaybettirenIslemSayisi"] = KaybettirenIslemSayisi.ToString();
        IstatistiklerNew["NotrIslemSayisi"] = NotrIslemSayisi.ToString();
        IstatistiklerNew["KazandiranAlisSayisi"] = KazandiranAlisSayisi.ToString();
        IstatistiklerNew["KaybettirenAlisSayisi"] = KaybettirenAlisSayisi.ToString();
        IstatistiklerNew["NotrAlisSayisi"] = NotrAlisSayisi.ToString();
        IstatistiklerNew["KazandiranSatisSayisi"] = KazandiranSatisSayisi.ToString();
        IstatistiklerNew["KaybettirenSatisSayisi"] = KaybettirenSatisSayisi.ToString();
        IstatistiklerNew["NotrSatisSayisi"] = NotrSatisSayisi.ToString();

        // Komut Sayilari
        IstatistiklerNew["AlKomutSayisi"] = AlKomutSayisi.ToString();
        IstatistiklerNew["SatKomutSayisi"] = SatKomutSayisi.ToString();
        IstatistiklerNew["PasGecKomutSayisi"] = PasGecKomutSayisi.ToString();
        IstatistiklerNew["KarAlKomutSayisi"] = KarAlKomutSayisi.ToString();
        IstatistiklerNew["ZararKesKomutSayisi"] = ZararKesKomutSayisi.ToString();
        IstatistiklerNew["FlatOlKomutSayisi"] = FlatOlKomutSayisi.ToString();

        // Komisyon
        IstatistiklerNew["KomisyonIslemSayisi"] = KomisyonIslemSayisi.ToString();
        IstatistiklerNew["KomisyonVarlikAdedSayisi"] = KomisyonVarlikAdedSayisi.ToString();
        IstatistiklerNew["KomisyonCarpan"] = KomisyonCarpan.ToString();
        IstatistiklerNew["KomisyonFiyat"] = KomisyonFiyat.ToString();
        IstatistiklerNew["KomisyonFiyatYuzde"] = KomisyonFiyatYuzde.ToString();
        IstatistiklerNew["KomisyonuDahilEt"] = KomisyonuDahilEt.ToString();

        // Kar Zarar  Min Max
        IstatistiklerNew["KarZararFiyat"] = KarZararFiyat.ToString();
        IstatistiklerNew["KarZararFiyatYuzde"] = KarZararFiyatYuzde.ToString();
        IstatistiklerNew["KarZararPuan"] = KarZararPuan.ToString();
        IstatistiklerNew["ToplamKarFiyat"] = ToplamKarFiyat.ToString();
        IstatistiklerNew["ToplamZararFiyat"] = ToplamZararFiyat.ToString();
        IstatistiklerNew["NetKarFiyat"] = NetKarFiyat.ToString();
        IstatistiklerNew["ToplamKarPuan"] = ToplamKarPuan.ToString();
        IstatistiklerNew["ToplamZararPuan"] = ToplamZararPuan.ToString();
        IstatistiklerNew["NetKarPuan"] = NetKarPuan.ToString();
        IstatistiklerNew["MaxKarFiyat"] = MaxKarFiyat.ToString();
        IstatistiklerNew["MaxZararFiyat"] = MaxZararFiyat.ToString();
        IstatistiklerNew["MaxKarPuan"] = MaxKarPuan.ToString();
        IstatistiklerNew["MaxZararPuan"] = MaxZararPuan.ToString();
        IstatistiklerNew["MaxZararFiyatIndex"] = MaxZararFiyatIndex.ToString();
        IstatistiklerNew["MaxKarFiyatIndex"] = MaxKarFiyatIndex.ToString();
        IstatistiklerNew["MaxZararPuanIndex"] = MaxZararPuanIndex.ToString();
        IstatistiklerNew["MaxKarPuanIndex"] = MaxKarPuanIndex.ToString();
        IstatistiklerNew["KardaBarSayisi"] = KardaBarSayisi.ToString();
        IstatistiklerNew["ZarardaBarSayisi"] = ZarardaBarSayisi.ToString();
        IstatistiklerNew["KarliIslemOrani"] = KarliIslemOrani.ToString("0.00");

        // GetiriMaxDD
        IstatistiklerNew["GetiriMaxDD"] = GetiriMaxDD.ToString();
        IstatistiklerNew["GetiriMaxDDTarih"] = GetiriMaxDDTarih.ToString("dd.MM.yyyy");
        IstatistiklerNew["GetiriMaxDDSaat"] = GetiriMaxDDTarih.TimeOfDay.ToString();
        IstatistiklerNew["GetiriMaxKayip"] = GetiriMaxKayip.ToString();

        // ProfitFactor
        IstatistiklerNew["ProfitFactor"] = ProfitFactor.ToString("0.00");
        IstatistiklerNew["ProfitFactorSistem"] = ProfitFactorSistem.ToString("0.00");

        // Ort İslem Sayilari
        IstatistiklerNew["OrtAylikIslemSayisi"] = OrtAylikIslemSayisi.ToString("0.00");
        IstatistiklerNew["OrtHaftalikIslemSayisi"] = OrtHaftalikIslemSayisi.ToString("0.00");
        IstatistiklerNew["OrtGunlukIslemSayisi"] = OrtGunlukIslemSayisi.ToString("0.00");
        IstatistiklerNew["OrtSaatlikIslemSayisi"] = OrtSaatlikIslemSayisi.ToString("0.00");

        // Son Emir İstatistikleri
        IstatistiklerNew["Sinyal"] = Sinyal.ToString();
        IstatistiklerNew["SonYon"] = SonYon.ToString();
        IstatistiklerNew["PrevYon"] = PrevYon.ToString();
        IstatistiklerNew["SonFiyat"] = SonFiyat.ToString();
        IstatistiklerNew["SonAFiyat"] = SonAFiyat.ToString();
        IstatistiklerNew["SonSFiyat"] = SonSFiyat.ToString();
        IstatistiklerNew["SonFFiyat"] = SonFFiyat.ToString();
        IstatistiklerNew["SonPFiyat"] = SonPFiyat.ToString();
        IstatistiklerNew["PrevFiyat"] = PrevFiyat.ToString();
        IstatistiklerNew["PrevAFiyat"] = PrevAFiyat.ToString();
        IstatistiklerNew["PrevSFiyat"] = PrevSFiyat.ToString();
        IstatistiklerNew["PrevFFiyat"] = PrevFFiyat.ToString();
        IstatistiklerNew["PrevPFiyat"] = PrevPFiyat.ToString();
        IstatistiklerNew["SonBarNo"] = SonBarNo.ToString();
        IstatistiklerNew["SonABarNo"] = SonABarNo.ToString();
        IstatistiklerNew["SonSBarNo"] = SonSBarNo.ToString();
        IstatistiklerNew["SonFBarNo"] = SonFBarNo.ToString();
        IstatistiklerNew["SonPBarNo"] = SonPBarNo.ToString();
        IstatistiklerNew["PrevBarNo"] = PrevBarNo.ToString();
        IstatistiklerNew["PrevABarNo"] = PrevABarNo.ToString();
        IstatistiklerNew["PrevSBarNo"] = PrevSBarNo.ToString();
        IstatistiklerNew["PrevFBarNo"] = PrevFBarNo.ToString();
        IstatistiklerNew["PrevPBarNo"] = PrevPBarNo.ToString();
        IstatistiklerNew["EmirKomut"] = EmirKomut.ToString();
        IstatistiklerNew["EmirStatus"] = EmirStatus.ToString();

        // Varlik
        IstatistiklerNew["HisseSayisi"] = HisseSayisi.ToString();

        IstatistiklerNew["KontratSayisi"] = KontratSayisi.ToString();
        IstatistiklerNew["VarlikAdedCarpani"] = VarlikAdedCarpani.ToString();
        IstatistiklerNew["VarlikAdedSayisi"] = VarlikAdedSayisi.ToString();
        IstatistiklerNew["KaymaMiktari"] = KaymaMiktari.ToString();
        IstatistiklerNew["KaymayiDahilEt"] = KaymayiDahilEt.ToString();

        // Getiri Fiyat-Puan Ay Hafta Gun Saat
        IstatistiklerNew["GetiriFiyatBuAy"] = GetiriFiyatBuAy.ToString("0.00");
        IstatistiklerNew["GetiriFiyatAy1"] = GetiriFiyatAy1.ToString("0.00");
        IstatistiklerNew["GetiriFiyatAy2"] = GetiriFiyatAy2.ToString("0.00");
        IstatistiklerNew["GetiriFiyatAy3"] = GetiriFiyatAy3.ToString("0.00");
        IstatistiklerNew["GetiriFiyatAy4"] = GetiriFiyatAy4.ToString("0.00");
        IstatistiklerNew["GetiriFiyatAy5"] = GetiriFiyatAy5.ToString("0.00");

        IstatistiklerNew["GetiriFiyatBuHafta"] = GetiriFiyatBuHafta.ToString("0.00");
        IstatistiklerNew["GetiriFiyatHafta1"] = GetiriFiyatHafta1.ToString("0.00");
        IstatistiklerNew["GetiriFiyatHafta2"] = GetiriFiyatHafta2.ToString("0.00");
        IstatistiklerNew["GetiriFiyatHafta3"] = GetiriFiyatHafta3.ToString("0.00");
        IstatistiklerNew["GetiriFiyatHafta4"] = GetiriFiyatHafta4.ToString("0.00");
        IstatistiklerNew["GetiriFiyatHafta5"] = GetiriFiyatHafta5.ToString("0.00");

        IstatistiklerNew["GetiriFiyatBuGun"] = GetiriFiyatBuGun.ToString("0.00");
        IstatistiklerNew["GetiriFiyatGun1"] = GetiriFiyatGun1.ToString("0.00");
        IstatistiklerNew["GetiriFiyatGun2"] = GetiriFiyatGun2.ToString("0.00");
        IstatistiklerNew["GetiriFiyatGun3"] = GetiriFiyatGun3.ToString("0.00");
        IstatistiklerNew["GetiriFiyatGun4"] = GetiriFiyatGun4.ToString("0.00");
        IstatistiklerNew["GetiriFiyatGun5"] = GetiriFiyatGun5.ToString("0.00");

        IstatistiklerNew["GetiriFiyatBuSaat"] = GetiriFiyatBuSaat.ToString("0.00");
        IstatistiklerNew["GetiriFiyatSaat1"] = GetiriFiyatSaat1.ToString("0.00");
        IstatistiklerNew["GetiriFiyatSaat2"] = GetiriFiyatSaat2.ToString("0.00");
        IstatistiklerNew["GetiriFiyatSaat3"] = GetiriFiyatSaat3.ToString("0.00");
        IstatistiklerNew["GetiriFiyatSaat4"] = GetiriFiyatSaat4.ToString("0.00");
        IstatistiklerNew["GetiriFiyatSaat5"] = GetiriFiyatSaat5.ToString("0.00");

        IstatistiklerNew["GetiriPuanBuAy"] = GetiriPuanBuAy.ToString("0.00");
        IstatistiklerNew["GetiriPuanAy1"] = GetiriPuanAy1.ToString("0.00");
        IstatistiklerNew["GetiriPuanAy2"] = GetiriPuanAy2.ToString("0.00");
        IstatistiklerNew["GetiriPuanAy3"] = GetiriPuanAy3.ToString("0.00");
        IstatistiklerNew["GetiriPuanAy4"] = GetiriPuanAy4.ToString("0.00");
        IstatistiklerNew["GetiriPuanAy5"] = GetiriPuanAy5.ToString("0.00");

        IstatistiklerNew["GetiriPuanBuHafta"] = GetiriPuanBuHafta.ToString("0.00");
        IstatistiklerNew["GetiriPuanHafta1"] = GetiriPuanHafta1.ToString("0.00");
        IstatistiklerNew["GetiriPuanHafta2"] = GetiriPuanHafta2.ToString("0.00");
        IstatistiklerNew["GetiriPuanHafta3"] = GetiriPuanHafta3.ToString("0.00");
        IstatistiklerNew["GetiriPuanHafta4"] = GetiriPuanHafta4.ToString("0.00");
        IstatistiklerNew["GetiriPuanHafta5"] = GetiriPuanHafta5.ToString("0.00");

        IstatistiklerNew["GetiriPuanBuGun"] = GetiriPuanBuGun.ToString("0.00");
        IstatistiklerNew["GetiriPuanGun1"] = GetiriPuanGun1.ToString("0.00");
        IstatistiklerNew["GetiriPuanGun2"] = GetiriPuanGun2.ToString("0.00");
        IstatistiklerNew["GetiriPuanGun3"] = GetiriPuanGun3.ToString("0.00");
        IstatistiklerNew["GetiriPuanGun4"] = GetiriPuanGun4.ToString("0.00");
        IstatistiklerNew["GetiriPuanGun5"] = GetiriPuanGun5.ToString("0.00");

        IstatistiklerNew["GetiriPuanBuSaat"] = GetiriPuanBuSaat.ToString("0.00");
        IstatistiklerNew["GetiriPuanSaat1"] = GetiriPuanSaat1.ToString("0.00");
        IstatistiklerNew["GetiriPuanSaat2"] = GetiriPuanSaat2.ToString("0.00");
        IstatistiklerNew["GetiriPuanSaat3"] = GetiriPuanSaat3.ToString("0.00");
        IstatistiklerNew["GetiriPuanSaat4"] = GetiriPuanSaat4.ToString("0.00");
        IstatistiklerNew["GetiriPuanSaat5"] = GetiriPuanSaat5.ToString("0.00");

//        IstatistiklerNew["ElapsedTimeDataRead"] = ElapsedTimeDataRead;
//        IstatistiklerNew["ElapsedTimeDataCopy"] = ElapsedTimeDataCopy;
//        IstatistiklerNew["ElapsedTimeForLoop"] = ElapsedTimeForLoop;
//        IstatistiklerNew["ElapsedTimeForAnaliz"] = ElapsedTimeForAnaliz;
//        IstatistiklerNew["ElapsedTimeIdealGetiriHesapla"] = ElapsedTimeIdealGetiriHesapla;
//        IstatistiklerNew["ElapsedTime"] = ElapsedTime;
    }

    List<int> colNums = new List<int>();
    List<int> rowNums = new List<int>();
    int PanelNo = 0;
    int ColNum = 0;
    int RowNum = 0;
    int FontSize = 9;
    int col = 0, row = 0, idx = 0;

    public CStatistics Panel(int PanelNo)
    {
        this.PanelNo = PanelNo;
        return this;
    }

    public CStatistics Column(int ColNum)
    {
        this.ColNum = 160 + 80 * ColNum;

        if (this.PanelNo != 1)
            this.ColNum = this.ColNum - 150;

        return this;
    }

    public CStatistics Row(int RowNum)
    {
        this.RowNum = 50 + 25 * RowNum;

        if (this.PanelNo != 1)
            this.RowNum = this.RowNum - 25;

        return this;
    }

    public CStatistics TextMessage(dynamic Sistem, string Message, Color MessageColor)
    {
        Sistem.ZeminYazisiEkle(Message, this.PanelNo, this.ColNum,this.RowNum, MessageColor, "Tahoma", FontSize);

        return this;
    }

    public string GetFormattedString(string Alias, string Tab, string Key)
    {
        string str = string.Format("{0}{1}{2}", Alias, Tab + ": ", IstatistiklerNew[Key] );

        return str;
    }

    public void ekranKoordianatlariniGoster(dynamic Sistem, int PanelNo = 1)
    {
        int ColNum = 0;
        string str;

        for (int col = 0; col < 20; col++)
        {
            for (int row = 0; row < 10; row++)
            {
                str = string.Format("{0}, {1}", col.ToString(""), row.ToString(""));
                Panel(PanelNo).Column(col).Row(row);

                str = string.Format("{0}, {1}", this.ColNum.ToString(""), this.RowNum.ToString(""));
                TextMessage(Sistem, str, Color.Gold);
            }
        }
    }

    public void IstatistikleriEkranaYaz(dynamic Sistem, int PanelNo = 1)
    {
        var str = "";
        string tab0 = "  ";
        string tab1 = "\t";
        string tab2 = "\t\t";
        string tab3 = "\t\t\t";
        var seperator = "----------------------------------------";
         // {0,-5} \t {1,-20}

        List<string> strList = new List<string>();

        strList.Clear();

        str = GetFormattedString("Profit Factor",           tab2, "ProfitFactor");              strList.Add( str );
        str = GetFormattedString("Profit Factor Sistem",    tab1, "ProfitFactorSistem");        strList.Add( str );
        str = GetFormattedString("Karli Islem Orani",       tab1, "KarliIslemOrani");           strList.Add( str );
        str = GetFormattedString("GetiriKz Sistem",         tab2, "GetiriKzSistem");            strList.Add( str );
        str = GetFormattedString("GetiriKz Sistem  (%)",    tab1, "GetiriKzSistemYuzde");       strList.Add( str );
                                                                                                strList.Add( str = seperator );
        str = GetFormattedString("Son Çalışma Zamani",      tab1, "LastExecutionTime");         strList.Add( str );
        str = GetFormattedString("Çalışma Süresi (ms)",     tab1, "ExecutionTimeInMSec");       strList.Add( str );



        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(0).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        strList.Clear();

        str = GetFormattedString("IlkBakiyeFiyat",          tab2, "IlkBakiyeFiyat");            strList.Add( str );
        str = GetFormattedString("BakiyeFiyat",             tab2, "BakiyeFiyat");               strList.Add( str );
                                                                                                strList.Add( str = seperator );
        str = GetFormattedString("GetiriFiyat",             tab2, "GetiriFiyat");               strList.Add( str );
        str = GetFormattedString("GetiriFiyatYuzde (%)",    tab1, "GetiriFiyatYuzde");          strList.Add( str );
        str = GetFormattedString("GetiriFiyatTipi",         tab2, "GetiriFiyatTipi");           strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(3).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        strList.Clear();

        str = GetFormattedString("Komisyon",                tab2, "KomisyonFiyat");             strList.Add( str );
        str = GetFormattedString("KomisyonYuzde (%)",       tab1, "KomisyonFiyatYuzde");        strList.Add( str );
                                                                                                strList.Add( str = seperator );
        str = GetFormattedString("GetiriFiyatNet",          tab2, "GetiriFiyatNet");            strList.Add( str );
        str = GetFormattedString("GetiriFiyatYuzdeNet (%)", tab0, "GetiriFiyatYuzdeNet");       strList.Add( str );
                                                                                                strList.Add( str = seperator );
        str = GetFormattedString("HisseSayisi",             tab2, "HisseSayisi");               strList.Add( str );
        str = GetFormattedString("KontratSayisi",           tab2, "KontratSayisi");             strList.Add( str );
        str = GetFormattedString("KomisyonVarlikSayisi",    tab1, "KomisyonVarlikAdedSayisi");  strList.Add( str );
        str = GetFormattedString("VarlikAdedCarpani",       tab1, "VarlikAdedCarpani");         strList.Add( str );
        str = GetFormattedString("VarlikAdedSayisi",        tab1, "VarlikAdedSayisi");          strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(6).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        strList.Clear();

        str = GetFormattedString("KomisyonIslemSayisi",     tab1, "KomisyonIslemSayisi");       strList.Add( str );
        str = GetFormattedString("ToplamIslemSayisi",       tab1, "IslemSayisi");               strList.Add( str );
        str = GetFormattedString("AlisSayisi",              tab2, "AlisSayisi");                strList.Add( str );
        str = GetFormattedString("SatisSayisi",             tab2, "SatisSayisi");               strList.Add( str );
        str = GetFormattedString("FlatSayisi",              tab2, "FlatSayisi");                strList.Add( str );
        str = GetFormattedString("PassSayisi",              tab2, "PassSayisi");                strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(9).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        strList.Clear();

        str = GetFormattedString("OrtAylikİslemSayisi",     tab1, "OrtAylikIslemSayisi");       strList.Add( str );
        str = GetFormattedString("OrtGunlukİslemSayisi",    tab1, "OrtGunlukIslemSayisi");      strList.Add( str );
                                                                                                strList.Add( str = seperator );
        str = GetFormattedString("KazandiranIslemSayisi",   tab1, "KazandiranIslemSayisi");     strList.Add( str );
        str = GetFormattedString("KaybettirenIslemSayisi",  tab0, "KaybettirenIslemSayisi");    strList.Add( str );
        str = GetFormattedString("NotrIslemSayisi",         tab1, "NotrIslemSayisi");           strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(12).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        strList.Clear();

        str = GetFormattedString("ToplamKarFiyat",          tab1, "ToplamKarFiyat");            strList.Add( str );
        str = GetFormattedString("ToplamZararFiyat",        tab1, "ToplamZararFiyat");          strList.Add( str );
        str = GetFormattedString("NetKarFiyat",             tab2, "NetKarFiyat");               strList.Add( str );
                                                                                                strList.Add( str = seperator );
        str = GetFormattedString("MaxKarFiyat",             tab2, "MaxKarFiyat");               strList.Add( str );
        str = GetFormattedString("MaxZararFiyat",           tab2, "MaxZararFiyat");             strList.Add( str );
        str = GetFormattedString("MaxKarFiyatIndex",        tab1, "MaxKarFiyatIndex");          strList.Add( str );
        str = GetFormattedString("MaxZararFiyatIndex",      tab1, "MaxZararFiyatIndex");        strList.Add( str );
                                                                                                strList.Add( str = seperator );
        str = GetFormattedString("MinBakiyeFiyat",          tab2, "MinBakiyeFiyat");            strList.Add( str );
        str = GetFormattedString("MaxBakiyeFiyat",          tab1, "MaxBakiyeFiyat");            strList.Add( str );
        str = GetFormattedString("MinBakiyeFiyatIndex",     tab1, "MinBakiyeFiyatIndex");       strList.Add( str );
        str = GetFormattedString("MaxBakiyeFiyatIndex",     tab1, "MaxBakiyeFiyatIndex");       strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(15).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        strList.Clear();

        str = GetFormattedString("GetiriMaxDD (Puan)",      tab1, "GetiriMaxDD");               strList.Add( str );
        str = GetFormattedString("GetiriMaxDDTarih",        tab1, "GetiriMaxDDTarih");          strList.Add( str );
        str = GetFormattedString("GetiriMaxDDSaat",         tab1, "GetiriMaxDDSaat");           strList.Add( str );
        str = GetFormattedString("GetiriMaxKayip",          tab2, "GetiriMaxKayip");            strList.Add( str );
                                                                                                strList.Add( str = seperator );
        str = GetFormattedString("Toplam Bar Sayisi",       tab1, "ToplamBarSayisi");           strList.Add( str );
        str = GetFormattedString("IlkBar Tarihi",           tab2, "IlkBarTarihi");              strList.Add( str );
        str = GetFormattedString("IlkBar Saati",            tab2, "IlkBarSaati");               strList.Add( str );
                                                                                                strList.Add( str = seperator );
        str = GetFormattedString("SonBar Tarihi",           tab2, "SonBarTarihi");              strList.Add( str );
        str = GetFormattedString("SonBar Saati",            tab2, "SonBarSaati");               strList.Add( str );
        str = GetFormattedString("Toplam Sure Ay",          tab1, "ToplamGecenSureAy");         strList.Add( str );
        str = GetFormattedString("Toplam Sure Gun",         tab1, "ToplamGecenSureGun");        strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(18).Row(i).TextMessage(Sistem, strList[i], Color.Gold);
    }

    public void IstatistikleriDosyayaYaz(dynamic Sistem, string FileName)
    {
        string delimiter = ";";

        CFileUtils myFileUtils = new CFileUtils();

        string logFileFullName = FileName.Trim();

        string logMessage = "";

        myFileUtils.Reset(Sistem).EnableLogging(Sistem).OpenLogFile(Sistem, logFileFullName, false, false);
        //myFileUtils.Reset(Sistem).DisableLogging(Sistem).OpenLogFile(Sistem, logFileName, false, false);

        logMessage = String.Format("{0} {1} {2}", "Kayit Zamani", delimiter, DateTime.Now.ToString("yyyy.MM.dd HH:mm:ss"));
        myFileUtils.WriteToLogFile(Sistem, logMessage);

        foreach (var entry in IstatistiklerNew)
        {
            logMessage = String.Format("{0} ; {1} ; \t", entry.Key.Trim(), entry.Value);
            myFileUtils.WriteToLogFile(Sistem, logMessage);
        }

        myFileUtils.CloseLogFile(Sistem);
    }

    private void GetiriIstatiskleriHesapla(dynamic Sistem)
    {
        var V = Sistem.GrafikVerileri;

        List<int> DateAyBarNoList = new List<int>();
        List<int> DateHaftaBarNoList = new List<int>();
        List<int> DateGunBarNoList = new List<int>();
        List<int> DateSaatBarNoList = new List<int>();
        List<int> DateXDakikaBarNoList = new List<int>();

        List<double> gunKzList = new List<double>();

        int weekNumber = 0;
        CTimeUtils timeUtils = new CTimeUtils();

        // Buradan devam edilecek!
        for (int i = 1; i < V.Count; i++)
        {
            bool yeniSaat = (V[i - 1].Date.Hour != V[i].Date.Hour);
            if (yeniSaat)
                DateSaatBarNoList.Add(i);

            bool yeniGun = (V[i - 1].Date.Day != V[i].Date.Day);
            if (yeniGun)
                DateGunBarNoList.Add(i);

            bool yeniHafta = yeniGun && (timeUtils.GetWeekNumber(Sistem, V[i - 1].Date) != timeUtils.GetWeekNumber(Sistem, V[i].Date));
            if (yeniHafta)
                DateHaftaBarNoList.Add(i);

            bool yeniAy = (V[i - 1].Date.Month != V[i].Date.Month);
            if (yeniAy)
                DateAyBarNoList.Add(i);
        }

        // -------------------------------------------------------------------------------

        var KZList = Sistem.Liste(0);

        for (int i = 0; i < V.Count; i++)
        {
            KZList[i] = Sistem.GetiriKZ[i];
        }

        // Reset icinde de sifirlandi
        GetiriPuanBuAy = 0f;
        GetiriPuanAy1 = 0f;
        GetiriPuanAy2 = 0f;
        GetiriPuanAy3 = 0f;
        GetiriPuanAy4 = 0f;
        GetiriPuanAy5 = 0f;

        GetiriPuanBuHafta = 0f;
        GetiriPuanHafta1 = 0f;
        GetiriPuanHafta2 = 0f;
        GetiriPuanHafta3 = 0f;
        GetiriPuanHafta4 = 0f;
        GetiriPuanHafta5 = 0f;

        GetiriPuanBuGun = 0f;
        GetiriPuanGun1 = 0f;
        GetiriPuanGun2 = 0f;
        GetiriPuanGun3 = 0f;
        GetiriPuanGun4 = 0f;
        GetiriPuanGun5 = 0f;

        GetiriPuanBuSaat = 0f;
        GetiriPuanSaat1 = 0f;
        GetiriPuanSaat2 = 0f;
        GetiriPuanSaat3 = 0f;
        GetiriPuanSaat4 = 0f;
        GetiriPuanSaat5 = 0f;

        // GunlukveAylikGetiriEgrisiBilgileriniHesapla
        var aykz1 = 0.0;
        var aykz2 = 0.0;
        var aykz3 = 0.0;
        var aykz4 = 0.0;
        var aykz5 = 0.0;

        if (DateAyBarNoList.Count > 5)
        {
            aykz1 = KZList[KZList.Count - 1] - 1 * KZList[DateAyBarNoList[DateAyBarNoList.Count - 1]];
            aykz2 = KZList[KZList.Count - 1] - 1 * KZList[DateAyBarNoList[DateAyBarNoList.Count - 2]];
            aykz3 = KZList[KZList.Count - 1] - 1 * KZList[DateAyBarNoList[DateAyBarNoList.Count - 3]];
            aykz4 = KZList[KZList.Count - 1] - 1 * KZList[DateAyBarNoList[DateAyBarNoList.Count - 4]];
            aykz5 = KZList[KZList.Count - 1] - 1 * KZList[DateAyBarNoList[DateAyBarNoList.Count - 5]];
        }

        var haftakz1 = 0.0;
        var haftakz2 = 0.0;
        var haftakz3 = 0.0;
        var haftakz4 = 0.0;
        var haftakz5 = 0.0;

        if (DateHaftaBarNoList.Count > 5)
        {
            haftakz1 = KZList[KZList.Count - 1] - 1 * KZList[DateHaftaBarNoList[DateHaftaBarNoList.Count - 1]];
            haftakz2 = KZList[KZList.Count - 1] - 1 * KZList[DateHaftaBarNoList[DateHaftaBarNoList.Count - 2]];
            haftakz3 = KZList[KZList.Count - 1] - 1 * KZList[DateHaftaBarNoList[DateHaftaBarNoList.Count - 3]];
            haftakz4 = KZList[KZList.Count - 1] - 1 * KZList[DateHaftaBarNoList[DateHaftaBarNoList.Count - 4]];
            haftakz5 = KZList[KZList.Count - 1] - 1 * KZList[DateHaftaBarNoList[DateHaftaBarNoList.Count - 5]];
        }

        var gunkz1 = 0.0;
        var gunkz2 = 0.0;
        var gunkz3 = 0.0;
        var gunkz4 = 0.0;
        var gunkz5 = 0.0;

        if (DateGunBarNoList.Count > 5)
        {
            gunkz1 = KZList[KZList.Count - 1] - 1 * KZList[DateGunBarNoList[DateGunBarNoList.Count - 1]];
            gunkz2 = KZList[KZList.Count - 1] - 1 * KZList[DateGunBarNoList[DateGunBarNoList.Count - 2]];
            gunkz3 = KZList[KZList.Count - 1] - 1 * KZList[DateGunBarNoList[DateGunBarNoList.Count - 3]];
            gunkz4 = KZList[KZList.Count - 1] - 1 * KZList[DateGunBarNoList[DateGunBarNoList.Count - 4]];
            gunkz5 = KZList[KZList.Count - 1] - 1 * KZList[DateGunBarNoList[DateGunBarNoList.Count - 5]];
        }

        var saatkz1 = 0.0;
        var saatkz2 = 0.0;
        var saatkz3 = 0.0;
        var saatkz4 = 0.0;
        var saatkz5 = 0.0;

        if (DateSaatBarNoList.Count > 5)
        {
            //                                       İlk degil, ikinci elamandan basla
            saatkz1 = KZList[KZList.Count - 1] - 1 * KZList[DateSaatBarNoList[DateSaatBarNoList.Count - 2]];
            saatkz2 = KZList[KZList.Count - 1] - 1 * KZList[DateSaatBarNoList[DateSaatBarNoList.Count - 3]];
            saatkz3 = KZList[KZList.Count - 1] - 1 * KZList[DateSaatBarNoList[DateSaatBarNoList.Count - 4]];
            saatkz4 = KZList[KZList.Count - 1] - 1 * KZList[DateSaatBarNoList[DateSaatBarNoList.Count - 5]];
            saatkz5 = KZList[KZList.Count - 1] - 1 * KZList[DateSaatBarNoList[DateSaatBarNoList.Count - 6]];
        }

        GetiriPuanBuAy = Convert.ToSingle(aykz1);
        GetiriPuanAy1 = Convert.ToSingle(aykz1);
        GetiriPuanAy2 = Convert.ToSingle(aykz2);
        GetiriPuanAy3 = Convert.ToSingle(aykz3);
        GetiriPuanAy4 = Convert.ToSingle(aykz4);
        GetiriPuanAy5 = Convert.ToSingle(aykz5);

        GetiriPuanBuHafta = Convert.ToSingle(haftakz1);
        GetiriPuanHafta1 = Convert.ToSingle(haftakz1);
        GetiriPuanHafta2 = Convert.ToSingle(haftakz2);
        GetiriPuanHafta3 = Convert.ToSingle(haftakz3);
        GetiriPuanHafta4 = Convert.ToSingle(haftakz4);
        GetiriPuanHafta5 = Convert.ToSingle(haftakz5);

        GetiriPuanBuGun = Convert.ToSingle(gunkz1);
        GetiriPuanGun1 = Convert.ToSingle(gunkz1);
        GetiriPuanGun2 = Convert.ToSingle(gunkz2);
        GetiriPuanGun3 = Convert.ToSingle(gunkz3);
        GetiriPuanGun4 = Convert.ToSingle(gunkz4);
        GetiriPuanGun5 = Convert.ToSingle(gunkz5);

        GetiriPuanBuSaat = Convert.ToSingle(saatkz1);
        GetiriPuanSaat1 = Convert.ToSingle(saatkz1);
        GetiriPuanSaat2 = Convert.ToSingle(saatkz2);
        GetiriPuanSaat3 = Convert.ToSingle(saatkz3);
        GetiriPuanSaat4 = Convert.ToSingle(saatkz4);
        GetiriPuanSaat5 = Convert.ToSingle(saatkz5);

        // -------------------------------------------------------------------------------

        for (int i = 0; i < V.Count; i++)
        {
            KZList[i] = Trader.Lists.GetiriFiyatList[i];    // Fiyat bazinda toplam getiri
            // KZList[i] = Trader.Lists.GetiriKz[i];        // Fiyat bazinda birim  getiri
        }

        // Reset icinde de sifirlandi
        GetiriFiyatBuAy = 0f;
        GetiriFiyatAy1 = 0f;
        GetiriFiyatAy2 = 0f;
        GetiriFiyatAy3 = 0f;
        GetiriFiyatAy4 = 0f;
        GetiriFiyatAy5 = 0f;

        GetiriFiyatBuHafta = 0f;
        GetiriFiyatHafta1 = 0f;
        GetiriFiyatHafta2 = 0f;
        GetiriFiyatHafta3 = 0f;
        GetiriFiyatHafta4 = 0f;
        GetiriFiyatHafta5 = 0f;

        GetiriFiyatBuGun = 0f;
        GetiriFiyatGun1 = 0f;
        GetiriFiyatGun2 = 0f;
        GetiriFiyatGun3 = 0f;
        GetiriFiyatGun4 = 0f;
        GetiriFiyatGun5 = 0f;

        GetiriFiyatBuSaat = 0f;
        GetiriFiyatSaat1 = 0f;
        GetiriFiyatSaat2 = 0f;
        GetiriFiyatSaat3 = 0f;
        GetiriFiyatSaat4 = 0f;
        GetiriFiyatSaat5 = 0f;

        // GunlukveAylikGetiriEgrisiBilgileriniHesapla
        aykz1 = 0.0;
        aykz2 = 0.0;
        aykz3 = 0.0;
        aykz4 = 0.0;
        aykz5 = 0.0;

        if (DateAyBarNoList.Count > 5)
        {
            aykz1 = KZList[KZList.Count - 1] - 1 * KZList[DateAyBarNoList[DateAyBarNoList.Count - 1]];
            aykz2 = KZList[KZList.Count - 1] - 1 * KZList[DateAyBarNoList[DateAyBarNoList.Count - 2]];
            aykz3 = KZList[KZList.Count - 1] - 1 * KZList[DateAyBarNoList[DateAyBarNoList.Count - 3]];
            aykz4 = KZList[KZList.Count - 1] - 1 * KZList[DateAyBarNoList[DateAyBarNoList.Count - 4]];
            aykz5 = KZList[KZList.Count - 1] - 1 * KZList[DateAyBarNoList[DateAyBarNoList.Count - 5]];
        }

        haftakz1 = 0.0;
        haftakz2 = 0.0;
        haftakz3 = 0.0;
        haftakz4 = 0.0;
        haftakz5 = 0.0;

        if (DateHaftaBarNoList.Count > 5)
        {
            haftakz1 = KZList[KZList.Count - 1] - 1 * KZList[DateHaftaBarNoList[DateHaftaBarNoList.Count - 1]];
            haftakz2 = KZList[KZList.Count - 1] - 1 * KZList[DateHaftaBarNoList[DateHaftaBarNoList.Count - 2]];
            haftakz3 = KZList[KZList.Count - 1] - 1 * KZList[DateHaftaBarNoList[DateHaftaBarNoList.Count - 3]];
            haftakz4 = KZList[KZList.Count - 1] - 1 * KZList[DateHaftaBarNoList[DateHaftaBarNoList.Count - 4]];
            haftakz5 = KZList[KZList.Count - 1] - 1 * KZList[DateHaftaBarNoList[DateHaftaBarNoList.Count - 5]];
        }

        gunkz1 = 0.0;
        gunkz2 = 0.0;
        gunkz3 = 0.0;
        gunkz4 = 0.0;
        gunkz5 = 0.0;

        if (DateGunBarNoList.Count > 5)
        {
            gunkz1 = KZList[KZList.Count - 1] - 1 * KZList[DateGunBarNoList[DateGunBarNoList.Count - 1]];
            gunkz2 = KZList[KZList.Count - 1] - 1 * KZList[DateGunBarNoList[DateGunBarNoList.Count - 2]];
            gunkz3 = KZList[KZList.Count - 1] - 1 * KZList[DateGunBarNoList[DateGunBarNoList.Count - 3]];
            gunkz4 = KZList[KZList.Count - 1] - 1 * KZList[DateGunBarNoList[DateGunBarNoList.Count - 4]];
            gunkz5 = KZList[KZList.Count - 1] - 1 * KZList[DateGunBarNoList[DateGunBarNoList.Count - 5]];
        }

        saatkz1 = 0.0;
        saatkz2 = 0.0;
        saatkz3 = 0.0;
        saatkz4 = 0.0;
        saatkz5 = 0.0;

        if (DateSaatBarNoList.Count > 5)
        {
            //                                       İlk degil, ikinci elamandan basla
            saatkz1 = KZList[KZList.Count - 1] - 1 * KZList[DateSaatBarNoList[DateSaatBarNoList.Count - 2]];
            saatkz2 = KZList[KZList.Count - 1] - 1 * KZList[DateSaatBarNoList[DateSaatBarNoList.Count - 3]];
            saatkz3 = KZList[KZList.Count - 1] - 1 * KZList[DateSaatBarNoList[DateSaatBarNoList.Count - 4]];
            saatkz4 = KZList[KZList.Count - 1] - 1 * KZList[DateSaatBarNoList[DateSaatBarNoList.Count - 5]];
            saatkz5 = KZList[KZList.Count - 1] - 1 * KZList[DateSaatBarNoList[DateSaatBarNoList.Count - 6]];
        }

        GetiriFiyatBuAy = Convert.ToSingle(aykz1);
        GetiriFiyatAy1 = Convert.ToSingle(aykz1);
        GetiriFiyatAy2 = Convert.ToSingle(aykz2);
        GetiriFiyatAy3 = Convert.ToSingle(aykz3);
        GetiriFiyatAy4 = Convert.ToSingle(aykz4);
        GetiriFiyatAy5 = Convert.ToSingle(aykz5);

        GetiriFiyatBuHafta = Convert.ToSingle(haftakz1);
        GetiriFiyatHafta1 = Convert.ToSingle(haftakz1);
        GetiriFiyatHafta2 = Convert.ToSingle(haftakz2);
        GetiriFiyatHafta3 = Convert.ToSingle(haftakz3);
        GetiriFiyatHafta4 = Convert.ToSingle(haftakz4);
        GetiriFiyatHafta5 = Convert.ToSingle(haftakz5);

        GetiriFiyatBuGun = Convert.ToSingle(gunkz1);
        GetiriFiyatGun1 = Convert.ToSingle(gunkz1);
        GetiriFiyatGun2 = Convert.ToSingle(gunkz2);
        GetiriFiyatGun3 = Convert.ToSingle(gunkz3);
        GetiriFiyatGun4 = Convert.ToSingle(gunkz4);
        GetiriFiyatGun5 = Convert.ToSingle(gunkz5);

        GetiriFiyatBuSaat = Convert.ToSingle(saatkz1);
        GetiriFiyatSaat1 = Convert.ToSingle(saatkz1);
        GetiriFiyatSaat2 = Convert.ToSingle(saatkz2);
        GetiriFiyatSaat3 = Convert.ToSingle(saatkz3);
        GetiriFiyatSaat4 = Convert.ToSingle(saatkz4);
        GetiriFiyatSaat5 = Convert.ToSingle(saatkz5);
    }

    public void GetiriIstatistikleriEkranaYaz(dynamic Sistem, int PanelNo = 2)
    {
        var str = "";
        string tab0 = "  ";
        string tab1 = "\t";
        string tab2 = "\t\t";
        string tab3 = "\t\t\t";
        var seperator = "----------------------------------------";

        List<string> strList = new List<string>();

        strList.Clear();

        str = string.Format("{0} : {1} / {2} : {3}", "HisseSayisi", IstatistiklerNew["HisseSayisi"], "KontratSayisi", IstatistiklerNew["KontratSayisi"] );
        strList.Add( str );
        str = string.Format("{0} : {1}", "KomisyonVarlikAdedSayisi", IstatistiklerNew["KomisyonVarlikAdedSayisi"] );
        strList.Add( str );
        str = string.Format("{0} : {1}", "VarlikAdedCarpani", IstatistiklerNew["VarlikAdedCarpani"] );
        strList.Add( str );
        str = string.Format("{0} : {1}", "VarlikAdedSayisi", IstatistiklerNew["VarlikAdedSayisi"] );
        strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(0).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        int ColumnOffset = 12;
/*
        strList.Clear();

        str = string.Format("{0}", " ");                                        strList.Add( str );
        str = string.Format("{0}", "----------");                               strList.Add( str );
        str = string.Format("{0}", "Bu");                                       strList.Add( str );
        str = string.Format("{0}", "Son 2");                                    strList.Add( str );
        str = string.Format("{0}", "Son 3");                                    strList.Add( str );
        str = string.Format("{0}", "Son 4");                                    strList.Add( str );
        str = string.Format("{0}", "Son 5");                                    strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(ColumnOffset+0).Row(i).TextMessage(Sistem, strList[i], Color.Gold);
*/
        strList.Clear();

        str = string.Format("{0}", "Puan");                                     strList.Add( str );
        str = string.Format("{0}", "----------");                               strList.Add( str );
        str = string.Format("{0}", "Bu");                                       strList.Add( str );
        str = string.Format("{0}", "Son 2");                                    strList.Add( str );
        str = string.Format("{0}", "Son 3");                                    strList.Add( str );
        str = string.Format("{0}", "Son 4");                                    strList.Add( str );
        str = string.Format("{0}", "Son 5");                                    strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(ColumnOffset+1).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        strList.Clear();

        str = string.Format("{0}", "Ay");                                       strList.Add( str );
        str = string.Format("{0}", "----------");                               strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanBuAy"]);         strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanAy2"]);          strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanAy3"]);          strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanAy4"]);          strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanAy5"]);          strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(ColumnOffset+2).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        strList.Clear();

        str = string.Format("{0}", "Hafta");                                    strList.Add( str );
        str = string.Format("{0}", "----------");                               strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanBuHafta"]);      strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanHafta2"]);       strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanHafta3"]);       strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanHafta4"]);       strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanHafta5"]);       strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(ColumnOffset+3).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        strList.Clear();

        str = string.Format("{0}", "Gün");                                      strList.Add( str );
        str = string.Format("{0}", "----------");                               strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanBuGun"]);        strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanGun2"]);         strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanGun3"]);         strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanGun4"]);         strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanGun5"]);         strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(ColumnOffset+4).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        strList.Clear();

        str = string.Format("{0}", "Saat");                                     strList.Add( str );
        str = string.Format("{0}", "----------");                               strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanBuSaat"]);       strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanSaat2"]);        strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanSaat3"]);        strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanSaat4"]);        strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriPuanSaat5"]);        strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(ColumnOffset+5).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        strList.Clear();

        str = string.Format("{0}", "Fiyat");                                    strList.Add( str );
        str = string.Format("{0}", "----------");                               strList.Add( str );
        str = string.Format("{0}", "Bu");                                       strList.Add( str );
        str = string.Format("{0}", "Son 2");                                    strList.Add( str );
        str = string.Format("{0}", "Son 3");                                    strList.Add( str );
        str = string.Format("{0}", "Son 4");                                    strList.Add( str );
        str = string.Format("{0}", "Son 5");                                    strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(ColumnOffset+6).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        strList.Clear();

        str = string.Format("{0}", "Ay");                                       strList.Add( str );
        str = string.Format("{0}", "----------");                               strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatBuAy"]);        strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatAy2"]);         strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatAy3"]);         strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatAy4"]);         strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatAy5"]);         strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(ColumnOffset+7).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        strList.Clear();

        str = string.Format("{0}", "Hafta");                                    strList.Add( str );
        str = string.Format("{0}", "----------");                               strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatBuHafta"]);     strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatHafta2"]);      strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatHafta3"]);      strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatHafta4"]);      strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatHafta5"]);      strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(ColumnOffset+8).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        strList.Clear();

        str = string.Format("{0}", "Gün");                                      strList.Add( str );
        str = string.Format("{0}", "----------");                               strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatBuGun"]);       strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatGun2"]);        strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatGun3"]);        strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatGun4"]);        strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatGun5"]);        strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(ColumnOffset+9).Row(i).TextMessage(Sistem, strList[i], Color.Gold);

        strList.Clear();

        str = string.Format("{0}", "Saat");                                     strList.Add( str );
        str = string.Format("{0}", "----------");                               strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatBuSaat"]);      strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatSaat2"]);       strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatSaat3"]);       strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatSaat4"]);       strList.Add( str );
        str = string.Format("{0}", IstatistiklerNew["GetiriFiyatSaat5"]);       strList.Add( str );

        for (int i = 0; i < strList.Count; i++) Panel(PanelNo).Column(ColumnOffset+10).Row(i).TextMessage(Sistem, strList[i], Color.Gold);
    }

    public void OptimizasyonIstatistiklerininBasliklariniDosyayaYaz(dynamic Sistem, string FileName)
    {
        CTxtFileWriter fileWriter = new CTxtFileWriter();

        fileWriter.OpenFile(FileName, true);

        if (!fileWriter.IsOpened())
            return;

        string delimiter = ";";

        string str = "";

        List<string> strList = new List<string>();

        strList.Clear();

        str = String.Format("{0} {1} ", "Index", delimiter);                                            strList.Add( str );
        str = String.Format("{0} {1} ", "TotalCount", delimiter);                                       strList.Add( str );

        str = String.Format("{0} {1} ", "GetiriKzSistem", delimiter);                                   strList.Add( str );

        str = String.Format("{0} {1} ", "GetiriKz", delimiter);                                         strList.Add( str );
        str = String.Format("{0} {1} ", "GetiriKzNet", delimiter);                                      strList.Add( str );

        str = String.Format("{0} {1} ", "ProfitFactor", delimiter);                                     strList.Add( str );
        str = String.Format("{0} {1} ", "ProfitFactorSistem", delimiter);                               strList.Add( str );

        str = String.Format("{0} {1} ", "KarliIslemOrani", delimiter);                                  strList.Add( str );

        str = String.Format("{0} {1} ", "IslemSayisi", delimiter);                                      strList.Add( str );
        str = String.Format("{0} {1} ", "KomisyonIslemSayisi", delimiter);                              strList.Add( str );

        str = String.Format("{0} {1} ", "KomisyonFiyat", delimiter);                                    strList.Add( str );
        str = String.Format("{0} {1} ", "KomisyonFiyatYuzde", delimiter);                               strList.Add( str );

        str = String.Format("{0} {1} ", "IlkBakiyeFiyat", delimiter);                                   strList.Add( str );

        str = String.Format("{0} {1} ", "BakiyeFiyat", delimiter);                                      strList.Add( str );
        str = String.Format("{0} {1} ", "GetiriFiyat", delimiter);                                      strList.Add( str );
        str = String.Format("{0} {1} ", "GetiriFiyatYuzde", delimiter);                                 strList.Add( str );

        str = String.Format("{0} {1} ", "BakiyeFiyatNet", delimiter);                                   strList.Add( str );
        str = String.Format("{0} {1} ", "GetiriFiyatNet", delimiter);                                   strList.Add( str );
        str = String.Format("{0} {1} ", "GetiriFiyatYuzdeNet", delimiter);                              strList.Add( str );

        str = String.Format("{0} {1} ", "MinBakiyeFiyat", delimiter);                                   strList.Add( str );
        str = String.Format("{0} {1} ", "MaxBakiyeFiyat", delimiter);                                   strList.Add( str );

        str = String.Format("{0} {1} ", "MinBakiyeFiyatYuzde", delimiter);                              strList.Add( str );
        str = String.Format("{0} {1} ", "MaxBakiyeFiyatYuzde", delimiter);                              strList.Add( str );

        str = String.Format("{0} {1} ", "MaxKarFiyat", delimiter);                                      strList.Add( str );
        str = String.Format("{0} {1} ", "MaxZararFiyat", delimiter);                                    strList.Add( str );
        str = String.Format("{0} {1} ", "MaxKarPuan", delimiter);                                       strList.Add( str );
        str = String.Format("{0} {1} ", "MaxZararPuan", delimiter);                                     strList.Add( str );

        str = String.Format("{0} {1} ", "KazandiranIslemSayisi", delimiter);                            strList.Add( str );
        str = String.Format("{0} {1} ", "KaybettirenIslemSayisi", delimiter);                           strList.Add( str );

        str = String.Format("{0} {1} ", "ToplamKarFiyat", delimiter);                                   strList.Add( str );
        str = String.Format("{0} {1} ", "ToplamZararFiyat", delimiter);                                 strList.Add( str );
        str = String.Format("{0} {1} ", "NetKarFiyat", delimiter);                                      strList.Add( str );

        str = String.Format("{0} {1} ", "ToplamKarPuan", delimiter);                                    strList.Add( str );
        str = String.Format("{0} {1} ", "ToplamZararPuan", delimiter);                                  strList.Add( str );
        str = String.Format("{0} {1} ", "NetKarPuan", delimiter);                                       strList.Add( str );

        str = String.Format("{0} {1} ", "GetiriMaxDD", delimiter);                                      strList.Add( str );
        str = String.Format("{0} {1} ", "GetiriMaxDDTarih", delimiter);                                 strList.Add( str );
        str = String.Format("{0} {1} ", "GetiriMaxDDSaat", delimiter);                                  strList.Add( str );
        str = String.Format("{0} {1} ", "GetiriMaxKayip", delimiter);                                   strList.Add( str );

        str = String.Format("{0} {1} ", "OrtAylikIslemSayisi", delimiter);                              strList.Add( str );
        str = String.Format("{0} {1} ", "OrtHaftalikIslemSayisi", delimiter);                           strList.Add( str );
        str = String.Format("{0} {1} ", "OrtGunlukIslemSayisi", delimiter);                             strList.Add( str );

        str = String.Format("{0} {1} ", "Eklenme Zamani", delimiter);                                   strList.Add( str );

/*
        str = String.Format("{0} {1} ", IstatistiklerNew["GrafikSembol"], delimiter);                   strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GrafikPeriyot"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["SistemName"], delimiter);                     strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriKzSistem"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriKzSistemYuzde"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriKzNetSistem"], delimiter);              strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriKzNetSistemYuzde"], delimiter);         strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["IlkBakiyeFiyat"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["BakiyeFiyat"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriFiyat"], delimiter);                    strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriFiyatYuzde"], delimiter);               strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["BakiyeFiyatNet"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriFiyatNet"], delimiter);                 strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriFiyatYuzdeNet"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriKz"], delimiter);                       strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriKzNet"], delimiter);                    strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["MinBakiyeFiyat"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxBakiyeFiyat"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MinBakiyeFiyatYuzde"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxBakiyeFiyatYuzde"], delimiter);            strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["MinBakiyeFiyatIndex"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxBakiyeFiyatIndex"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MinBakiyeFiyatNet"], delimiter);              strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxBakiyeFiyatNet"], delimiter);              strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["MinBakiyeFiyatNetIndex"], delimiter);         strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxBakiyeFiyatNetIndex"], delimiter);         strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MinBakiyeFiyatNetYuzde"], delimiter);         strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxBakiyeFiyatNetYuzde"], delimiter);         strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["IslemSayisi"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["AlisSayisi"], delimiter);                     strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["SatisSayisi"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["FlatSayisi"], delimiter);                     strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["PassSayisi"], delimiter);                     strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KarAlSayisi"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ZararKesSayisi"], delimiter);                 strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["KazandiranIslemSayisi"], delimiter);          strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KaybettirenIslemSayisi"], delimiter);         strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["NotrIslemSayisi"], delimiter);                strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KazandiranAlisSayisi"], delimiter);           strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KaybettirenAlisSayisi"], delimiter);          strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["NotrAlisSayisi"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KazandiranSatisSayisi"], delimiter);          strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KaybettirenSatisSayisi"], delimiter);         strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["NotrSatisSayisi"], delimiter);                strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["AlKomutSayisi"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["SatKomutSayisi"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["PasGecKomutSayisi"], delimiter);              strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KarAlKomutSayisi"], delimiter);               strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ZararKesKomutSayisi"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["FlatOlKomutSayisi"], delimiter);              strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["KomisyonIslemSayisi"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KomisyonVarlikAdedSayisi"], delimiter);       strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KomisyonCarpan"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KomisyonFiyat"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KomisyonFiyatYuzde"], delimiter);             strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KomisyonuDahilEt"], delimiter);               strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["KarZararFiyat"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KarZararFiyatYuzde"], delimiter);             strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KarZararPuan"], delimiter);                   strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ToplamKarFiyat"], delimiter);                 strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["ToplamZararFiyat"], delimiter);               strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["NetKarFiyat"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ToplamKarPuan"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ToplamZararPuan"], delimiter);                strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["NetKarPuan"], delimiter);                     strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxKarFiyat"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxZararFiyat"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxKarPuan"], delimiter);                     strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["MaxZararPuan"], delimiter);                   strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxZararFiyatIndex"], delimiter);             strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxKarFiyatIndex"], delimiter);               strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxZararPuanIndex"], delimiter);              strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["MaxKarPuanIndex"], delimiter);                strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KardaBarSayisi"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ZarardaBarSayisi"], delimiter);               strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KarliIslemOrani"], delimiter);                strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriMaxDD"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriMaxDDTarih"], delimiter);               strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriMaxDDSaat"], delimiter);                strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriMaxKayip"], delimiter);                 strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["ProfitFactor"], delimiter);                   strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ProfitFactorSistem"], delimiter);             strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["OrtAylikIslemSayisi"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["OrtHaftalikIslemSayisi"], delimiter);         strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["OrtGunlukIslemSayisi"], delimiter);           strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["OrtSaatlikIslemSayisi"], delimiter);          strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["KontratSayisi"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["VarlikAdedCarpani"], delimiter);              strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["VarlikAdedSayisi"], delimiter);               strList.Add( str );
*/
        for (int i = 0; i < strList.Count; i++)  fileWriter.Write(strList[i]);

        fileWriter.Write("\n");

        fileWriter.CloseFile();
    }

    public void OptimizasyonIstatistikleriniDosyayaYaz(dynamic Sistem, string FileName, int Index, int TotalCount)
    {
        CTxtFileWriter fileWriter = new CTxtFileWriter();

        fileWriter.OpenFile(FileName, true);

        if (!fileWriter.IsOpened())
            return;

        string delimiter = ";";

        string str = "";

        List<string> strList = new List<string>();

        strList.Clear();

        str = String.Format("{0} {1} ", Index, delimiter);                                              strList.Add( str );
        str = String.Format("{0} {1} ", TotalCount, delimiter);                                         strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriKzSistem"], delimiter);                 strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriKz"], delimiter);                       strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriKzNet"], delimiter);                    strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["ProfitFactor"], delimiter);                   strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ProfitFactorSistem"], delimiter);             strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["KarliIslemOrani"], delimiter);                strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["IslemSayisi"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KomisyonIslemSayisi"], delimiter);            strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["KomisyonFiyat"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KomisyonFiyatYuzde"], delimiter);             strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["IlkBakiyeFiyat"], delimiter);                 strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["BakiyeFiyat"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriFiyat"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriFiyatYuzde"], delimiter);               strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["BakiyeFiyatNet"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriFiyatNet"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriFiyatYuzdeNet"], delimiter);            strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["MinBakiyeFiyat"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxBakiyeFiyat"], delimiter);                 strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["MinBakiyeFiyatYuzde"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxBakiyeFiyatYuzde"], delimiter);            strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["MaxKarFiyat"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxZararFiyat"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxKarPuan"], delimiter);                     strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxZararPuan"], delimiter);                   strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["KazandiranIslemSayisi"], delimiter);          strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KaybettirenIslemSayisi"], delimiter);         strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["ToplamKarFiyat"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ToplamZararFiyat"], delimiter);               strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["NetKarFiyat"], delimiter);                    strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["ToplamKarPuan"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ToplamZararPuan"], delimiter);                strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["NetKarPuan"], delimiter);                     strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriMaxDD"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriMaxDDTarih"], delimiter);               strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriMaxDDSaat"], delimiter);                strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriMaxKayip"], delimiter);                 strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["OrtAylikIslemSayisi"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["OrtHaftalikIslemSayisi"], delimiter);         strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["OrtGunlukIslemSayisi"], delimiter);           strList.Add( str );

        str = String.Format("{0} {1} ", DateTime.Now.ToString("yyyy.MM.dd HH:mm:ss"), delimiter);       strList.Add( str );

/*
        str = String.Format("{0} {1} ", IstatistiklerNew["GrafikSembol"], delimiter);                   strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GrafikPeriyot"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["SistemName"], delimiter);                     strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriKzSistem"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriKzSistemYuzde"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriKzNetSistem"], delimiter);              strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriKzNetSistemYuzde"], delimiter);         strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["IlkBakiyeFiyat"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["BakiyeFiyat"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriFiyat"], delimiter);                    strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriFiyatYuzde"], delimiter);               strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["BakiyeFiyatNet"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriFiyatNet"], delimiter);                 strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriFiyatYuzdeNet"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriKz"], delimiter);                       strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriKzNet"], delimiter);                    strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["MinBakiyeFiyat"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxBakiyeFiyat"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MinBakiyeFiyatYuzde"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxBakiyeFiyatYuzde"], delimiter);            strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["MinBakiyeFiyatIndex"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxBakiyeFiyatIndex"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MinBakiyeFiyatNet"], delimiter);              strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxBakiyeFiyatNet"], delimiter);              strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["MinBakiyeFiyatNetIndex"], delimiter);         strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxBakiyeFiyatNetIndex"], delimiter);         strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MinBakiyeFiyatNetYuzde"], delimiter);         strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxBakiyeFiyatNetYuzde"], delimiter);         strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["IslemSayisi"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["AlisSayisi"], delimiter);                     strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["SatisSayisi"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["FlatSayisi"], delimiter);                     strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["PassSayisi"], delimiter);                     strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KarAlSayisi"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ZararKesSayisi"], delimiter);                 strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["KazandiranIslemSayisi"], delimiter);          strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KaybettirenIslemSayisi"], delimiter);         strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["NotrIslemSayisi"], delimiter);                strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KazandiranAlisSayisi"], delimiter);           strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KaybettirenAlisSayisi"], delimiter);          strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["NotrAlisSayisi"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KazandiranSatisSayisi"], delimiter);          strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KaybettirenSatisSayisi"], delimiter);         strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["NotrSatisSayisi"], delimiter);                strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["AlKomutSayisi"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["SatKomutSayisi"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["PasGecKomutSayisi"], delimiter);              strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KarAlKomutSayisi"], delimiter);               strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ZararKesKomutSayisi"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["FlatOlKomutSayisi"], delimiter);              strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["KomisyonIslemSayisi"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KomisyonVarlikAdedSayisi"], delimiter);       strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KomisyonCarpan"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KomisyonFiyat"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KomisyonFiyatYuzde"], delimiter);             strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KomisyonuDahilEt"], delimiter);               strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["KarZararFiyat"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KarZararFiyatYuzde"], delimiter);             strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KarZararPuan"], delimiter);                   strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ToplamKarFiyat"], delimiter);                 strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["ToplamZararFiyat"], delimiter);               strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["NetKarFiyat"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ToplamKarPuan"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ToplamZararPuan"], delimiter);                strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["NetKarPuan"], delimiter);                     strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxKarFiyat"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxZararFiyat"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxKarPuan"], delimiter);                     strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["MaxZararPuan"], delimiter);                   strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxZararFiyatIndex"], delimiter);             strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxKarFiyatIndex"], delimiter);               strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["MaxZararPuanIndex"], delimiter);              strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["MaxKarPuanIndex"], delimiter);                strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KardaBarSayisi"], delimiter);                 strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ZarardaBarSayisi"], delimiter);               strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["KarliIslemOrani"], delimiter);                strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriMaxDD"], delimiter);                    strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriMaxDDTarih"], delimiter);               strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriMaxDDSaat"], delimiter);                strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["GetiriMaxKayip"], delimiter);                 strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["ProfitFactor"], delimiter);                   strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["ProfitFactorSistem"], delimiter);             strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["OrtAylikIslemSayisi"], delimiter);            strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["OrtHaftalikIslemSayisi"], delimiter);         strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["OrtGunlukIslemSayisi"], delimiter);           strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["OrtSaatlikIslemSayisi"], delimiter);          strList.Add( str );

        str = String.Format("{0} {1} ", IstatistiklerNew["KontratSayisi"], delimiter);                  strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["VarlikAdedCarpani"], delimiter);              strList.Add( str );
        str = String.Format("{0} {1} ", IstatistiklerNew["VarlikAdedSayisi"], delimiter);               strList.Add( str );
*/
        for (int i = 0; i < strList.Count; i++)  fileWriter.Write(strList[i]);

        fileWriter.Write("\n");

        fileWriter.CloseFile();
    }
}
