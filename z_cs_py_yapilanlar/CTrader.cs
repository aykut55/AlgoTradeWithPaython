public class CTrader : CBase
    {
        public CSignals Signals = null;
        public CStatus Status = null;
        public CFlags Flags = null;
        public CLists Lists = null;
        public CKarZarar KarZarar = null;
        public CKomisyon Komisyon = null;
        public CBakiye Bakiye = null;
        public CVarlikManager VarlikManager = null;
        public CStatistics Statistics = null;
        public CKarAlZararKes KarAlZararKes = null;
        public CTimeFilter TimeFilter = null;
        bool BakiyeInitialized = false;
        int ExecutionStepNumber = 0;
        public string LastResetTime;
        public string LastExecutionTime;
        public string LastExecutionTimeStart;
        public string LastExecutionTimeStop;
        public string LastStatisticsCalculationTime;
        //public int LastExecutionId;
        private CTimeUtils TimeUtils;
        public UInt64 ExecutionTimeInMSec = 0;

        public const string DateTimeStringFormat = "dd.MM.yyyy HH:mm:ss";
        public const string DateStringFormat = "dd.MM.yyyy";
        public const string TimeStringFormat = "HH:mm:ss";
	
        public DateTime StartDateTime { get; set; }
        public DateTime StopDateTime { get; set; }
        public DateTime StartDate { get; set; }
        public DateTime StopDate { get; set; }
        public DateTime StartTime { get; set; }
        public DateTime StopTime { get; set; }     
        
        public string StartDateTimeStr { get; set; }
        public string StopDateTimeStr { get; set; } 
        public string StartDateStr { get; set; }
        public string StopDateStr { get; set; } 
        public string StartTimeStr { get; set; }
        public string StopTimeStr { get; set; } 
                
        public DateTime EnableDateTime { get; set; }
        public DateTime DisableDateTime { get; set; }
        public DateTime EnableDate { get; set; }
        public DateTime DisableDate { get; set; }
        public DateTime EnableTime { get; set; }
        public DateTime DisableTime { get; set; }
        
        public string EnableDateTimeStr { get; set; }
        public string DisableDateTimeStr { get; set; }
        public string EnableDateStr { get; set; }
        public string DisableDateStr { get; set; }
        public string EnableTimeStr { get; set; }
        public string DisableTimeStr { get; set; }

        ~CTrader()
        {

        }

        public CTrader(dynamic Sistem, int Id = 0)
        {
            this.Id = Id;

            Signals = new CSignals();

            Status = new CStatus();

            Flags = new CFlags();

            Lists = new CLists();

            KarZarar = new CKarZarar();

            Komisyon = new CKomisyon();

            Bakiye = new CBakiye();

            Statistics = new CStatistics();

            KarAlZararKes = new CKarAlZararKes();

            TimeFilter = new CTimeFilter();

            TimeUtils = new CTimeUtils();
        }

        public CTrader Initialize(dynamic Sistem, dynamic V, dynamic Open, dynamic High, dynamic Low, dynamic Close, dynamic Volume, dynamic Lot, dynamic VarlikManager)
        {
            this.VarlikManager = VarlikManager;

            SetData(Sistem, V, Open, High, Low, Close, Volume, Lot);

            Signals.Initialize(Sistem);

            Status.Initialize(Sistem);

            Flags.Initialize(Sistem);

            Lists.Initialize(Sistem);

            Lists.CreateLists(Sistem, this.BarCount);

            KarZarar.Initialize(Sistem, this);

            Komisyon.Initialize(Sistem, this);

            Bakiye.Initialize(Sistem, this);

            Statistics.Initialize(Sistem, this);

            KarAlZararKes.Initialize(Sistem, this);
            
            TimeFilter.Initialize(Sistem, this);

            TimeUtils.Initialize(Sistem, V, Open, High, Low, Close, Volume, Lot);

            Reset(Sistem);

            BakiyeInitialized = false;

            return this;
        }

        public CTrader Reset(dynamic Sistem)
        {
            Signals.Reset(Sistem);

            Status.Reset(Sistem);

            Flags.Reset(Sistem);

            Lists.Reset(Sistem);

            KarZarar.Reset(Sistem);

            Komisyon.Reset(Sistem);

            Bakiye.Reset(Sistem);

            Statistics.Reset(Sistem);

            KarAlZararKes.Reset(Sistem);

            TimeFilter.Reset(Sistem);

            TimeUtils.Reset(Sistem);

            ExecutionStepNumber = 0;

            LastResetTime = DateTime.Now.ToString("yyyy.MM.dd HH:mm:ss");

            LastExecutionTime = "";

            LastExecutionTimeStart = "";

            LastExecutionTimeStop = "";

            LastStatisticsCalculationTime = "";

            ExecutionTimeInMSec = 0;
            
            ResetDateTimes(Sistem); 

            return this;
        }

        public void Start(dynamic Sistem)
        {
            LastExecutionTimeStart = DateTime.Now.ToString("yyyy.MM.dd HH:mm:ss");

            LastExecutionTime = LastExecutionTimeStart;

            TimeUtils.StartTimer(Sistem);
        }

        public void Stop(dynamic Sistem)
        {
            LastExecutionTimeStop = DateTime.Now.ToString("yyyy.MM.dd HH:mm:ss");

            TimeUtils.StopTimer(Sistem);

            ExecutionTimeInMSec = TimeUtils.GetExecutionTimeInMSec(Sistem);
        }

        public int DonguBasiDegiskenleriResetle(dynamic Sistem, int BarIndex)
        {
            int result = 0;

            int i = BarIndex;

            Lists.BarIndexList[i] = i;

            Lists.YonList[i] = "";
            Lists.SeviyeList[i] = 0f;
            Lists.SinyalList[i] = 0f;

            Lists.KarZararPuanList[i] = 0f;
            Lists.KarZararFiyatList[i] = 0f;
            Lists.KarZararFiyatYuzdeList[i] = 0f;

            Status.KarZararFiyat = 0f;
            Status.KarZararPuan = 0f;
            Status.KarZararFiyatYuzde = 0f;

            Lists.KarAlList[i] = 0f;
            Lists.IzleyenStopList[i] = 0f;

            Lists.IslemSayisiList[i] = 0f;
            Lists.AlisSayisiList[i] = 0f;
            Lists.SatisSayisiList[i] = 0f;
            Lists.FlatSayisiList[i] = 0f;
            Lists.PassSayisiList[i] = 0f;

            Lists.KontratSayisiList[i] = 0f;
            Lists.VarlikAdedSayisiList[i] = 0f;
            Lists.KomisyonVarlikAdedSayisiList[i] = 0f;
            Lists.KomisyonIslemSayisiList[i] = 0f;
            Lists.KomisyonFiyatList[i] = 0f;
            Lists.KardaBarSayisiList[i] = 0f;
            Lists.ZarardaBarSayisiList[i] = 0f;

            Lists.BakiyeFiyatList[i] = Status.BakiyeFiyat;
            Lists.GetiriFiyatList[i] = Lists.BakiyeFiyatList[i] - Status.BakiyeFiyat;
            Lists.BakiyePuanList[i] = Status.BakiyePuan;
            Lists.GetiriPuanList[i] = Lists.BakiyePuanList[i] - Status.BakiyePuan;

            Lists.EmirKomutList[i] = 0f;
            Lists.EmirStatusList[i] = 0f;

            // First Step
            if (ExecutionStepNumber == 0)
            {

            }

            ExecutionStepNumber++;

            return result;
        }

        public int DonguBasiDegiskenleriGuncelle(dynamic Sistem, int BarIndex)
        {
            int result = 0;

            int i = BarIndex;

            Status.KomisyonVarlikAdedSayisi = VarlikManager.KomisyonVarlikAdedSayisi;

            Status.KomisyonCarpan = VarlikManager.KomisyonCarpan;

            Flags.KomisyonuDahilEt = VarlikManager.KomisyonuDahilEt;

            Status.KaymaMiktari = VarlikManager.KaymaMiktari;

            Flags.KaymayiDahilEt = VarlikManager.KaymayiDahilEt;

            Status.VarlikAdedSayisi = VarlikManager.VarlikAdedSayisi;

            Status.VarlikAdedCarpani = VarlikManager.VarlikAdedCarpani;

            Status.KontratSayisi = VarlikManager.KontratSayisi;

            Status.HisseSayisi = VarlikManager.HisseSayisi;

            Status.IlkBakiyeFiyat = VarlikManager.IlkBakiyeFiyat;

            Status.IlkBakiyePuan = VarlikManager.IlkBakiyePuan;

            Status.GetiriFiyatTipi = VarlikManager.GetiriFiyatTipi;

            if (BakiyeInitialized == false)
            {
                BakiyeInitialized = true;

                Status.BakiyeFiyat = Status.IlkBakiyeFiyat;

                Status.BakiyePuan = Status.IlkBakiyePuan;
            }

            return result;
        }

        public int AnlikKarZararHesapla(dynamic Sistem, int BarIndex, string Type = "C")
        {
            return KarZarar.AnlikKarZararHesapla(Sistem, BarIndex, Type);
        }

        public int EmirleriResetle(dynamic Sistem, int BarIndex)
        {
            int result = 0;

            int i = BarIndex;

            Signals.Al = false;
            Signals.Sat = false;
            Signals.FlatOl = false;
            Signals.PasGec = false;
            Signals.KarAl = false;
            Signals.ZararKes = false;

            return result;
        }

        public int EmirleriSetle(dynamic Sistem, int BarIndex, bool Al, bool Sat, bool FlatOl = false, bool PasGec = false, bool KarAl = false, bool ZararKes = false)
        {
            int result = 0;

            int i = BarIndex;

            Signals.Al = Al;
            Signals.Sat = Sat;
            Signals.FlatOl = FlatOl;
            Signals.PasGec = PasGec;
            Signals.KarAl = KarAl;
            Signals.ZararKes = ZararKes;

            return result;
        }

        public int EmirleriUygula(dynamic Sistem, int BarIndex)
        {
            int result = 0;

            int i = BarIndex;

            Flags.AGerceklesti = false;
            Flags.SGerceklesti = false;
            Flags.FGerceklesti = false;
            Flags.PGerceklesti = false;

            float AnlikKapanisFiyati = V[i].Close;
            float AnlikYuksekFiyati = V[i].High;
            float AnlikDusukFiyati = V[i].Low;

            // Sira Onemli
            // -------------------------------------------------------------------
            if (Signals.Al)       { Signals.Sinyal = "A"; Signals.EmirKomut = 1; Status.AlKomutSayisi++;       }    // Strateji
            if (Signals.Sat)      { Signals.Sinyal = "S"; Signals.EmirKomut = 2; Status.SatKomutSayisi++;      }
            if (Signals.PasGec)   { Signals.Sinyal = "P"; Signals.EmirKomut = 3; Status.PasGecKomutSayisi++;   }    // İşlem Filtreleri
            if (Signals.KarAl)    { Signals.Sinyal = "F"; Signals.EmirKomut = 4; Status.KarAlKomutSayisi++;    }    // Stop Loss ve Kar Al
            if (Signals.ZararKes) { Signals.Sinyal = "F"; Signals.EmirKomut = 5; Status.ZararKesKomutSayisi++; }
            if (Signals.FlatOl)   { Signals.Sinyal = "F"; Signals.EmirKomut = 6; Status.FlatOlKomutSayisi++;   }    // Poz Kapat (Flat Ol)

            // -------------------------------------------------------------------
            Status.KarAlSayisi = Status.KarAlKomutSayisi;
            Status.ZararKesSayisi = Status.ZararKesKomutSayisi;
            // -------------------------------------------------------------------

            // -------------------------------------------------------------------
            // Yön Kalıbı
            if (Signals.Sinyal == "A" && Signals.SonYon != "A")
            {
                Signals.PrevAFiyat = Signals.SonAFiyat;
                Signals.PrevABarNo = Signals.SonABarNo;

                Signals.PrevYon = Signals.SonYon;
                Signals.PrevFiyat = Signals.SonFiyat;
                Signals.PrevBarNo = Signals.SonBarNo;

                if (Signals.PrevYon == "F") { } 
                if (Signals.PrevYon == "S") { } 

                Lists.YonList[i] = "A";
                Signals.SonYon = Lists.YonList[i];
                Signals.SonFiyat = AnlikKapanisFiyati;

                if (Flags.KaymayiDahilEt)
                    Signals.SonFiyat = AnlikYuksekFiyati;

                Lists.SeviyeList[i] = Signals.SonFiyat;
                Signals.SonBarNo = i;

                Signals.SonAFiyat = Signals.SonFiyat;
                Signals.SonABarNo = Signals.SonBarNo;

                if (Signals.PrevYon == "F")
                {
                    Status.KomisyonIslemSayisi += 1;
                    Signals.EmirStatus = 1;
                }
                if (Signals.PrevYon == "S")
                {
                    var fark = Signals.SonFiyat - Signals.SonSFiyat;
                    if (fark< 0)
                    {
                        Status.KazandiranSatisSayisi++;
                    }
                    else if (fark > 0)
                    {
                        Status.KaybettirenSatisSayisi++;
                    }
                    else
                    {
                        Status.NotrSatisSayisi++;
                    }

                    Status.KomisyonIslemSayisi += 2;
                    Signals.EmirStatus = 2;
                }

                Flags.BakiyeGuncelle = true;
                Flags.KomisyonGuncelle = true;
                Flags.DonguSonuIstatistikGuncelle = true;
                Status.IslemSayisi++;
                Status.AlisSayisi++;
                Flags.AGerceklesti = true;
            }
            else if (Signals.Sinyal == "S" && Signals.SonYon != "S")
            {
                Signals.PrevSFiyat = Signals.SonSFiyat;
                Signals.PrevSBarNo = Signals.SonSBarNo;

                Signals.PrevYon = Signals.SonYon;
                Signals.PrevFiyat = Signals.SonFiyat;
                Signals.PrevBarNo = Signals.SonBarNo;

                if (Signals.PrevYon == "F") { } 
                if (Signals.PrevYon == "A") { } 

                Lists.YonList[i] = "S";
                Signals.SonYon = Lists.YonList[i];
                Signals.SonFiyat = AnlikKapanisFiyati;

                if (Flags.KaymayiDahilEt)
                    Signals.SonFiyat = AnlikDusukFiyati;

                Lists.SeviyeList[i] = Signals.SonFiyat;
                Signals.SonBarNo = i;

                Signals.SonSFiyat = Signals.SonFiyat;
                Signals.SonSBarNo = Signals.SonSBarNo;

                if (Signals.PrevYon == "F")
                {
                    Status.KomisyonIslemSayisi += 1;
                    Signals.EmirStatus = 3;
                }
                if (Signals.PrevYon == "A")
                {
                    var fark = Signals.SonFiyat - Signals.SonAFiyat;
                    if (fark > 0)
                    {
                        Status.KazandiranAlisSayisi++;
                    }
                    else if (fark< 0)
                    {
                        Status.KaybettirenAlisSayisi++;
                    }
                    else
                    {
                        Status.NotrAlisSayisi++;
                    }

                    Status.KomisyonIslemSayisi += 2;
                    Signals.EmirStatus = 4;
                }

                Flags.BakiyeGuncelle = true;
                Flags.KomisyonGuncelle = true;
                Flags.DonguSonuIstatistikGuncelle = true;
                Status.IslemSayisi++;
                Status.SatisSayisi++;
                Flags.SGerceklesti = true;
            }
            else if (Signals.Sinyal == "F" && Signals.SonYon != "F")
            {
                Signals.PrevFFiyat = Signals.SonFFiyat;
                Signals.PrevFBarNo = Signals.SonFBarNo;

                Signals.PrevYon = Signals.SonYon;
                Signals.PrevFiyat = Signals.SonFiyat;
                Signals.PrevBarNo = Signals.SonBarNo;

                if (Signals.PrevYon == "A") { } 
                if (Signals.PrevYon == "S") { } 

                Lists.YonList[i] = "F";
                Signals.SonYon = Lists.YonList[i];
                Signals.SonFiyat = AnlikKapanisFiyati;

                if (Flags.KaymayiDahilEt)
                {
                    if (Signals.PrevYon == "A")
                        Signals.SonFiyat = AnlikDusukFiyati;
                    if (Signals.PrevYon == "S")
                        Signals.SonFiyat = AnlikYuksekFiyati;
                }
                Lists.SeviyeList[i] = Signals.SonFiyat;
                Signals.SonBarNo = i;

                Signals.SonFFiyat = Signals.SonFiyat;
                Signals.SonFBarNo = Signals.SonFBarNo;

                if (Signals.PrevYon == "A")
                {
                    var fark = Signals.SonFiyat - Signals.SonAFiyat;
                    if (fark > 0)
                    {
                        Status.KazandiranAlisSayisi++;
                    }
                    else if (fark< 0)
                    {
                        Status.KaybettirenAlisSayisi++;
                    }
                    else
                    {
                        Status.NotrAlisSayisi++;
                    }

                    Status.KomisyonIslemSayisi += 1;
                    Signals.EmirStatus = 5;
                }
                if (Signals.PrevYon == "S")
                {
                    var fark = Signals.SonFiyat - Signals.SonSFiyat;
                    if (fark< 0)
                    {
                        Status.KazandiranSatisSayisi++;
                    }
                    else if (fark > 0)
                    {
                        Status.KaybettirenSatisSayisi++;
                    }
                    else
                    {
                        Status.NotrSatisSayisi++;
                    }

                    Status.KomisyonIslemSayisi += 1;
                    Signals.EmirStatus = 6;
                }

                Flags.BakiyeGuncelle = true;
                Flags.KomisyonGuncelle = true;
                Flags.DonguSonuIstatistikGuncelle = true;
                Status.IslemSayisi++;
                Status.FlatSayisi++;
                Flags.FGerceklesti = true;
            }
            else if (Signals.Sinyal == "P" || Signals.Sinyal == "")
            {
                Signals.PrevPFiyat = Signals.SonPFiyat;
                Signals.PrevPBarNo = Signals.SonPBarNo;

                Signals.SonPFiyat = AnlikKapanisFiyati;
                Signals.SonPBarNo = i;

                if (Signals.SonYon == "A")
                {
                    Signals.EmirStatus = 7;
                }

                if (Signals.SonYon == "S")
                {
                    Signals.EmirStatus = 8;
                }

                if (Signals.SonYon == "F")
                {
                    Signals.EmirStatus = 9;
                }

                Flags.BakiyeGuncelle = true;
                Flags.KomisyonGuncelle = true;
                Flags.DonguSonuIstatistikGuncelle = true;
                Status.PassSayisi++;
                Flags.PGerceklesti = true;
            }

            Status.KazandiranIslemSayisi = Status.KazandiranAlisSayisi + Status.KazandiranSatisSayisi;
            Status.KaybettirenIslemSayisi = Status.KaybettirenAlisSayisi + Status.KaybettirenSatisSayisi;
            Status.NotrIslemSayisi = Status.NotrAlisSayisi + Status.NotrSatisSayisi;

            if (Flags.AGerceklesti || Flags.SGerceklesti || Flags.FGerceklesti)
            {
                Status.KardaBarSayisi = 0;
                Status.ZarardaBarSayisi = 0;
            }

            if (Status.IslemSayisi > 0)
            {
                Flags.AnlikKarZararHesaplaEnabled = true;
                Flags.KarAlYuzdeHesaplaEnabled = true;
                Flags.IzleyenStopYuzdeHesaplaEnabled = true;
                Flags.ZararKesYuzdeHesaplaEnabled = true;
                Flags.KarAlSeviyeHesaplaEnabled = true;
                Flags.ZararKesSeviyeHesaplaEnabled = true;
            }

            Flags.AGerceklesti = false;
            Flags.SGerceklesti = false;
            Flags.FGerceklesti = false;
            Flags.PGerceklesti = false;

            Lists.EmirKomutList[i] = Signals.EmirKomut;
            Lists.EmirStatusList[i] = Signals.EmirStatus;

            return result;
        }

        public int SistemYonListesiniGuncelle(dynamic Sistem, int BarIndex)
        {
            int result = 0;

            int i = BarIndex;

            Sistem.Yon[i] = Lists.YonList[i];

            return result;
        }

        public int SistemSeviyeListesiniGuncelle(dynamic Sistem, int BarIndex)
        {
            int result = 0;

            int i = BarIndex;

            Sistem.Seviye[i] = Lists.SeviyeList[i];

            return result;
        }

        public int SinyalListesiniGuncelle(dynamic Sistem, int BarIndex)
        {
            int result = 0;

            int i = BarIndex;

            if (Signals.SonYon == "A")
            {
                Lists.SinyalList[i] = 1f;
            }
            else if (Signals.SonYon == "S")
            {
                Lists.SinyalList[i] = -1f;
            }
            else if (Signals.SonYon == "F")
            {
                Lists.SinyalList[i] = 0f;
            }

            return result;
        }

        public int IslemListesiniGuncelle(dynamic Sistem, int BarIndex)
        {
            int result = 0;

            int i = BarIndex;

            Lists.IslemSayisiList[i] = Status.IslemSayisi;
            Lists.AlisSayisiList[i] = Status.AlisSayisi;
            Lists.SatisSayisiList[i] = Status.SatisSayisi;
            Lists.FlatSayisiList[i] = Status.FlatSayisi;
            Lists.PassSayisiList[i] = Status.PassSayisi;
            Lists.VarlikAdedSayisiList[i] = Status.VarlikAdedSayisi;
            Lists.KontratSayisiList[i] = Status.KontratSayisi;
            Lists.KomisyonVarlikAdedSayisiList[i] = Status.KomisyonVarlikAdedSayisi;
            Lists.KomisyonIslemSayisiList[i] = Status.KomisyonIslemSayisi;
            Lists.KomisyonFiyatList[i] = Status.KomisyonFiyat;
            Lists.KardaBarSayisiList[i] = Status.KardaBarSayisi;
            Lists.ZarardaBarSayisiList[i] = Status.ZarardaBarSayisi;

            return result;
        }

        public int KomisyonListesiniGuncelle(dynamic Sistem, int BarIndex)
        {
            int result = 0;

            int i = BarIndex;

            Komisyon.Hesapla(Sistem, i);

            if (Flags.KomisyonGuncelle)
                Flags.KomisyonGuncelle = false;

            return result;
        }

        public int BakiyeListesiniGuncelle(dynamic Sistem, int BarIndex)
        {
            int result = 0;

            int i = BarIndex;

            Bakiye.Hesapla(Sistem, i);

            if (Flags.BakiyeGuncelle)
                Flags.BakiyeGuncelle = false;

            return result;
        }

        public int DonguSonuDegiskenleriSetle(dynamic Sistem, int BarIndex)
        {
            int result = 0;

            int i = BarIndex;

            return result;
        }

        public int IstatistikleriHesapla(dynamic Sistem)
        {
            int result = 0;

            Statistics.Hesapla(Sistem);

            return result;
        }

        public int IstatistikleriEkranaYaz(dynamic Sistem, int PanelNo = 1)
        {
            int result = 0;

            Statistics.IstatistikleriEkranaYaz(Sistem, PanelNo);

            return result;
        }

        public int GetiriIstatistikleriEkranaYaz(dynamic Sistem, int PanelNo = 2)
        {
            int result = 0;

            Statistics.GetiriIstatistikleriEkranaYaz(Sistem, PanelNo);

            return result;
        }

        public void IstatistikleriDosyayaYaz(dynamic Sistem, string FileName)
        {
            Statistics.IstatistikleriDosyayaYaz(Sistem, FileName);
        }

        public void OptimizasyonIstatistiklerininBasliklariniDosyayaYaz(dynamic Sistem, string FileName)
        {
            Statistics.OptimizasyonIstatistiklerininBasliklariniDosyayaYaz(Sistem, FileName);
        }

        public void OptimizasyonIstatistikleriniDosyayaYaz(dynamic Sistem, string FileName, int Index, int TotalCount)
        {
            Statistics.OptimizasyonIstatistikleriniDosyayaYaz(Sistem, FileName, Index, TotalCount);
        }

        public bool IsSonYonA(dynamic Sistem)
        {
            return Signals.SonYon == "A";
        }

        public bool IsSonYonS(dynamic Sistem)
        {
            return Signals.SonYon == "S";
        }

        public bool IsSonYonF(dynamic Sistem)
        {
            return Signals.SonYon == "F";
        }

        public void IdealGetiriHesapla(dynamic Sistem, double KaymaMiktari = 0.00, string BaslangicTarihi = "01/01/1900", string BitisTarihi = "01/01/2100")
        {
            if (Flags.IdealGetiriHesapla == false) return;

            Flags.IdealGetiriHesaplandi = true;

            // -------------------------------------------------------------------
            // İdeal Getiri Hesapla
            //double KaymaMiktari = 0.0 + 0 * 0.35;
            Sistem.GetiriHesapla(BaslangicTarihi, KaymaMiktari); // Sistem.GetiriKZ, Sistem.GetiriMiktar, Sistem.GetiriPozisyon
            Sistem.GetiriMaxDDHesapla(BaslangicTarihi, BitisTarihi);

            for (int i = 0; i < Sistem.GetiriKZ.Count; i++)
            {
                Lists.GetiriKzSistem[i] = Sistem.GetiriKZ[i];
                Lists.GetiriKzNetSistem[i] = 0f;
            }

            Status.GetiriKzSistem = Lists.GetiriKzSistem[Lists.GetiriKzSistem.Count - 1];
            Status.GetiriKzNetSistem = 0f;
        }

        public bool GunSonuPozKapat(dynamic Sistem, int BarIndex, bool GunSonuPozKapatEnabled = true)
        {
            int i = BarIndex;

            bool GunSonuPozKapatildi = false;

            if (GunSonuPozKapatEnabled)
            {
                if (i < BarCount - 1 && V[i].Date.Day != V[i+1].Date.Day)
                {
                    Signals.FlatOl = true;
                    GunSonuPozKapatildi = true;
                }
            }

            return GunSonuPozKapatildi;
        }

        public bool GunSonuPozKapat2(dynamic Sistem, int BarIndex, bool GunSonuPozKapatEnabled = true, int Hour = 18, int Minute = 0)
        {
            int i = BarIndex;

            bool GunSonuPozKapatildi = false;

            if (GunSonuPozKapatEnabled)
            {
                if (V[i].Date.Hour == 18 && V[i].Date.Minute >= 00)
                {
                    Signals.FlatOl = true;
                    GunSonuPozKapatildi = true;
                }
            }

            return GunSonuPozKapatildi;
        }

        public int SinyalleriEkranaCiz(dynamic Sistem, int k, bool SistemGetiriKZDahilEt = false)
        {
            CTrader myTrader = this;

            var BarIndexList                 = myTrader.Lists.BarIndexList;
            var SeviyeList                   = myTrader.Lists.SeviyeList;
            var SinyalList                   = myTrader.Lists.SinyalList;

            var KarZararPuanList             = myTrader.Lists.KarZararPuanList;
            var KarZararFiyatList            = myTrader.Lists.KarZararFiyatList;
            var KarZararFiyatYuzdeList       = myTrader.Lists.KarZararFiyatYuzdeList;

            var IslemSayisiList              = myTrader.Lists.IslemSayisiList;
            var AlisSayisiList               = myTrader.Lists.AlisSayisiList;
            var SatisSayisiList              = myTrader.Lists.SatisSayisiList;
            var FlatSayisiList               = myTrader.Lists.FlatSayisiList;
            var PassSayisiList               = myTrader.Lists.PassSayisiList;

            var KontratSayisiList            = myTrader.Lists.KontratSayisiList;
            var VarlikAdedSayisiList         = myTrader.Lists.VarlikAdedSayisiList;
            var KomisyonVarlikAdedSayisiList = myTrader.Lists.KomisyonVarlikAdedSayisiList;
            var KomisyonIslemSayisiList      = myTrader.Lists.KomisyonIslemSayisiList;
            var KomisyonFiyatList            = myTrader.Lists.KomisyonFiyatList;
            var KardaBarSayisiList           = myTrader.Lists.KardaBarSayisiList;
            var ZarardaBarSayisiList         = myTrader.Lists.ZarardaBarSayisiList;

            var BakiyePuanList               = myTrader.Lists.BakiyePuanList;
            var BakiyeFiyatList              = myTrader.Lists.BakiyeFiyatList;
            var GetiriPuanList               = myTrader.Lists.GetiriPuanList;
            var GetiriFiyatList              = myTrader.Lists.GetiriFiyatList;
            var GetiriPuanYuzdeList          = myTrader.Lists.GetiriPuanYuzdeList;
            var GetiriFiyatYuzdeList         = myTrader.Lists.GetiriFiyatYuzdeList;

            var BakiyePuanNetList            = myTrader.Lists.BakiyePuanNetList;
            var BakiyeFiyatNetList           = myTrader.Lists.BakiyeFiyatNetList;
            var GetiriPuanNetList            = myTrader.Lists.GetiriPuanNetList;
            var GetiriFiyatNetList           = myTrader.Lists.GetiriFiyatNetList;
            var GetiriPuanYuzdeNetList       = myTrader.Lists.GetiriPuanYuzdeNetList;
            var GetiriFiyatYuzdeNetList      = myTrader.Lists.GetiriFiyatYuzdeNetList;

            var GetiriKz                     = myTrader.Lists.GetiriKz;
            var GetiriKzNet                  = myTrader.Lists.GetiriKzNet;

            var EmirKomutList                = myTrader.Lists.EmirKomutList;
            var EmirStatusList               = myTrader.Lists.EmirStatusList;

            var SistemGetiriKZ               = SistemGetiriKZDahilEt ? Sistem.GetiriKZ : Sistem.Liste(0);

            Sistem.Cizgiler[k].Deger = Sistem.GetiriKZ;
            Sistem.Cizgiler[k++].Aciklama = "Sistem.GetiriKZ ";

            Sistem.Cizgiler[k].Deger = KarZararFiyatList;
            Sistem.Cizgiler[k++].Aciklama = "KarZararFiyat";

            Sistem.Cizgiler[k].Deger = KarZararFiyatYuzdeList;
            Sistem.Cizgiler[k++].Aciklama = "KarZararFiyatYuzdeList";

            Sistem.Cizgiler[k].Deger = GetiriKz;
            Sistem.Cizgiler[k++].Aciklama = "GetiriKz";

            Sistem.Cizgiler[k].Deger = GetiriKzNet;
            Sistem.Cizgiler[k++].Aciklama = "GetiriKzNet";

            Sistem.Cizgiler[k].Deger = GetiriFiyatList;
            Sistem.Cizgiler[k++].Aciklama = "GetiriFiyat";

            Sistem.Cizgiler[k].Deger = BakiyeFiyatList;
            Sistem.Cizgiler[k++].Aciklama = "BakiyeFiyat";

            Sistem.Cizgiler[k].Deger = GetiriFiyatNetList;
            Sistem.Cizgiler[k++].Aciklama = "GetiriFiyatNet";

            Sistem.Cizgiler[k].Deger = BakiyeFiyatNetList;
            Sistem.Cizgiler[k++].Aciklama = "BakiyeFiyatNet";

            Sistem.Cizgiler[k].Deger = KomisyonFiyatList;
            Sistem.Cizgiler[k++].Aciklama = "KomisyonFiyatList";

            Sistem.Cizgiler[k].Deger = KomisyonIslemSayisiList;
            Sistem.Cizgiler[k++].Aciklama = "KomisyonIslemSayisiList";

            Sistem.Cizgiler[k].Deger = IslemSayisiList;
            Sistem.Cizgiler[k++].Aciklama = "IslemSayisiList ";

            Sistem.Cizgiler[k].Deger = EmirKomutList;
            Sistem.Cizgiler[k++].Aciklama = "EmirKomutList ";

            Sistem.Cizgiler[k].Deger = EmirStatusList;
            Sistem.Cizgiler[k++].Aciklama = "EmirStatusList ";

            Sistem.Cizgiler[k].Deger = BarIndexList;
            Sistem.Cizgiler[k++].Aciklama = "BarIndexList";

            return k;

        }

        public string GetBarValuesDescription(dynamic Sistem)
        {
            string delimiter = ";";

            string LogMessage = "";

            int mode = 1;

            if (mode == 0)
            {
                LogMessage = String.Format("#  {0},{1},{2},{3},{4},{5},{6},{7},{8},{9}", "Column Names", "No", "Date", "Time", "Open", "High", "Low", "Close", "Vol", "Size(Lot)");
            }
            else
            {
                LogMessage = String.Format("#  {0,-12} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18} {19}",  "Sutunlar", delimiter, "No", delimiter, "Date", delimiter, "Time", delimiter, "Open", delimiter, "High", delimiter, "Low", delimiter, "Close", delimiter, "Vol", delimiter, "Size(Lot)", delimiter );
            }

            // ------   -------------------     -------         -------         -------         -------         -----------    ------

            return LogMessage;
        }

        public string GetBarValuesAsString(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;

            string delimiter = ";";

            string LogMessage = "";

            int mode = 1;

            if (mode == 0)
            {
                LogMessage = String.Format("{0,-5} 	 {1,-20} 	 {2, 5} 	 {3, 5} 	 {4, 5} 	 {5, 5} 	 {6, 10} 	 {7, 5} 	 ", i, V[i].Date.ToString("yyyy.MM.dd HH:mm:ss"), V[i].Open.ToString("0.00"), V[i].High.ToString("0.00"), V[i].Low.ToString("0.00"), V[i].Close.ToString("0.00"), V[i].Vol.ToString("0"), V[i].Size.ToString("0"));
            }
            else
            {
                LogMessage = String.Format("{0} {1,-5} {2} {3, 10} {4} {5, 10} {6} {7, 10} {8} {9, 10} {10} {11, 10} {12} {13, 10} {14} {15, 10} {16} {17, 10} {18}",
                    delimiter, i,
                    delimiter, V[i].Date.ToString("yyyy.MM.dd"),
                    delimiter, V[i].Date.ToString("HH:mm:ss"),
                    delimiter, V[i].Open.ToString("0.00"),
                    delimiter, V[i].High.ToString("0.00"),
                    delimiter, V[i].Low.ToString("0.00"),
                    delimiter, V[i].Close.ToString("0.00") ,
                    delimiter, V[i].Vol.ToString("0"),
                    delimiter, V[i].Size.ToString("0"),
                    delimiter
                    );

            }

            return LogMessage;
        }

        public void WriteDataToFile_OHLC(dynamic Sistem, string FileName)
        {
            string delimiter = ";";

            CFileUtils myFileUtils = new CFileUtils();

            CTimeUtils myTimeUtils = new CTimeUtils();

            myTimeUtils.Initialize(Sistem, V, Open, High, Low, Close, Volume, Lot);

            myTimeUtils.GecenZamanBilgileriniAl(Sistem);

            string aciklama1 = "...";
            string aciklama2 = "...";
            string aciklama3 = "...";
            string aciklama4 = "...";
            string aciklama5 = "...";
            string aciklama6 = "...";

            string logFileFullName = FileName.Trim();

            string logMessage = "";

            myFileUtils.Reset(Sistem).EnableLogging(Sistem).OpenLogFile(Sistem, logFileFullName, false, false);
            //myFileUtils.Reset(Sistem).DisableLogging(Sistem).OpenLogFile(Sistem, logFileName, false, false);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Aciklama (1)", aciklama1.Trim() );
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Aciklama (2)", aciklama2.Trim() );
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Aciklama (3)", aciklama3.Trim() );
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Aciklama (4)", aciklama4.Trim() );
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Aciklama (5)", aciklama5.Trim() );
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Aciklama (6)", aciklama6.Trim() );
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Log Zamani", DateTime.Now.ToString("yyyy.MM.dd HH:mm:ss"));
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Sembol", Sistem.Sembol);
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Periyod", Sistem.Periyot);
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Bar Sayisi", BarCount);
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -10}   ; {1} 	", "Ilk Bar Zamani", V[0].Date.ToString("yyyy.MM.dd HH:mm:ss"));
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -10}   ; {1} 	", "Son Bar Zamani", V[V.Count-1].Date.ToString("yyyy.MM.dd HH:mm:ss"));
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -10}   ; {1} 	", "Gecen Sure (A)", myTimeUtils.GecenSure(Sistem, "A").ToString("0.0"));
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -10}   ; {1} 	", "Gecen Sure (G)", myTimeUtils.GecenSure(Sistem, "G").ToString("0"));
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = GetBarValuesDescription(Sistem);
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            for (int i = 0; i < BarCount; i++)
            {
                logMessage = String.Format("{0}", GetBarValuesAsString(Sistem, i) );
                myFileUtils.WriteToLogFile(Sistem, logMessage);
            }

            myFileUtils.CloseLogFile(Sistem);
        }

        public void WriteDataToFile_Custom(dynamic Sistem, string FileName, List< List<float> > DataLists, List< string > CaptionList)
        {
            string delimiter = ";";

            CFileUtils myFileUtils = new CFileUtils();

            CTimeUtils myTimeUtils = new CTimeUtils();

            myTimeUtils.Initialize(Sistem, V, Open, High, Low, Close, Volume, Lot);

            myTimeUtils.GecenZamanBilgileriniAl(Sistem);

            string aciklama1 = "...";
            string aciklama2 = "...";
            string aciklama3 = "...";
            string aciklama4 = "...";
            string aciklama5 = "...";
            string aciklama6 = "...";

            string logFileFullName = FileName.Trim();

            string logMessage = "";

            myFileUtils.Reset(Sistem).EnableLogging(Sistem).OpenLogFile(Sistem, logFileFullName, false, false);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Aciklama (1)", aciklama1.Trim() );
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Aciklama (2)", aciklama2.Trim() );
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Aciklama (3)", aciklama3.Trim() );
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Aciklama (4)", aciklama4.Trim() );
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Aciklama (5)", aciklama5.Trim() );
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Aciklama (6)", aciklama6.Trim() );
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Log Zamani", DateTime.Now.ToString("yyyy.MM.dd HH:mm:ss"));
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Sembol", Sistem.Sembol);
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Periyod", Sistem.Periyot);
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -14}   ; {1} 	", "Bar Sayisi", BarCount);
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -10}   ; {1} 	", "Ilk Bar Zamani", V[0].Date.ToString("yyyy.MM.dd HH:mm:ss"));
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -10}   ; {1} 	", "Son Bar Zamani", V[V.Count-1].Date.ToString("yyyy.MM.dd HH:mm:ss"));
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -10}   ; {1} 	", "Gecen Sure (A)", myTimeUtils.GecenSure(Sistem, "A").ToString("0.0"));
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0, -10}   ; {1} 	", "Gecen Sure (G)", myTimeUtils.GecenSure(Sistem, "G").ToString("0"));
            myFileUtils.WriteToLogFile(Sistem, logMessage);

            logMessage = String.Format("#  {0,-12} {1} {2} ",  "Sutunlar", delimiter, "No");
            for (int j = 0; j < CaptionList.Count; j++)
                logMessage = logMessage + delimiter + CaptionList[j];

            myFileUtils.WriteToLogFile(Sistem, logMessage);

            for (int i = 0; i < BarCount; i++)
            {
                logMessage = String.Format("{0} {1,-5} ", delimiter, i);

                for (int j = 0; j < DataLists.Count; j++)
                {
                    var column = DataLists[j];
                    //var caption = Captions[j];
                    logMessage = logMessage + delimiter + column[i];
                }

                myFileUtils.WriteToLogFile(Sistem, logMessage);
            }

            myFileUtils.CloseLogFile(Sistem);
        }

        public CTrader ResetDateTimes(dynamic Sistem)
        {
	        bool useLastBarDateTime = true;	
	        
			StartDateTime    = V[0].Date;
	        StopDateTime     = useLastBarDateTime ? V[Sistem.BarSayisi - 1].Date : DateTime.Now; 
			StartDate        = StartDateTime;
        	StopDate         = StopDateTime;
            StartTime        = StartDateTime;             				 	               
	        StopTime         = StopDateTime;
    	    StartDateTimeStr = StartDateTime.ToString(DateTimeStringFormat);
        	StopDateTimeStr  = StopDateTime.ToString(DateTimeStringFormat);
            StartDateStr     = StartDate.ToString(DateStringFormat);	        
	        StopDateStr      = StopDate.ToString(DateStringFormat); 
	        StartTimeStr     = StartTime.ToString(TimeStringFormat);
    	    StopTimeStr      = StopTime.ToString(TimeStringFormat);
	          
            return this;
        }

        public CTrader SetDateTimes(dynamic Sistem, string StartDate, string StartTime, string StopDate, string StopTime)
        {
	        string date1       = StartDate.Trim();
	        string time1       = StartTime.Trim();
	        string date2       = StopDate.Trim();
	        string time2       = StopTime.Trim();
	        string dateTime1   = date1 + " " + time1;
	        string dateTime2   = date2 + " " + time2;
	        string suffixDate  = "09:30:00";
	        string prefixTime  = "01.01.1900";

	        this.StartDateTime = TimeUtils.GetDateTime(date1, time1);
	        this.StopDateTime  = TimeUtils.GetDateTime(date2, time2);        
	        this.StartDate     = TimeUtils.GetDateTime(date1 + " " + suffixDate);      
	        this.StopDate      = TimeUtils.GetDateTime(date2 + " " + suffixDate);
            this.StartTime     = TimeUtils.GetDateTime(prefixTime + " " + time1);
            this.StopTime      = TimeUtils.GetDateTime(prefixTime + " " + time2);

    	    this.StartDateTimeStr = this.StartDateTime.ToString(DateTimeStringFormat);
        	this.StopDateTimeStr  = this.StopDateTime.ToString(DateTimeStringFormat);
            this.StartDateStr     = this.StartDate.ToString(DateStringFormat);
	        this.StopDateStr      = this.StopDate.ToString(DateStringFormat); 
	        this.StartTimeStr     = this.StartTime.ToString(TimeStringFormat);
    	    this.StopTimeStr      = this.StopTime.ToString(TimeStringFormat);
            	        
            return this;                
        }
                
        public CTrader SetDateTimes(dynamic Sistem, string StartDateTime, string StopDateTime)
        {
        	// string[] ssize = myStr.Split(null); //Or myStr.Split()
	        string dateTime1   = StartDateTime.Trim();
	        string dateTime2   = StopDateTime.Trim();	        
	        string date1       = dateTime1.Substring(0, 10);
	        string time1       = dateTime1.Substring(11);
	        string date2       = dateTime2.Substring(0, 10);
	        string time2       = dateTime2.Substring(11);
	        string suffixDate  = "09:30:00";
	        string prefixTime  = "01.01.1900";
            
	        this.StartDateTime = TimeUtils.GetDateTime(dateTime1);
	        this.StopDateTime  = TimeUtils.GetDateTime(dateTime2);
	        this.StartDate     = TimeUtils.GetDateTime(date1 + " " + suffixDate);
	        this.StopDate      = TimeUtils.GetDateTime(date2 + " " + suffixDate);
            this.StartTime     = TimeUtils.GetDateTime(prefixTime + " " + time1);
            this.StopTime      = TimeUtils.GetDateTime(prefixTime + " " + time2);

    	    this.StartDateTimeStr = this.StartDateTime.ToString(DateTimeStringFormat);
        	this.StopDateTimeStr  = this.StopDateTime.ToString(DateTimeStringFormat);
            this.StartDateStr     = this.StartDate.ToString(DateStringFormat);
	        this.StopDateStr      = this.StopDate.ToString(DateStringFormat); 
	        this.StartTimeStr     = this.StartTime.ToString(TimeStringFormat);
    	    this.StopTimeStr      = this.StopTime.ToString(TimeStringFormat);
	        
            return this;     
        }

        public CTrader SetDateTime(dynamic Sistem, string StartDate, string StartTime)
        {
	        string StartDateTime = StartDate + " " + StartTime;        
        	return SetDateTime(Sistem, StartDateTime);
        }
                
        public CTrader SetDateTime(dynamic Sistem, string StartDateTime)
        {
        	return SetDateTimes(Sistem, StartDateTime, DateTime.Now.ToString());
        }

        private CTrader SetStartDateTime_Kullanma(dynamic Sistem, string StartDateTime)
        {
            this.StartDateTime = DateTime.ParseExact(StartDateTime.Trim(), DateTimeStringFormat.Trim(), System.Globalization.CultureInfo.InvariantCulture);
            return this;
        } 
        
        private CTrader SetStopDateTime_Kullanma(dynamic Sistem, string StopDateTime)
        {
            this.StopDateTime = DateTime.ParseExact(StopDateTime.Trim(), DateTimeStringFormat.Trim(), System.Globalization.CultureInfo.InvariantCulture);
            return this;            
        }        
        
        private CTrader SetStartDate_Kullanma(dynamic Sistem, string StartDate)
        {
            this.StartDate = DateTime.ParseExact(StartDate.Trim(), DateStringFormat.Trim(), System.Globalization.CultureInfo.InvariantCulture);
            return this;
        } 
        
        private CTrader SetStopDate_Kullanma(dynamic Sistem, string StopDate)
        {
            this.StopDate = DateTime.ParseExact(StopDate.Trim(), DateStringFormat.Trim(), System.Globalization.CultureInfo.InvariantCulture);
            return this;            
        }   
        
        private CTrader SetStartTime_Kullanma(dynamic Sistem, string StartTime)
        {
            this.StartTime = DateTime.ParseExact(StartTime.Trim(), TimeStringFormat.Trim(), System.Globalization.CultureInfo.InvariantCulture);
            return this;
        } 
        
        private CTrader SetStopTime_Kullanma(dynamic Sistem, string StopTime)
        {
            this.StopTime = DateTime.ParseExact(StopTime.Trim(), TimeStringFormat.Trim(), System.Globalization.CultureInfo.InvariantCulture);
            return this;            
        }

        public bool IsTradingEnabledByDateTime_Kontrol_Edilmesi_Gerek_Kullanma(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;
            DateTime BarDateTime = V[i].Date;
            bool result = ( TimeUtils.CompareDates(BarDateTime, this.StartDateTime) >= 0 ) && ( TimeUtils.CompareDates(BarDateTime, this.StopDateTime) < 0 );            
            return result;
        }
        
        public bool IsTradingEnabledByDate_Kontrol_Edilmesi_Gerek_Kullanma(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;
            DateTime BarDateTime = V[i].Date;
            bool result = ( TimeUtils.CompareDates(BarDateTime, this.StartDate) >= 0 ) && ( TimeUtils.CompareDates(BarDateTime, this.StopDate) < 0 );
            return result;
        }

        public bool IsTradingEnabledByTime_Kontrol_Edilmesi_Gerek_Kullanma(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;
            DateTime BarDateTime = V[i].Date;
            bool result = ( TimeUtils.CompareTimes(BarDateTime, this.StartTime) >= 0 ) && ( TimeUtils.CompareTimes(BarDateTime, this.StopTime) < 0 );
            return result;
        }
        
        public int CheckBarTimeValidty_Kontrol_Edilmesi_Gerek_Kullanma(dynamic Sistem, int BarIndex, string startTime, string stopTime)
        {
        	int result = 0;
        	
            int i = BarIndex;
            DateTime BarDateTime = V[i].Date;
            
            if (TimeUtils.CheckBarTimeWith(Sistem, i, startTime) < 0)
            	result = -1;
            else if (TimeUtils.CheckBarTimeWith(Sistem, i, stopTime) >= 0)
            	result = 1;
            else if (TimeUtils.CheckBarTimeWith(Sistem, i, startTime) >= 0 && TimeUtils.CheckBarTimeWith(Sistem, i, stopTime) < 0)
            	result = 0;
            	
            return result;
        } 
        
        public int CheckBarTimeValidty_Kontrol_Edilmesi_Gerek_Kullanma(dynamic Sistem, int BarIndex)
        {
            return CheckBarTimeValidty_Kontrol_Edilmesi_Gerek_Kullanma(Sistem, BarIndex, this.StartTimeStr , this.StopTimeStr);
        }

        public int IslemZamanFiltresiUygula(dynamic Sistem, int BarIndex, int FilterMode, ref bool IsTradeEnabled, ref bool IsPozKapatEnabled, ref int CheckResult)
        {
            int i = BarIndex;
            DateTime BarDateTime = V[i].Date;
	                    
            string startDateTime = this.StartDateTimeStr;
            string stopDateTime  = this.StopDateTimeStr;            
            string startDate     = this.StartDateStr;
            string stopDate      = this.StopDateStr;
            string startTime     = this.StartTimeStr;
            string stopTime      = this.StopTimeStr;
            var nowDateTime      = DateTime.Now.ToString("dd.MM.yyyy HH:mm:ss");
            var nowDate          = DateTime.Now.ToString("dd.MM.yyyy");
            var nowTime          = DateTime.Now.ToString("HH:mm:ss");
                    
            bool useTimeFiltering = Signals.TimeFilteringEnabled ? true : false;
            if (useTimeFiltering)
            {
                if (i == Sistem.BarSayisi - 1)
                {
                    var s = "";
                    s += "  " + startDateTime.ToString() + Environment.NewLine;
                    s += "  " + stopDateTime.ToString() + Environment.NewLine;
                    s += "  " + startDate.ToString() + Environment.NewLine;
                    s += "  " + stopDate.ToString() + Environment.NewLine;
                    s += "  " + startTime.ToString() + Environment.NewLine;
                    s += "  " + stopTime.ToString() + Environment.NewLine;
                    s += "  " + nowDateTime.ToString() + Environment.NewLine;
                    s += "  " + nowDate.ToString() + Environment.NewLine;
                    s += "  " + nowTime.ToString() + Environment.NewLine;
                    s += "  " + ("FilterMode = " + FilterMode.ToString()).ToString() + Environment.NewLine;
                    s += "  " + "CTrader::IslemZamanFiltresiUygula".ToString() + Environment.NewLine;

                }

                if (FilterMode == 0)
                {
                    IsTradeEnabled = true;
                    CheckResult = 0;
                }
                else if (FilterMode == 1)
                {
                    if (TimeUtils.CheckBarTimeWith(Sistem, i, startTime) >= 0 && TimeUtils.CheckBarTimeWith(Sistem, i, stopTime) < 0)
                    {
                        IsTradeEnabled = true;
                        CheckResult = 0;
                    }
                    else if (TimeUtils.CheckBarTimeWith(Sistem, i, startTime) < 0)
                    {
                        if (!IsSonYonF(Sistem))
                            IsPozKapatEnabled = true;
                        CheckResult = -1;
                    }
                    else if (TimeUtils.CheckBarTimeWith(Sistem, i, stopTime) >= 0)
                    {
                        if (!IsSonYonF(Sistem))
                            IsPozKapatEnabled = true;
                        CheckResult = 1;
                    }
                }
                else if (FilterMode == 2)
                {
                    if (TimeUtils.CheckBarDateWith(Sistem, i, startDate) >= 0 && TimeUtils.CheckBarDateWith(Sistem, i, stopDate) < 0)
                    {
                        IsTradeEnabled = true;
                        CheckResult = 0;
                    }
                    else if (TimeUtils.CheckBarDateWith(Sistem, i, startDate) < 0)
                    {
                        if (!IsSonYonF(Sistem))
                            IsPozKapatEnabled = true;
                        CheckResult = -1;
                    }
                    else if (TimeUtils.CheckBarDateWith(Sistem, i, stopDate) >= 0)
                    {
                        if (!IsSonYonF(Sistem))
                            IsPozKapatEnabled = true;
                        CheckResult = 1;
                    }
                }
                else if (FilterMode == 3)
                {
                    if (TimeUtils.CheckBarDateTimeWith(Sistem, i, startDateTime) >= 0 && TimeUtils.CheckBarDateTimeWith(Sistem, i, stopDateTime) < 0)
                    {
                        IsTradeEnabled = true;
                        CheckResult = 0;
                    }
                    else if (TimeUtils.CheckBarDateTimeWith(Sistem, i, startDateTime) < 0)
                    {
                        if (!IsSonYonF(Sistem))
                            IsPozKapatEnabled = true;
                        CheckResult = -1;
                    }
                    else if (TimeUtils.CheckBarDateTimeWith(Sistem, i, stopDateTime) >= 0)
                    {
                        if (!IsSonYonF(Sistem))
                            IsPozKapatEnabled = true;
                        CheckResult = 1;
                    }
                }
                else if (FilterMode == 4)
                {
                    if (TimeUtils.CheckBarTimeWith(Sistem, i, startTime) >= 0)
                    {
                        IsTradeEnabled = true;
                        CheckResult = 0;
                    }
                    else if (TimeUtils.CheckBarTimeWith(Sistem, i, startTime) < 0)
                    {
                        if (!IsSonYonF(Sistem))
                            IsPozKapatEnabled = true;
                        CheckResult = -1;
                    }
                }
                else if (FilterMode == 5)
                {
                    if (TimeUtils.CheckBarDateWith(Sistem, i, startDate) >= 0)
                    {
                        IsTradeEnabled = true;
                        CheckResult = 0;
                    }
                    else if (TimeUtils.CheckBarDateWith(Sistem, i, startDate) < 0)
                    {
                        if (!IsSonYonF(Sistem))
                            IsPozKapatEnabled = true;
                        CheckResult = -1;
                    }
                }
                else if (FilterMode == 6)
                {
                    if (TimeUtils.CheckBarDateTimeWith(Sistem, i, startDateTime) >= 0)
                    {
                        IsTradeEnabled = true;
                        CheckResult = 0;
                    }
                    else if (TimeUtils.CheckBarDateTimeWith(Sistem, i, startDateTime) < 0)
                    {
                        if (!IsSonYonF(Sistem))
                            IsPozKapatEnabled = true;
                        CheckResult = -1;
                    }
                }
            }

            return 0;
        }

    }
