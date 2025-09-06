public class CBarUtils : CBase
    {
        private CTimeUtils TimeUtils;

        public List<float> BarOpen { get; set; }
        public List<float> BarHigh { get; set; }
        public List<float> BarLow { get; set; }
        public List<float> BarClose { get; set; }
        public List<float> BarVolume { get; set; }
        public List<float> BarLot { get; set; }

        public string BarMAYontem { get; set; }
        public int BarPeriyot { get; set; }
        public int BarPanelNo { get; set; }
        public int BarTip { get; set; }

        public DateTime SecilenBarTarihi { get; set; }
        public int SecilenBarNumarasi { get; set; }
        public float SecilenBarAcilisFiyati { get; set; }
        public float SecilenBarYuksekFiyati { get; set; }
        public float SecilenBarDusukFiyati { get; set; }
        public float SecilenBarKapanisFiyati { get; set; }

        public DateTime SonBarTarihi { get; set; }
        public float SonBarAcilisFiyati { get; set; }
        public float SonBarYuksekFiyati { get; set; }
        public float SonBarDusukFiyati { get; set; }
        public float SonBarKapanisFiyati { get; set; }
        public int SonBarIndex { get; set; }

        public List<int> AyBarIndeksList { get; set; }
        public List<int> HaftaBarIndeksList { get; set; }
        public List<int> GunBarIndeksList { get; set; }
        public List<int> SaatBarIndeksList { get; set; }
        public List<int> DakikaBarIndeksList { get; set; }

        public List<float> KapanisListesi5Dak { get; set; }
        float Kapanis5Dak = 0f;
        public List<float> KapanisListesi10Dak { get; set; }
        float Kapanis10Dak = 0f;
        public List<float> KapanisListesi15Dak { get; set; }
        float Kapanis15Dak = 0f;
        public List<float> KapanisListesi20Dak { get; set; }
        float Kapanis20Dak = 0f;
        public List<float> KapanisListesi25Dak { get; set; }
        float Kapanis25Dak = 0f;
        public List<float> KapanisListesi30Dak { get; set; }
        float Kapanis30Dak = 0f;
        public List<float> KapanisListesi35Dak { get; set; }
        float Kapanis35Dak = 0f;
        public List<float> KapanisListesi40Dak { get; set; }
        float Kapanis40Dak = 0f;
        public List<float> KapanisListesi45Dak { get; set; }
        float Kapanis45Dak = 0f;
        public List<float> KapanisListesi50Dak { get; set; }
        float Kapanis50Dak = 0f;
        public List<float> KapanisListesi60Dak { get; set; }
        float Kapanis60Dak = 0f;
        public List<float> KapanisListesi90Dak { get; set; }
        float Kapanis90Dak = 0f;
        public List<float> KapanisListesi100Dak { get; set; }
        float Kapanis100Dak = 0f;
        public List<float> KapanisListesi120Dak { get; set; }
        float Kapanis120Dak = 0f;
        public List<float> KapanisListesi150Dak { get; set; }
        float Kapanis150Dak = 0f;
        public List<float> KapanisListesi180Dak { get; set; }
        float Kapanis180Dak = 0f;
        public List<float> KapanisListesi200Dak { get; set; }
        float Kapanis200Dak = 0f;
        public List<float> KapanisListesi240Dak { get; set; }
        float Kapanis240Dak = 0f;
        public List<float> KapanisListesi300Dak { get; set; }
        float Kapanis300Dak = 0f;
        public List<float> KapanisListesi360Dak { get; set; }
        float Kapanis360Dak = 0f;
        public List<float> KapanisListesi400Dak { get; set; }
        float Kapanis400Dak = 0f;
        public List<float> KapanisListesi420Dak { get; set; }
        float Kapanis420Dak = 0f;
        public List<float> KapanisListesi480Dak { get; set; }
        float Kapanis480Dak = 0f;
        public List<float> KapanisListesi500Dak { get; set; }
        float Kapanis500Dak = 0f;
        public List<float> KapanisListesi600Dak { get; set; }
        float Kapanis600Dak = 0f;
        public List<float> KapanisListesiSaat { get; set; }
        float KapanisSaat = 0f;
        public List<float> KapanisListesiGun { get; set; }
        float KapanisGun = 0f;
        public List<float> KapanisListesiHafta { get; set; }
        float KapanisHafta = 0f;
        public List<float> KapanisListesiAy { get; set; }
        float KapanisAy = 0f;

        public List<float> YuksekListesi5Dak { get; set; }
        float Yuksek5Dak = 0f;
        public List<float> YuksekListesi10Dak { get; set; }
        float Yuksek10Dak = 0f;
        public List<float> YuksekListesi15Dak { get; set; }
        float Yuksek15Dak = 0f;
        public List<float> YuksekListesi20Dak { get; set; }
        float Yuksek20Dak = 0f;
        public List<float> YuksekListesi25Dak { get; set; }
        float Yuksek25Dak = 0f;
        public List<float> YuksekListesi30Dak { get; set; }
        float Yuksek30Dak = 0f;
        public List<float> YuksekListesi35Dak { get; set; }
        float Yuksek35Dak = 0f;
        public List<float> YuksekListesi40Dak { get; set; }
        float Yuksek40Dak = 0f;
        public List<float> YuksekListesi45Dak { get; set; }
        float Yuksek45Dak = 0f;
        public List<float> YuksekListesi50Dak { get; set; }
        float Yuksek50Dak = 0f;
        public List<float> YuksekListesi60Dak { get; set; }
        float Yuksek60Dak = 0f;
        public List<float> YuksekListesi90Dak { get; set; }
        float Yuksek90Dak = 0f;
        public List<float> YuksekListesi100Dak { get; set; }
        float Yuksek100Dak = 0f;
        public List<float> YuksekListesi120Dak { get; set; }
        float Yuksek120Dak = 0f;
        public List<float> YuksekListesi150Dak { get; set; }
        float Yuksek150Dak = 0f;
        public List<float> YuksekListesi180Dak { get; set; }
        float Yuksek180Dak = 0f;
        public List<float> YuksekListesi200Dak { get; set; }
        float Yuksek200Dak = 0f;
        public List<float> YuksekListesi240Dak { get; set; }
        float Yuksek240Dak = 0f;
        public List<float> YuksekListesi300Dak { get; set; }
        float Yuksek300Dak = 0f;
        public List<float> YuksekListesi360Dak { get; set; }
        float Yuksek360Dak = 0f;
        public List<float> YuksekListesi400Dak { get; set; }
        float Yuksek400Dak = 0f;
        public List<float> YuksekListesi420Dak { get; set; }
        float Yuksek420Dak = 0f;
        public List<float> YuksekListesi480Dak { get; set; }
        float Yuksek480Dak = 0f;
        public List<float> YuksekListesi500Dak { get; set; }
        float Yuksek500Dak = 0f;
        public List<float> YuksekListesi600Dak { get; set; }
        float Yuksek600Dak = 0f;
        public List<float> YuksekListesiSaat { get; set; }
        float YuksekSaat = 0f;
        public List<float> YuksekListesiGun { get; set; }
        float YuksekGun = 0f;
        public List<float> YuksekListesiHafta { get; set; }
        float YuksekHafta = 0f;
        public List<float> YuksekListesiAy { get; set; }
        float YuksekAy = 0f;

        public List<float> DusukListesi5Dak { get; set; }
        float Dusuk5Dak = 0f;
        public List<float> DusukListesi10Dak { get; set; }
        float Dusuk10Dak = 0f;
        public List<float> DusukListesi15Dak { get; set; }
        float Dusuk15Dak = 0f;
        public List<float> DusukListesi20Dak { get; set; }
        float Dusuk20Dak = 0f;
        public List<float> DusukListesi25Dak { get; set; }
        float Dusuk25Dak = 0f;
        public List<float> DusukListesi30Dak { get; set; }
        float Dusuk30Dak = 0f;
        public List<float> DusukListesi35Dak { get; set; }
        float Dusuk35Dak = 0f;
        public List<float> DusukListesi40Dak { get; set; }
        float Dusuk40Dak = 0f;
        public List<float> DusukListesi45Dak { get; set; }
        float Dusuk45Dak = 0f;
        public List<float> DusukListesi50Dak { get; set; }
        float Dusuk50Dak = 0f;
        public List<float> DusukListesi60Dak { get; set; }
        float Dusuk60Dak = 0f;
        public List<float> DusukListesi90Dak { get; set; }
        float Dusuk90Dak = 0f;
        public List<float> DusukListesi100Dak { get; set; }
        float Dusuk100Dak = 0f;
        public List<float> DusukListesi120Dak { get; set; }
        float Dusuk120Dak = 0f;
        public List<float> DusukListesi150Dak { get; set; }
        float Dusuk150Dak = 0f;
        public List<float> DusukListesi180Dak { get; set; }
        float Dusuk180Dak = 0f;
        public List<float> DusukListesi200Dak { get; set; }
        float Dusuk200Dak = 0f;
        public List<float> DusukListesi240Dak { get; set; }
        float Dusuk240Dak = 0f;
        public List<float> DusukListesi300Dak { get; set; }
        float Dusuk300Dak = 0f;
        public List<float> DusukListesi360Dak { get; set; }
        float Dusuk360Dak = 0f;
        public List<float> DusukListesi400Dak { get; set; }
        float Dusuk400Dak = 0f;
        public List<float> DusukListesi420Dak { get; set; }
        float Dusuk420Dak = 0f;
        public List<float> DusukListesi480Dak { get; set; }
        float Dusuk480Dak = 0f;
        public List<float> DusukListesi500Dak { get; set; }
        float Dusuk500Dak = 0f;
        public List<float> DusukListesi600Dak { get; set; }
        float Dusuk600Dak = 0f;
        public List<float> DusukListesiSaat { get; set; }
        float DusukSaat = 0f;
        public List<float> DusukListesiGun { get; set; }
        float DusukGun = 0f;
        public List<float> DusukListesiHafta { get; set; }
        float DusukHafta = 0f;
        public List<float> DusukListesiAy { get; set; }
        float DusukAy = 0f;

        public List<float> AcilisListesi5Dak { get; set; }
        float Acilis5Dak = 0f;
        public List<float> AcilisListesi10Dak { get; set; }
        float Acilis10Dak = 0f;
        public List<float> AcilisListesi15Dak { get; set; }
        float Acilis15Dak = 0f;
        public List<float> AcilisListesi20Dak { get; set; }
        float Acilis20Dak = 0f;
        public List<float> AcilisListesi25Dak { get; set; }
        float Acilis25Dak = 0f;
        public List<float> AcilisListesi30Dak { get; set; }
        float Acilis30Dak = 0f;
        public List<float> AcilisListesi35Dak { get; set; }
        float Acilis35Dak = 0f;
        public List<float> AcilisListesi40Dak { get; set; }
        float Acilis40Dak = 0f;
        public List<float> AcilisListesi45Dak { get; set; }
        float Acilis45Dak = 0f;
        public List<float> AcilisListesi50Dak { get; set; }
        float Acilis50Dak = 0f;
        public List<float> AcilisListesi60Dak { get; set; }
        float Acilis60Dak = 0f;
        public List<float> AcilisListesi90Dak { get; set; }
        float Acilis90Dak = 0f;
        public List<float> AcilisListesi100Dak { get; set; }
        float Acilis100Dak = 0f;
        public List<float> AcilisListesi120Dak { get; set; }
        float Acilis120Dak = 0f;
        public List<float> AcilisListesi150Dak { get; set; }
        float Acilis150Dak = 0f;
        public List<float> AcilisListesi180Dak { get; set; }
        float Acilis180Dak = 0f;
        public List<float> AcilisListesi200Dak { get; set; }
        float Acilis200Dak = 0f;
        public List<float> AcilisListesi240Dak { get; set; }
        float Acilis240Dak = 0f;
        public List<float> AcilisListesi300Dak { get; set; }
        float Acilis300Dak = 0f;
        public List<float> AcilisListesi360Dak { get; set; }
        float Acilis360Dak = 0f;
        public List<float> AcilisListesi400Dak { get; set; }
        float Acilis400Dak = 0f;
        public List<float> AcilisListesi420Dak { get; set; }
        float Acilis420Dak = 0f;
        public List<float> AcilisListesi480Dak { get; set; }
        float Acilis480Dak = 0f;
        public List<float> AcilisListesi500Dak { get; set; }
        float Acilis500Dak = 0f;
        public List<float> AcilisListesi600Dak { get; set; }
        float Acilis600Dak = 0f;
        public List<float> AcilisListesiSaat { get; set; }
        float AcilisSaat = 0f;
        public List<float> AcilisListesiGun { get; set; }
        float AcilisGun = 0f;
        public List<float> AcilisListesiHafta { get; set; }
        float AcilisHafta = 0f;
        public List<float> AcilisListesiAy { get; set; }
        float AcilisAy = 0f;

        public List<int> XBarIndeksList { get; set; }
        int XBarIndeks = 0;

        public List<float> XBarKapanisListesi { get; set; }
        float XBarKapanis = 0f;
        public List<float> XBarAcilisListesi { get; set; }
        float XBarAcilis = 0f;
        public List<float> XBarYuksekListesi { get; set; }
        float XBarYuksek = 0f;
        public List<float> XBarDusukListesi { get; set; }
        float XBarDusuk = 0f;
        public List<float> XBarHacimListesi { get; set; }
        float XBarHacim = 0f;
        public List<float> XBarLotListesi { get; set; }
        float XBarLot = 0f;

        int MaxSayacCount = 20;
        int[] Sayac5Dak = new int[20];

        int Sayac60Dk;

        float BarYuzdesi = 0f;
        public List<float> BarYuzdeListesiHL  { get; set; }
        public List<float> BarYuzdeListesiOC  { get; set; }

        ~CBarUtils()
        {
        }

        public CBarUtils()
        {

        }

        public CBarUtils Initialize(dynamic Sistem, dynamic V, dynamic Open, dynamic High, dynamic Low, dynamic Close, dynamic Volume, dynamic Lot)
        {
            SetData(Sistem, V, Open, High, Low, Close, Volume, Lot);

            TimeUtils = new CTimeUtils();
            TimeUtils.Initialize(Sistem, V, Open, High, Low, Close, Volume, Lot);

            AyBarIndeksList = new List<int>();
            HaftaBarIndeksList = new List<int>();
            GunBarIndeksList = new List<int>();
            SaatBarIndeksList = new List<int>();
            DakikaBarIndeksList = new List<int>();
            XBarIndeksList = new List<int>();

            Reset(Sistem);

            return this;
        }

        public CBarUtils Reset(dynamic Sistem)
        {
            AyBarIndeksList.Clear();
            HaftaBarIndeksList.Clear();
            GunBarIndeksList.Clear();
            SaatBarIndeksList.Clear();
            DakikaBarIndeksList.Clear();
            XBarIndeksList.Clear();

            KapanisListeleriniResetle(Sistem);
            YuksekListeleriniResetle(Sistem);
            DusukListleriniResetle(Sistem);
            AcilisListleriniResetle(Sistem);

            XBarListleriniResetle(Sistem);

            XBarIndeksListesiniResetle(Sistem);

            BarYuzdeListesiniResetle(Sistem);

            return this;
        }

        public string GetBarValuesDescription(dynamic Sistem)
        {
            string LogMessage = String.Format("#  {0} : {1},{2},{3},{4},{5},{6},{7},{8},{9}", "Column Names", "No", "Date", "Time", "Open", "High", "Low", "Close", "Vol", "Size(Lot)");
            return LogMessage;
        }

        public string GetBarValuesAsString(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;
            string LogMessage = String.Format("{0,-5} 	 {1,-20} 	 {2, 5} 	 {3, 5} 	 {4, 5} 	 {5, 5} 	 {6, 10} 	 {7, 5} 	 ", i, V[i].Date.ToString("yyyy.MM.dd HH:mm:ss"), V[i].Open.ToString("0.00"), V[i].High.ToString("0.00"), V[i].Low.ToString("0.00"), V[i].Close.ToString("0.00"), V[i].Vol.ToString("0"), V[i].Size.ToString("0"));
            return LogMessage;
        }

        public void BarlariYumusat(dynamic Sistem, int BarPeriyot = 5, string BarMAYontem = "Exp", bool GercekVerilereUygula = false)
        {

            this.BarMAYontem = BarMAYontem;
            this.BarPeriyot = BarPeriyot;
            BarOpen = Sistem.MA(Open, BarMAYontem, BarPeriyot);
            BarHigh = Sistem.MA(High, BarMAYontem, BarPeriyot);
            BarLow = Sistem.MA(Low, BarMAYontem, BarPeriyot);
            BarClose = Sistem.MA(Close, BarMAYontem, BarPeriyot);
            BarVolume = Volume;

            if (GercekVerilereUygula)
            {
                Open = BarOpen;
                High = BarHigh;
                Low = BarLow;
                Close = BarClose;
                Volume = BarVolume;
                Lot = BarLot;
            }
        }

        public void BarlariCiz(dynamic Sistem, int BarPanelNo = 1, int BarTip = 2)
        {
            this.BarPanelNo = BarPanelNo;
            this.BarTip = BarTip;

            Sistem.BarCiz(BarPanelNo, BarTip, BarOpen, BarHigh, BarLow, BarClose, Color.Green, Color.Red);
        }

        public void SonBarBilgileriniAl(dynamic Sistem)
        {
            var LastBar = V.Count - 1;
            SonBarTarihi = V[LastBar].Date;
            SonBarAcilisFiyati = V[LastBar].Open;
            SonBarYuksekFiyati = V[LastBar].High;
            SonBarDusukFiyati = V[LastBar].Low;
            SonBarKapanisFiyati = V[LastBar].Close;
            SonBarIndex = V.Count - 1;
        }

        public void SecilenBarBilgileriniAl(dynamic Sistem)
        {
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
        }

        public void AyGunSaatDakikaIndeksListeleriniGuncelle(dynamic Sistem, int BarIndex, CTimeUtils TimeUtils)
        {
            var IsYeniAy = TimeUtils.IsYeniAy(Sistem, BarIndex);
            var IsYeniHafta = TimeUtils.IsYeniHafta(Sistem, BarIndex);
            var IsYeniGun = TimeUtils.IsYeniGun(Sistem, BarIndex);
            var IsYeniSaat = TimeUtils.IsYeniSaat(Sistem, BarIndex);

            if (IsYeniAy)
                AyBarIndeksList.Add(BarIndex);

            if (IsYeniHafta)
                HaftaBarIndeksList.Add(BarIndex);

            if (IsYeniGun)
                GunBarIndeksList.Add(BarIndex);

            if (IsYeniSaat)
                SaatBarIndeksList.Add(BarIndex);

        }

        public DateTime GetBarDateTime(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;
            return V[i].Date;
        }

        public int KapanisListleriniResetle(dynamic Sistem)
        {
            Sayac60Dk = 0;

            Kapanis5Dak = 0f;
            Kapanis10Dak = 0f;
            Kapanis15Dak = 0f;
            Kapanis20Dak = 0f;
            Kapanis25Dak = 0f;
            Kapanis30Dak = 0f;
            Kapanis35Dak = 0f;
            Kapanis40Dak = 0f;
            Kapanis45Dak = 0f;
            Kapanis50Dak = 0f;
            Kapanis60Dak = 0f;
            Kapanis90Dak = 0f;
            Kapanis100Dak = 0f;
            Kapanis120Dak = 0f;
            Kapanis150Dak = 0f;
            Kapanis180Dak = 0f;
            Kapanis200Dak = 0f;
            Kapanis240Dak = 0f;
            Kapanis300Dak = 0f;
            Kapanis360Dak = 0f;
            Kapanis400Dak = 0f;
            Kapanis420Dak = 0f;
            Kapanis480Dak = 0f;
            Kapanis500Dak = 0f;
            Kapanis600Dak = 0f;
            KapanisSaat = 0f;
            KapanisGun = 0f;
            KapanisHafta = 0f;
            KapanisAy = 0f;

            KapanisListesi5Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi10Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi15Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi20Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi25Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi30Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi35Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi40Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi45Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi50Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi60Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi90Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi100Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi120Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi150Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi180Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi200Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi240Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi300Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi360Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi400Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi420Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi480Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi500Dak = Sistem.Liste(BarCount, 0);

            KapanisListesi600Dak = Sistem.Liste(BarCount, 0);

            KapanisListesiSaat = Sistem.Liste(BarCount, 0);

            KapanisListesiGun = Sistem.Liste(BarCount, 0);

            KapanisListesiHafta = Sistem.Liste(BarCount, 0);

            KapanisListesiAy = Sistem.Liste(BarCount, 0);

            for (int i = 0; i < MaxSayacCount; i++)
                Sayac5Dak[i] = 0;

            return 0;
        }

        public void KapanisListleriniGuncelle(dynamic Sistem, int BarIndex, CTimeUtils TimeUtils)
        {
            int i = BarIndex;

            var BarDateTime = GetBarDateTime(Sistem, BarIndex);
            var IsYeniAy = TimeUtils.IsYeniAy(Sistem, BarIndex);
            var IsYeniHafta = TimeUtils.IsYeniHafta(Sistem, BarIndex);
            var IsYeniGun = TimeUtils.IsYeniGun(Sistem, BarIndex);
            var IsYeniSaat = TimeUtils.IsYeniSaat(Sistem, BarIndex);
            var Source = Close;

            {
                if (BarDateTime.Minute % 5 == 0)
                {
                    for (int j = 0; j < MaxSayacCount; j++)
                        Sayac5Dak[j] += 1;
                }

                if (BarDateTime.Minute % 5 == 0)
                    Kapanis5Dak = Source[i];

                if (BarDateTime.Minute % 10 == 0)
                    Kapanis10Dak = Source[i];

                if (BarDateTime.Minute % 15 == 0)
                    Kapanis15Dak = Source[i];

                if (BarDateTime.Minute % 20 == 0)
                    Kapanis20Dak = Source[i];

                if (BarDateTime.Minute % 25 == 0)
                    Kapanis25Dak = Source[i];

                if (BarDateTime.Minute % 30 == 0)
                    Kapanis30Dak = Source[i];

                if (BarDateTime.Minute % 35 == 0)
                    Kapanis35Dak = Source[i];

                if (BarDateTime.Minute % 40 == 0)
                    Kapanis40Dak = Source[i];

                if (BarDateTime.Minute % 45 == 0)
                    Kapanis45Dak = Source[i];

                if (BarDateTime.Minute % 50 == 0)
                    Kapanis50Dak = Source[i];

                if (BarDateTime.Minute % 60 == 0)
                {
                    Kapanis60Dak = Source[i];
                    Sayac60Dk++;
                }

                if (Sayac5Dak[0] % (90 / 5) == 0)
                    Kapanis90Dak = Source[i];

                if (Sayac5Dak[0] % (100 / 5) == 0)
                    Kapanis100Dak = Source[i];

                if (Sayac60Dk % 2 == 0)
                    Kapanis120Dak = Source[i];

                if (Sayac5Dak[0] % (150 / 5) == 0)
                    Kapanis150Dak = Source[i];

                if (Sayac60Dk % 3 == 0)
                    Kapanis180Dak = Source[i];

                if (Sayac5Dak[0] % (200 / 5) == 0)
                    Kapanis200Dak = Source[i];

                if (Sayac60Dk % 4 == 0)
                    Kapanis240Dak = Source[i];

                if (Sayac60Dk % 5 == 0)
                    Kapanis300Dak = Source[i];

                if (Sayac60Dk % 6 == 0)
                    Kapanis360Dak = Source[i];

                if (Sayac5Dak[0] % (400 / 5) == 0)
                    Kapanis400Dak = Source[i];

                if (Sayac60Dk % 7 == 0)
                    Kapanis420Dak = Source[i];

                if (Sayac60Dk % 8 == 0)
                    Kapanis480Dak = Source[i];

                if (Sayac5Dak[0] % (500 / 5) == 0)
                    Kapanis500Dak = Source[i];

                if (Sayac5Dak[0] % (600 / 5) == 0)
                    Kapanis600Dak = Source[i];

                if (IsYeniSaat)
                    KapanisSaat = Source[i];

                if (IsYeniGun)
                    KapanisGun = Source[i];

                if (IsYeniHafta)
                    KapanisHafta = Source[i];

                if (IsYeniAy)
                    KapanisAy = Source[i];
            }

            KapanisListesi5Dak[i] = Kapanis5Dak;
            KapanisListesi10Dak[i] = Kapanis10Dak;
            KapanisListesi15Dak[i] = Kapanis15Dak;
            KapanisListesi20Dak[i] = Kapanis20Dak;
            KapanisListesi25Dak[i] = Kapanis25Dak;
            KapanisListesi30Dak[i] = Kapanis30Dak;
            KapanisListesi35Dak[i] = Kapanis35Dak;
            KapanisListesi40Dak[i] = Kapanis40Dak;
            KapanisListesi45Dak[i] = Kapanis45Dak;
            KapanisListesi50Dak[i] = Kapanis50Dak;
            KapanisListesi60Dak[i] = Kapanis60Dak;
            KapanisListesi90Dak[i] = Kapanis90Dak;
            KapanisListesi100Dak[i] = Kapanis100Dak;
            KapanisListesi120Dak[i] = Kapanis120Dak;
            KapanisListesi150Dak[i] = Kapanis150Dak;
            KapanisListesi180Dak[i] = Kapanis180Dak;
            KapanisListesi200Dak[i] = Kapanis200Dak;
            KapanisListesi240Dak[i] = Kapanis240Dak;
            KapanisListesi300Dak[i] = Kapanis300Dak;
            KapanisListesi360Dak[i] = Kapanis360Dak;
            KapanisListesi400Dak[i] = Kapanis400Dak;
            KapanisListesi420Dak[i] = Kapanis420Dak;
            KapanisListesi480Dak[i] = Kapanis480Dak;
            KapanisListesi500Dak[i] = Kapanis500Dak;
            KapanisListesi600Dak[i] = Kapanis600Dak;
            KapanisListesiSaat[i] = KapanisSaat;
            KapanisListesiGun[i] = KapanisGun;
            KapanisListesiHafta[i] = KapanisHafta;
            KapanisListesiAy[i] = KapanisAy;

            if (IsYeniSaat)
            {
                for (int j = 0; j < MaxSayacCount; j++)
                    Sayac5Dak[j] = 0;
            }

        }

        public int YuksekListleriniResetle(dynamic Sistem)
        {
            Sayac60Dk = 0;

            Yuksek5Dak = 0f;
            Yuksek10Dak = 0f;
            Yuksek15Dak = 0f;
            Yuksek20Dak = 0f;
            Yuksek25Dak = 0f;
            Yuksek30Dak = 0f;
            Yuksek35Dak = 0f;
            Yuksek40Dak = 0f;
            Yuksek45Dak = 0f;
            Yuksek50Dak = 0f;
            Yuksek60Dak = 0f;
            Yuksek90Dak = 0f;
            Yuksek100Dak = 0f;
            Yuksek120Dak = 0f;
            Yuksek150Dak = 0f;
            Yuksek180Dak = 0f;
            Yuksek200Dak = 0f;
            Yuksek240Dak = 0f;
            Yuksek300Dak = 0f;
            Yuksek360Dak = 0f;
            Yuksek400Dak = 0f;
            Yuksek420Dak = 0f;
            Yuksek480Dak = 0f;
            Yuksek500Dak = 0f;
            Yuksek600Dak = 0f;
            YuksekSaat = 0f;
            YuksekGun = 0f;
            YuksekHafta = 0f;
            YuksekAy = 0f;

            YuksekListesi5Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi10Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi15Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi20Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi25Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi30Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi35Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi40Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi45Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi50Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi60Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi90Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi100Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi120Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi150Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi180Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi200Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi240Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi300Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi360Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi400Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi420Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi480Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi500Dak = Sistem.Liste(BarCount, 0);

            YuksekListesi600Dak = Sistem.Liste(BarCount, 0);

            YuksekListesiSaat = Sistem.Liste(BarCount, 0);

            YuksekListesiGun = Sistem.Liste(BarCount, 0);

            YuksekListesiHafta = Sistem.Liste(BarCount, 0);

            YuksekListesiAy = Sistem.Liste(BarCount, 0);

            for (int i = 0; i < MaxSayacCount; i++)
                Sayac5Dak[i] = 0;

            return 0;
        }

        public void YuksekListleriniGuncelle(dynamic Sistem, int BarIndex, CTimeUtils TimeUtils)
        {
            int i = BarIndex;

            var BarDateTime = GetBarDateTime(Sistem, BarIndex);
            var IsYeniAy = TimeUtils.IsYeniAy(Sistem, BarIndex);
            var IsYeniHafta = TimeUtils.IsYeniHafta(Sistem, BarIndex);
            var IsYeniGun = TimeUtils.IsYeniGun(Sistem, BarIndex);
            var IsYeniSaat = TimeUtils.IsYeniSaat(Sistem, BarIndex);
            var Source = High;

            {
                if (BarDateTime.Minute % 5 == 0)
                {
                    for (int j = 0; j < MaxSayacCount; j++)
                        Sayac5Dak[j] += 1;
                }

                if (BarDateTime.Minute % 5 == 0)
                    Yuksek5Dak = Source[i];

                if (BarDateTime.Minute % 10 == 0)
                    Yuksek10Dak = Source[i];

                if (BarDateTime.Minute % 15 == 0)
                    Yuksek15Dak = Source[i];

                if (BarDateTime.Minute % 20 == 0)
                    Yuksek20Dak = Source[i];

                if (BarDateTime.Minute % 25 == 0)
                    Yuksek25Dak = Source[i];

                if (BarDateTime.Minute % 30 == 0)
                    Yuksek30Dak = Source[i];

                if (BarDateTime.Minute % 35 == 0)
                    Yuksek35Dak = Source[i];

                if (BarDateTime.Minute % 40 == 0)
                    Yuksek40Dak = Source[i];

                if (BarDateTime.Minute % 45 == 0)
                    Yuksek45Dak = Source[i];

                if (BarDateTime.Minute % 50 == 0)
                    Yuksek50Dak = Source[i];

                if (BarDateTime.Minute % 60 == 0)
                {
                    Yuksek60Dak = Source[i];
                    Sayac60Dk++;
                }

                if (Sayac5Dak[0] % (90 / 5) == 0)
                    Yuksek90Dak = Source[i];

                if (Sayac5Dak[0] % (100 / 5) == 0)
                    Yuksek100Dak = Source[i];

                if (Sayac60Dk % 2 == 0)
                    Yuksek120Dak = Source[i];

                if (Sayac5Dak[0] % (150 / 5) == 0)
                    Yuksek150Dak = Source[i];

                if (Sayac60Dk % 3 == 0)
                    Yuksek180Dak = Source[i];

                if (Sayac5Dak[0] % (200 / 5) == 0)
                    Yuksek200Dak = Source[i];

                if (Sayac60Dk % 4 == 0)
                    Yuksek240Dak = Source[i];

                if (Sayac60Dk % 5 == 0)
                    Yuksek300Dak = Source[i];

                if (Sayac60Dk % 6 == 0)
                    Yuksek360Dak = Source[i];

                if (Sayac5Dak[0] % (400 / 5) == 0)
                    Yuksek400Dak = Source[i];

                if (Sayac60Dk % 7 == 0)
                    Yuksek420Dak = Source[i];

                if (Sayac60Dk % 8 == 0)
                    Yuksek480Dak = Source[i];

                if (Sayac5Dak[0] % (500 / 5) == 0)
                    Yuksek500Dak = Source[i];

                if (Sayac5Dak[0] % (600 / 5) == 0)
                    Yuksek600Dak = Source[i];

                if (IsYeniSaat)
                    YuksekSaat = Source[i];

                if (IsYeniGun)
                    YuksekGun = Source[i];

                if (IsYeniHafta)
                    YuksekHafta = Source[i];

                if (IsYeniAy)
                    YuksekAy = Source[i];
            }

            YuksekListesi5Dak[i] = Yuksek5Dak;
            YuksekListesi10Dak[i] = Yuksek10Dak;
            YuksekListesi15Dak[i] = Yuksek15Dak;
            YuksekListesi20Dak[i] = Yuksek20Dak;
            YuksekListesi25Dak[i] = Yuksek25Dak;
            YuksekListesi30Dak[i] = Yuksek30Dak;
            YuksekListesi35Dak[i] = Yuksek35Dak;
            YuksekListesi40Dak[i] = Yuksek40Dak;
            YuksekListesi45Dak[i] = Yuksek45Dak;
            YuksekListesi50Dak[i] = Yuksek50Dak;
            YuksekListesi60Dak[i] = Yuksek60Dak;
            YuksekListesi90Dak[i] = Yuksek90Dak;
            YuksekListesi100Dak[i] = Yuksek100Dak;
            YuksekListesi120Dak[i] = Yuksek120Dak;
            YuksekListesi150Dak[i] = Yuksek150Dak;
            YuksekListesi180Dak[i] = Yuksek180Dak;
            YuksekListesi200Dak[i] = Yuksek200Dak;
            YuksekListesi240Dak[i] = Yuksek240Dak;
            YuksekListesi300Dak[i] = Yuksek300Dak;
            YuksekListesi360Dak[i] = Yuksek360Dak;
            YuksekListesi400Dak[i] = Yuksek400Dak;
            YuksekListesi420Dak[i] = Yuksek420Dak;
            YuksekListesi480Dak[i] = Yuksek480Dak;
            YuksekListesi500Dak[i] = Yuksek500Dak;
            YuksekListesi600Dak[i] = Yuksek600Dak;
            YuksekListesiSaat[i] = YuksekSaat;
            YuksekListesiGun[i] = YuksekGun;
            YuksekListesiHafta[i] = YuksekHafta;
            YuksekListesiAy[i] = YuksekAy;

            if (IsYeniSaat)
            {
                for (int j = 0; j < MaxSayacCount; j++)
                    Sayac5Dak[j] = 0;
            }
        }

        public int DusukListleriniResetle(dynamic Sistem)
        {
            Sayac60Dk = 0;

            Dusuk5Dak = 0f;
            Dusuk10Dak = 0f;
            Dusuk15Dak = 0f;
            Dusuk20Dak = 0f;
            Dusuk25Dak = 0f;
            Dusuk30Dak = 0f;
            Dusuk35Dak = 0f;
            Dusuk40Dak = 0f;
            Dusuk45Dak = 0f;
            Dusuk50Dak = 0f;
            Dusuk60Dak = 0f;
            Dusuk90Dak = 0f;
            Dusuk100Dak = 0f;
            Dusuk120Dak = 0f;
            Dusuk150Dak = 0f;
            Dusuk180Dak = 0f;
            Dusuk200Dak = 0f;
            Dusuk240Dak = 0f;
            Dusuk300Dak = 0f;
            Dusuk360Dak = 0f;
            Dusuk400Dak = 0f;
            Dusuk420Dak = 0f;
            Dusuk480Dak = 0f;
            Dusuk500Dak = 0f;
            Dusuk600Dak = 0f;
            DusukSaat = 0f;
            DusukGun = 0f;
            DusukHafta = 0f;
            DusukAy = 0f;

            DusukListesi5Dak = Sistem.Liste(BarCount, 0);

            DusukListesi10Dak = Sistem.Liste(BarCount, 0);

            DusukListesi15Dak = Sistem.Liste(BarCount, 0);

            DusukListesi20Dak = Sistem.Liste(BarCount, 0);

            DusukListesi25Dak = Sistem.Liste(BarCount, 0);

            DusukListesi30Dak = Sistem.Liste(BarCount, 0);

            DusukListesi35Dak = Sistem.Liste(BarCount, 0);

            DusukListesi40Dak = Sistem.Liste(BarCount, 0);

            DusukListesi45Dak = Sistem.Liste(BarCount, 0);

            DusukListesi50Dak = Sistem.Liste(BarCount, 0);

            DusukListesi60Dak = Sistem.Liste(BarCount, 0);

            DusukListesi90Dak = Sistem.Liste(BarCount, 0);

            DusukListesi100Dak = Sistem.Liste(BarCount, 0);

            DusukListesi120Dak = Sistem.Liste(BarCount, 0);

            DusukListesi150Dak = Sistem.Liste(BarCount, 0);

            DusukListesi180Dak = Sistem.Liste(BarCount, 0);

            DusukListesi200Dak = Sistem.Liste(BarCount, 0);

            DusukListesi240Dak = Sistem.Liste(BarCount, 0);

            DusukListesi300Dak = Sistem.Liste(BarCount, 0);

            DusukListesi360Dak = Sistem.Liste(BarCount, 0);

            DusukListesi400Dak = Sistem.Liste(BarCount, 0);

            DusukListesi420Dak = Sistem.Liste(BarCount, 0);

            DusukListesi480Dak = Sistem.Liste(BarCount, 0);

            DusukListesi500Dak = Sistem.Liste(BarCount, 0);

            DusukListesi600Dak = Sistem.Liste(BarCount, 0);

            DusukListesiSaat = Sistem.Liste(BarCount, 0);

            DusukListesiGun = Sistem.Liste(BarCount, 0);

            DusukListesiHafta = Sistem.Liste(BarCount, 0);

            DusukListesiAy = Sistem.Liste(BarCount, 0);

            for (int i = 0; i < MaxSayacCount; i++)
                Sayac5Dak[i] = 0;

            return 0;
        }

        public void DusukListleriniGuncelle(dynamic Sistem, int BarIndex, CTimeUtils TimeUtils)
        {
            int i = BarIndex;

            var BarDateTime = GetBarDateTime(Sistem, BarIndex);
            var IsYeniAy = TimeUtils.IsYeniAy(Sistem, BarIndex);
            var IsYeniHafta = TimeUtils.IsYeniHafta(Sistem, BarIndex);
            var IsYeniGun = TimeUtils.IsYeniGun(Sistem, BarIndex);
            var IsYeniSaat = TimeUtils.IsYeniSaat(Sistem, BarIndex);
            var Source = Low;

            {
                if (BarDateTime.Minute % 5 == 0)
                {
                    for (int j = 0; j < MaxSayacCount; j++)
                        Sayac5Dak[j] += 1;
                }

                if (BarDateTime.Minute % 5 == 0)
                    Dusuk5Dak = Source[i];

                if (BarDateTime.Minute % 10 == 0)
                    Dusuk10Dak = Source[i];

                if (BarDateTime.Minute % 15 == 0)
                    Dusuk15Dak = Source[i];

                if (BarDateTime.Minute % 20 == 0)
                    Dusuk20Dak = Source[i];

                if (BarDateTime.Minute % 25 == 0)
                    Dusuk25Dak = Source[i];

                if (BarDateTime.Minute % 30 == 0)
                    Dusuk30Dak = Source[i];

                if (BarDateTime.Minute % 35 == 0)
                    Dusuk35Dak = Source[i];

                if (BarDateTime.Minute % 40 == 0)
                    Dusuk40Dak = Source[i];

                if (BarDateTime.Minute % 45 == 0)
                    Dusuk45Dak = Source[i];

                if (BarDateTime.Minute % 50 == 0)
                    Dusuk50Dak = Source[i];

                if (BarDateTime.Minute % 60 == 0)
                {
                    Dusuk60Dak = Source[i];
                    Sayac60Dk++;
                }

                if (Sayac5Dak[0] % (90 / 5) == 0)
                    Dusuk90Dak = Source[i];

                if (Sayac5Dak[0] % (100 / 5) == 0)
                    Dusuk100Dak = Source[i];

                if (Sayac60Dk % 2 == 0)
                    Dusuk120Dak = Source[i];

                if (Sayac5Dak[0] % (150 / 5) == 0)
                    Dusuk150Dak = Source[i];

                if (Sayac60Dk % 3 == 0)
                    Dusuk180Dak = Source[i];

                if (Sayac5Dak[0] % (200 / 5) == 0)
                    Dusuk200Dak = Source[i];

                if (Sayac60Dk % 4 == 0)
                    Dusuk240Dak = Source[i];

                if (Sayac60Dk % 5 == 0)
                    Dusuk300Dak = Source[i];

                if (Sayac60Dk % 6 == 0)
                    Dusuk360Dak = Source[i];

                if (Sayac5Dak[0] % (400 / 5) == 0)
                    Dusuk400Dak = Source[i];

                if (Sayac60Dk % 7 == 0)
                    Dusuk420Dak = Source[i];

                if (Sayac60Dk % 8 == 0)
                    Dusuk480Dak = Source[i];

                if (Sayac5Dak[0] % (500 / 5) == 0)
                    Dusuk500Dak = Source[i];

                if (Sayac5Dak[0] % (600 / 5) == 0)
                    Dusuk600Dak = Source[i];

                if (IsYeniSaat)
                    DusukSaat = Source[i];

                if (IsYeniGun)
                    DusukGun = Source[i];

                if (IsYeniHafta)
                    DusukHafta = Source[i];

                if (IsYeniAy)
                    DusukAy = Source[i];
            }

            DusukListesi5Dak[i] = Dusuk5Dak;
            DusukListesi10Dak[i] = Dusuk10Dak;
            DusukListesi15Dak[i] = Dusuk15Dak;
            DusukListesi20Dak[i] = Dusuk20Dak;
            DusukListesi25Dak[i] = Dusuk25Dak;
            DusukListesi30Dak[i] = Dusuk30Dak;
            DusukListesi35Dak[i] = Dusuk35Dak;
            DusukListesi40Dak[i] = Dusuk40Dak;
            DusukListesi45Dak[i] = Dusuk45Dak;
            DusukListesi50Dak[i] = Dusuk50Dak;
            DusukListesi60Dak[i] = Dusuk60Dak;
            DusukListesi90Dak[i] = Dusuk90Dak;
            DusukListesi100Dak[i] = Dusuk100Dak;
            DusukListesi120Dak[i] = Dusuk120Dak;
            DusukListesi150Dak[i] = Dusuk150Dak;
            DusukListesi180Dak[i] = Dusuk180Dak;
            DusukListesi200Dak[i] = Dusuk200Dak;
            DusukListesi240Dak[i] = Dusuk240Dak;
            DusukListesi300Dak[i] = Dusuk300Dak;
            DusukListesi360Dak[i] = Dusuk360Dak;
            DusukListesi400Dak[i] = Dusuk400Dak;
            DusukListesi420Dak[i] = Dusuk420Dak;
            DusukListesi480Dak[i] = Dusuk480Dak;
            DusukListesi500Dak[i] = Dusuk500Dak;
            DusukListesi600Dak[i] = Dusuk600Dak;
            DusukListesiSaat[i] = DusukSaat;
            DusukListesiGun[i] = DusukGun;
            DusukListesiHafta[i] = DusukHafta;
            DusukListesiAy[i] = DusukAy;

            if (IsYeniSaat)
            {
                for (int j = 0; j < MaxSayacCount; j++)
                    Sayac5Dak[j] = 0;
            }
        }

        public int AcilisListleriniResetle(dynamic Sistem)
        {
            Sayac60Dk = 0;

            Acilis5Dak = 0f;
            Acilis10Dak = 0f;
            Acilis15Dak = 0f;
            Acilis20Dak = 0f;
            Acilis25Dak = 0f;
            Acilis30Dak = 0f;
            Acilis35Dak = 0f;
            Acilis40Dak = 0f;
            Acilis45Dak = 0f;
            Acilis50Dak = 0f;
            Acilis60Dak = 0f;
            Acilis90Dak = 0f;
            Acilis100Dak = 0f;
            Acilis120Dak = 0f;
            Acilis150Dak = 0f;
            Acilis180Dak = 0f;
            Acilis200Dak = 0f;
            Acilis240Dak = 0f;
            Acilis300Dak = 0f;
            Acilis360Dak = 0f;
            Acilis400Dak = 0f;
            Acilis420Dak = 0f;
            Acilis480Dak = 0f;
            Acilis500Dak = 0f;
            Acilis600Dak = 0f;
            AcilisSaat = 0f;
            AcilisGun = 0f;
            AcilisHafta = 0f;
            AcilisAy = 0f;

            AcilisListesi5Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi10Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi15Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi20Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi25Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi30Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi35Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi40Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi45Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi50Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi60Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi90Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi100Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi120Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi150Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi180Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi200Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi240Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi300Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi360Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi400Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi420Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi480Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi500Dak = Sistem.Liste(BarCount, 0);

            AcilisListesi600Dak = Sistem.Liste(BarCount, 0);

            AcilisListesiSaat = Sistem.Liste(BarCount, 0);

            AcilisListesiGun = Sistem.Liste(BarCount, 0);

            AcilisListesiHafta = Sistem.Liste(BarCount, 0);

            AcilisListesiAy = Sistem.Liste(BarCount, 0);

            for (int i = 0; i < MaxSayacCount; i++)
                Sayac5Dak[i] = 0;

            return 0;
        }

        public void AcilisListleriniGuncelle(dynamic Sistem, int BarIndex, CTimeUtils TimeUtils)
        {
            int i = BarIndex;

            var BarDateTime = GetBarDateTime(Sistem, BarIndex);
            var IsYeniAy = TimeUtils.IsYeniAy(Sistem, BarIndex);
            var IsYeniHafta = TimeUtils.IsYeniHafta(Sistem, BarIndex);
            var IsYeniGun = TimeUtils.IsYeniGun(Sistem, BarIndex);
            var IsYeniSaat = TimeUtils.IsYeniSaat(Sistem, BarIndex);
            var Source = Close;

            {
                if (BarDateTime.Minute % 5 == 0)
                {
                    for (int j = 0; j < MaxSayacCount; j++)
                        Sayac5Dak[j] += 1;
                }

                if (BarDateTime.Minute % 5 == 0)
                    Acilis5Dak = Source[i];

                if (BarDateTime.Minute % 10 == 0)
                    Acilis10Dak = Source[i];

                if (BarDateTime.Minute % 15 == 0)
                    Acilis15Dak = Source[i];

                if (BarDateTime.Minute % 20 == 0)
                    Acilis20Dak = Source[i];

                if (BarDateTime.Minute % 25 == 0)
                    Acilis25Dak = Source[i];

                if (BarDateTime.Minute % 30 == 0)
                    Acilis30Dak = Source[i];

                if (BarDateTime.Minute % 35 == 0)
                    Acilis35Dak = Source[i];

                if (BarDateTime.Minute % 40 == 0)
                    Acilis40Dak = Source[i];

                if (BarDateTime.Minute % 45 == 0)
                    Acilis45Dak = Source[i];

                if (BarDateTime.Minute % 50 == 0)
                    Acilis50Dak = Source[i];

                if (BarDateTime.Minute % 60 == 0)
                {
                    Acilis60Dak = Source[i];
                    Sayac60Dk++;
                }

                if (Sayac5Dak[0] % (90 / 5) == 0)
                    Acilis90Dak = Source[i];

                if (Sayac5Dak[0] % (100 / 5) == 0)
                    Acilis100Dak = Source[i];

                if (Sayac60Dk % 2 == 0)
                    Acilis120Dak = Source[i];

                if (Sayac5Dak[0] % (150 / 5) == 0)
                    Acilis150Dak = Source[i];

                if (Sayac60Dk % 3 == 0)
                    Acilis180Dak = Source[i];

                if (Sayac5Dak[0] % (200 / 5) == 0)
                    Acilis200Dak = Source[i];

                if (Sayac60Dk % 4 == 0)
                    Acilis240Dak = Source[i];

                if (Sayac60Dk % 5 == 0)
                    Acilis300Dak = Source[i];

                if (Sayac60Dk % 6 == 0)
                    Acilis360Dak = Source[i];

                if (Sayac5Dak[0] % (400 / 5) == 0)
                    Acilis400Dak = Source[i];

                if (Sayac60Dk % 7 == 0)
                    Acilis420Dak = Source[i];

                if (Sayac60Dk % 8 == 0)
                    Acilis480Dak = Source[i];

                if (Sayac5Dak[0] % (500 / 5) == 0)
                    Acilis500Dak = Source[i];

                if (Sayac5Dak[0] % (600 / 5) == 0)
                    Acilis600Dak = Source[i];

                if (IsYeniSaat)
                    AcilisSaat = Source[i];

                if (IsYeniGun)
                    AcilisGun = Source[i];

                if (IsYeniHafta)
                    AcilisHafta = Source[i];

                if (IsYeniAy)
                    AcilisAy = Source[i];
            }

            AcilisListesi5Dak[i] = Acilis5Dak;
            AcilisListesi10Dak[i] = Acilis10Dak;
            AcilisListesi15Dak[i] = Acilis15Dak;
            AcilisListesi20Dak[i] = Acilis20Dak;
            AcilisListesi25Dak[i] = Acilis25Dak;
            AcilisListesi30Dak[i] = Acilis30Dak;
            AcilisListesi35Dak[i] = Acilis35Dak;
            AcilisListesi40Dak[i] = Acilis40Dak;
            AcilisListesi45Dak[i] = Acilis45Dak;
            AcilisListesi50Dak[i] = Acilis50Dak;
            AcilisListesi60Dak[i] = Acilis60Dak;
            AcilisListesi90Dak[i] = Acilis90Dak;
            AcilisListesi100Dak[i] = Acilis100Dak;
            AcilisListesi120Dak[i] = Acilis120Dak;
            AcilisListesi150Dak[i] = Acilis150Dak;
            AcilisListesi180Dak[i] = Acilis180Dak;
            AcilisListesi200Dak[i] = Acilis200Dak;
            AcilisListesi240Dak[i] = Acilis240Dak;
            AcilisListesi300Dak[i] = Acilis300Dak;
            AcilisListesi360Dak[i] = Acilis360Dak;
            AcilisListesi400Dak[i] = Acilis400Dak;
            AcilisListesi420Dak[i] = Acilis420Dak;
            AcilisListesi480Dak[i] = Acilis480Dak;
            AcilisListesi500Dak[i] = Acilis500Dak;
            AcilisListesi600Dak[i] = Acilis600Dak;
            AcilisListesiSaat[i] = AcilisSaat;
            AcilisListesiGun[i] = AcilisGun;
            AcilisListesiHafta[i] = AcilisHafta;
            AcilisListesiAy[i] = AcilisAy;

            if (IsYeniSaat)
            {
                for (int j = 0; j < MaxSayacCount; j++)
                    Sayac5Dak[j] = 0;
            }
        }

        public int XBarListleriniResetle(dynamic Sistem)
        {
            XBarAcilis = 0f;
            XBarYuksek = 0f;
            XBarDusuk = 0f;
            XBarKapanis = 0f;
            XBarHacim = 0f;
            XBarLot = 0f;

            XBarAcilisListesi = Sistem.Liste(BarCount, 0);

            XBarYuksekListesi = Sistem.Liste(BarCount, 0);

            XBarDusukListesi = Sistem.Liste(BarCount, 0);

            XBarKapanisListesi = Sistem.Liste(BarCount, 0);

            XBarHacimListesi = Sistem.Liste(BarCount, 0);

            XBarLotListesi = Sistem.Liste(BarCount, 0);

            return 0;
        }

        public int XBarListleriniGuncelle(dynamic Sistem, int BarIndex, string xBarTimeStr)
        {
            return XBarListleriniGuncelle(Sistem, BarIndex, TimeUtils.GetTime(xBarTimeStr));
        }

        public int XBarListleriniGuncelle(dynamic Sistem, int BarIndex, DateTime xBarTime)
        {
            int i = BarIndex;

            var IsYeniXBar = IsXBar(Sistem, BarIndex, xBarTime);
            if (IsYeniXBar)
            {
                XBarIndeks = BarIndex;
                XBarAcilis = Open[BarIndex];
                XBarYuksek = High[BarIndex];
                XBarDusuk = Low[BarIndex];
                XBarKapanis = Close[BarIndex];
                XBarHacim = Volume[BarIndex];
                XBarLot = Lot[BarIndex];
            }

            XBarAcilisListesi[i] = XBarAcilis;
            XBarYuksekListesi[i] = XBarYuksek;
            XBarDusukListesi[i] = XBarDusuk;
            XBarKapanisListesi[i] = XBarKapanis;
            XBarHacimListesi[i] = XBarHacim;
            XBarLotListesi[i] = XBarLot;

            return 0;
        }

        public bool IsXBar(dynamic Sistem, int BarIndex, DateTime xBarTime)
        {
            var BarDateTime = GetBarDateTime(Sistem, BarIndex);
            bool isXBar = false;

            if (TimeSpan.Compare(BarDateTime.TimeOfDay, xBarTime.TimeOfDay) == 0)
            {
                isXBar = true;
            }

            return isXBar;
        }

        public bool IsXBar(dynamic Sistem, int BarIndex, string xBarTimeStr)
        {
            return IsXBar(Sistem, BarIndex, TimeUtils.GetTime(xBarTimeStr));
        }

        public void XBarIndeksListesiniResetle(dynamic Sistem)
        {
            XBarIndeksList.Clear();
        }

        public void XBarIndeksListesiniGuncelle(dynamic Sistem, int BarIndex, string xBarTimeStr)
        {
            XBarIndeksListesiniGuncelle(Sistem, BarIndex, TimeUtils.GetTime(xBarTimeStr));
        }

        public void XBarIndeksListesiniGuncelle(dynamic Sistem, int BarIndex, DateTime xBarTime)
        {
            var IsYeniXBar = IsXBar(Sistem, BarIndex, xBarTime);
            if (IsYeniXBar)
                XBarIndeksList.Add(BarIndex);
        }

        public void BarYuzdeListesiniResetle(dynamic Sistem)
        {
            BarYuzdeListesiOC = Sistem.Liste(BarCount, 0);
            BarYuzdeListesiHL = Sistem.Liste(BarCount, 0);
        }

        public int BarYuzdeListesiniGuncelle(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;

            var fark = (High[i]-Low[i]);
            fark = Close[i] >= Open[i] ? fark : -fark;
            BarYuzdeListesiHL[i] = 100.0f * fark / Low[i];

            fark = (Close[i]-Open[i]);
            if (fark > 0)
                BarYuzdeListesiOC[i] = 100.0f * fark / Open[i];
            else if (fark < 0)
                BarYuzdeListesiOC[i] = 100.0f * fark / Close[i];
            else
                BarYuzdeListesiOC[i] = 0.0f;

            return 0;
        }
    }
