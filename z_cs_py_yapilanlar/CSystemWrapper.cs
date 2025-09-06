public class CSystemWrapper : CBase
{
    string GrafikSembol = "";
    string GrafikPeriyot = "";
    string SistemAdi = "";

    public dynamic myVarlik         { get; set; }
    public dynamic myTrader         { get; set; }
    public dynamic myUtils          { get; set; }
    public dynamic myTimeUtils      { get; set; }
    public dynamic myBarUtils       { get; set; }
    public dynamic myFileUtils      { get; set; }
    public dynamic myExcelUtils     { get; set; }
    public dynamic mySharedMemory   { get; set; }
    public dynamic myConfig         { get; set; }
    public dynamic myIndicators     { get; set; }

    public int HisseSayisi = 0;
    public int KontratSayisi = 10;
    public float KomisyonCarpan = 0.0f;
    public int VarlikAdedCarpani = 1;

    public string InputsDir = "";
    public string OutputsDir = "";
    public string ParamsInputFileName = "";
    public string IstatistiklerOutputFileName = "";
    public string IstatistiklerOptOutputFileName = "";

    public bool bUseParamsFromInputFile = false;
    public bool bOptEnabled = false;
    public bool bIdealGetiriHesapla = true;
    public bool bIstatistikleriHesapla = true;
    public bool bIstatistikleriEkranaYaz = true;
    public bool bGetiriIstatistikleriEkranaYaz = true;
    public bool bIstatistikleriDosyayaYaz = true;
    public bool bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz = true;
    public bool bOptimizasyonIstatistikleriniDosyayaYaz = true;
    public bool bSinyalleriEkranaCiz = true;
    public int CurrentRunIndex = 0;
    public int TotalRunCount = 1;

    public bool Al = false;
    public bool Sat = false;
    public bool FlatOl = false;
    public bool PasGec = false;
    public bool KarAl = false;
    public bool ZararKes = false;

    public int InputParamsCount = 50;
    public string [] InputParams = new string[50];

    ~CSystemWrapper()
    {

    }

    public CSystemWrapper()
    {

    }

    public CSystemWrapper CreateModules(dynamic Sistem, dynamic Lib)
    {
        myVarlik       = Lib.GetVarlik(Sistem);
        myTrader       = Lib.GetTrader(Sistem);
        myUtils        = Lib.GetUtils(Sistem);
        myTimeUtils    = Lib.GetTimeUtils(Sistem);
        myBarUtils     = Lib.GetBarUtils(Sistem);
        myFileUtils    = Lib.GetFileUtils(Sistem);
        myExcelUtils   = Lib.GetExcelFileHandler(Sistem);
        mySharedMemory = Lib.GetSharedMemory(Sistem);
        myConfig       = Lib.GetConfig(Sistem);
        myIndicators   = Lib.GetIndicatorManager(Sistem);
        return this;
    }

    public CSystemWrapper Initialize(dynamic Sistem, dynamic V, dynamic Open, dynamic High, dynamic Low, dynamic Close, dynamic Volume, dynamic Lot)
    {
        GrafikSembol = Sistem.Sembol;
        GrafikPeriyot = Sistem.Periyot;
        SistemAdi = Sistem.Name;

        SetData(Sistem, V, Open, High, Low, Close, Volume, Lot);

        myVarlik.Initialize(Sistem);
        myTrader.Initialize(Sistem, V, Open, High, Low, Close, Volume, Lot, myVarlik);
        myUtils.Initialize(Sistem);
        myTimeUtils.Initialize(Sistem, V, Open, High, Low, Close, Volume, Lot);
        myBarUtils.Initialize(Sistem, V, Open, High, Low, Close, Volume, Lot);
        myIndicators.Initialize(Sistem, V, Open, High, Low, Close, Volume, Lot);

        return this;
    }

    public CSystemWrapper Reset(dynamic Sistem)
    {
        myVarlik.Reset(Sistem);
        myTrader.Reset(Sistem);
        myUtils.Reset(Sistem);
        myTimeUtils.Reset(Sistem);
        myBarUtils.Reset(Sistem);
        myIndicators.Reset(Sistem);

        for (int i = 0; i < InputParamsCount; i++)
            InputParams[i] = "";

        return this;
    }

    public CVarlikManager GetVarlik(dynamic Sistem)
    {
        return myVarlik;
    }

    public CTrader GetTrader(dynamic Sistem)
    {
        return myTrader;
    }

    public CUtils GetUtils(dynamic Sistem)
    {
        return myUtils;
    }

    public CTimeUtils GetTimeUtils(dynamic Sistem)
    {
        return myTimeUtils;
    }

    public CBarUtils GetBarUtils(dynamic Sistem)
    {
        return myBarUtils;
    }

    public CFileUtils GetFileUtils(dynamic Sistem)
    {
        return myFileUtils;
    }

    public CExcelFileHandler GetExcelFileHandler(dynamic Sistem)
    {
        return myExcelUtils;
    }

    public CSharedMemory GetSharedMemory(dynamic Sistem)
    {
        return mySharedMemory;
    }

    public CConfigManager GetConfig(dynamic Sistem)
    {
        return myConfig;
    }

    public CIndicatorManager GetIndicatorManager(dynamic Sistem)
    {
        return myIndicators;
    }

    public CSystemWrapper InitializeParamsWithDefaults(dynamic Sistem)
    {
        HisseSayisi = 0;
        KontratSayisi = 10;
        KomisyonCarpan = 0.0f;
        VarlikAdedCarpani = 1;

        InputsDir = "Aykut/Exports/";
        OutputsDir = "Aykut/Exports/";
        ParamsInputFileName = InputsDir + SistemAdi + "_params.txt";
        IstatistiklerOutputFileName = OutputsDir + "Istatistikler.csv";
        IstatistiklerOptOutputFileName = OutputsDir + "IstatistiklerOpt.csv";

        bUseParamsFromInputFile = false;
        bOptEnabled = false;
        bIdealGetiriHesapla = true;
        bIstatistikleriHesapla = true;
        bIstatistikleriEkranaYaz = true;
        bGetiriIstatistikleriEkranaYaz = true;
        bIstatistikleriDosyayaYaz = true;
        bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz = false;
        bOptimizasyonIstatistikleriniDosyayaYaz = false;
        bSinyalleriEkranaCiz = true;

        CurrentRunIndex = 0;
        TotalRunCount = 1;

        myVarlik.SetKontratParamsFxOnsAltinMicro(Sistem, KontratSayisi = 1, VarlikAdedCarpani = 1).SetKomisyonParams(Sistem, KomisyonCarpan = 0.0f);

        myTrader.Signals.KarAlEnabled = false;
        myTrader.Signals.ZararKesEnabled = false;
        myTrader.Signals.KarAlindi = false;
        myTrader.Signals.ZararKesildi = false;
        myTrader.Signals.FlatOlundu = false;
        myTrader.Signals.PozAcilabilir = false;
        myTrader.Signals.PozAcildi = false;
        myTrader.Signals.PozKapatilabilir = false;
        myTrader.Signals.PozKapatildi = false;
        myTrader.Signals.PozAcilabilirAlis = false;
        myTrader.Signals.PozAcilabilirSatis = false;
        myTrader.Signals.PozAcildiAlis = false;
        myTrader.Signals.PozAcildiSatis = false;
        myTrader.Signals.GunSonuPozKapatEnabled = false;
        myTrader.Signals.GunSonuPozKapatildi = false;
        myTrader.Signals.TimeFilteringEnabled = false;

        return this;
    }

    public CSystemWrapper SetParamsForSingleRun(dynamic Sistem, bool IdealGetiriHesapla = true, bool IstatistikleriHesapla = true,
        bool IstatistikleriEkranaYaz = true, bool GetiriIstatistikleriEkranaYaz = true, bool IstatistikleriDosyayaYaz = true, bool SinyalleriEkranaCiz = true)
    {
        this.bIdealGetiriHesapla = IdealGetiriHesapla;
        this.bIstatistikleriHesapla = IstatistikleriHesapla;
        this.bIstatistikleriEkranaYaz = IstatistikleriEkranaYaz;
        this.bGetiriIstatistikleriEkranaYaz = GetiriIstatistikleriEkranaYaz;
        this.bIstatistikleriDosyayaYaz = IstatistikleriDosyayaYaz;
        this.bSinyalleriEkranaCiz = SinyalleriEkranaCiz;

        return this;
    }

    public void Start(dynamic Sistem)
    {
        myTimeUtils.StartTimer(Sistem);

        myTrader.Start(Sistem);
    }

    public void EmirleriResetle(dynamic Sistem, int BarIndex)
    {
        this.Al = false;
        this.Sat = false;
        this.FlatOl = false;
        this.PasGec = false;
        this.KarAl = false;
        this.ZararKes = false;
    }

    public void EmirOncesiDonguFoksiyonlariniCalistir(dynamic Sistem, int BarIndex)
    {
        int i = BarIndex;

        myTrader.DonguBasiDegiskenleriResetle(Sistem, i);

        myTrader.DonguBasiDegiskenleriGuncelle(Sistem, i);

        if (i < 1) return;

        myTrader.AnlikKarZararHesapla(Sistem, i);

        myTrader.EmirleriResetle(Sistem, i);

        bool isYeniGun = (V[i].Date.Day != V[i - 1].Date.Day); if (isYeniGun) Sistem.DikeyCizgiEkle(i, Color.DimGray, 2, 2);

        bool isYeniSaat = (V[i].Date.Hour != V[i - 1].Date.Hour);

        if (myTrader.Signals.GunSonuPozKapatildi)
        {
            myTrader.Signals.GunSonuPozKapatildi = false;
        }

        if (myTrader.Signals.KarAlindi || myTrader.Signals.ZararKesildi || myTrader.Signals.FlatOlundu)
        {
            myTrader.Signals.KarAlindi = false;
            myTrader.Signals.ZararKesildi = false;
            myTrader.Signals.FlatOlundu = false;
            myTrader.Signals.PozAcilabilir = false;
        }

        if (myTrader.Signals.PozAcilabilir == false)
        {
            myTrader.Signals.PozAcilabilir = true;
            myTrader.Signals.PozAcildi = false;
        }
    }

    public void EmirleriSetle(dynamic Sistem, int BarIndex, bool Al, bool Sat, bool FlatOl = false, bool PasGec = false, bool KarAl = false, bool ZararKes = false)
    {
        int i = BarIndex;

        this.Al = Al;
        this.Sat = Sat;
        this.FlatOl = FlatOl;
        this.PasGec = PasGec;
        this.KarAl = KarAl;
        this.ZararKes = ZararKes;
    }

    public void EmirSonrasiDonguFoksiyonlariniCalistir(dynamic Sistem, int BarIndex)
    {
        int i = BarIndex;

        myTrader.EmirleriSetle(Sistem, i, Al, Sat, FlatOl, PasGec, KarAl, ZararKes);

        myTrader.Signals.GunSonuPozKapatildi = myTrader.GunSonuPozKapat(Sistem, i, myTrader.Signals.GunSonuPozKapatEnabled);

        myTrader.EmirleriUygula(Sistem, i);

        if (myTrader.Signals.KarAlindi == false && myTrader.Signals.KarAl) { myTrader.Signals.KarAlindi = true; }

        if (myTrader.Signals.ZararKesildi == false && myTrader.Signals.ZararKes) { myTrader.Signals.ZararKesildi = true; }

        if (myTrader.Signals.FlatOlundu == false && myTrader.Signals.FlatOl) { myTrader.Signals.FlatOlundu = true; }

        myTrader.SistemYonListesiniGuncelle(Sistem, i);

        myTrader.SistemSeviyeListesiniGuncelle(Sistem, i);

        myTrader.SinyalListesiniGuncelle(Sistem, i);

        myTrader.IslemListesiniGuncelle(Sistem, i);

        myTrader.KomisyonListesiniGuncelle(Sistem, i);

        myTrader.BakiyeListesiniGuncelle(Sistem, i);

        myTrader.DonguSonuDegiskenleriSetle(Sistem, i);

    }

    public void Stop(dynamic Sistem)
    {
        myTrader.Stop(Sistem);

        myTimeUtils.StopTimer(Sistem);
    }

    public void HesaplamalariYap(dynamic Sistem)
    {
        if (bIdealGetiriHesapla)
            myTrader.IdealGetiriHesapla(Sistem);

        if (bIstatistikleriHesapla)
            myTrader.IstatistikleriHesapla(Sistem);
    }

    public void SonuclariEkrandaGoster(dynamic Sistem)
    {
        if (bIstatistikleriEkranaYaz)
            myTrader.IstatistikleriEkranaYaz(Sistem, 1);

        if (bGetiriIstatistikleriEkranaYaz)
            myTrader.GetiriIstatistikleriEkranaYaz(Sistem, 2);
    }

    public void SonuclariDosyayaYaz(dynamic Sistem)
    {
        if (bIstatistikleriDosyayaYaz)
            myTrader.IstatistikleriDosyayaYaz(Sistem, IstatistiklerOutputFileName);
    }

    public int SinyalleriEkranaCiz(dynamic Sistem, int k)
    {
        int k_ = 0;

        if (bSinyalleriEkranaCiz)
            k_ = myTrader.SinyalleriEkranaCiz(Sistem, k);

        return k_;
    }

    public void AciklamalariTemizle(dynamic Sistem, int k)
    {
        for (int i = k; i < 50; i++)
            Sistem.Cizgiler[i].Aciklama = "---";
    }
    
    public void RastgeleRenkAta(dynamic Sistem, int k)
    {
        var randomNumber = new Random();
        
        for (int i = k; i < 50; i++) {
            int r = randomNumber.Next(255);
            int g = randomNumber.Next(255);
            int b = randomNumber.Next(255);
            Sistem.Cizgiler[i].Renk = Color.FromArgb(255, r, g, b);
        } 
    }    

    public CSystemWrapper SetParamsForOptimizasyon(dynamic Sistem, bool OptEnabled = true, bool IdealGetiriHesapla = true, bool IstatistikleriHesapla = true,
        bool IstatistikleriEkranaYaz = false, bool GetiriIstatistikleriEkranaYaz = false, bool IstatistikleriDosyayaYaz = true, bool SinyalleriEkranaCiz = false,
        bool OptimizasyonIstatistiklerininBasliklariniDosyayaYaz = true, bool OptimizasyonIstatistikleriniDosyayaYaz = true)
    {
        this.bOptEnabled = OptEnabled;
        this.bIdealGetiriHesapla = IdealGetiriHesapla;
        this.bIstatistikleriHesapla = IstatistikleriHesapla;
        this.bIstatistikleriEkranaYaz = IstatistikleriEkranaYaz;
        this.bGetiriIstatistikleriEkranaYaz = GetiriIstatistikleriEkranaYaz;
        this.bIstatistikleriDosyayaYaz = IstatistikleriDosyayaYaz;
        this.bSinyalleriEkranaCiz = SinyalleriEkranaCiz;

        this.bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz = OptimizasyonIstatistiklerininBasliklariniDosyayaYaz;
        this.bOptimizasyonIstatistikleriniDosyayaYaz = OptimizasyonIstatistikleriniDosyayaYaz;

        return this;
    }

    public void OptimizasyonIstatistiklerininBasliklariniDosyayaYaz(dynamic Sistem)
    {
        if (bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz)
            myTrader.OptimizasyonIstatistiklerininBasliklariniDosyayaYaz(Sistem, IstatistiklerOptOutputFileName);
    }

    public void OptimizasyonIstatistikleriniDosyayaYaz(dynamic Sistem, int CurrentRunIndex, int TotalRunCount)
    {
        if (bOptimizasyonIstatistikleriniDosyayaYaz)
            myTrader.OptimizasyonIstatistikleriniDosyayaYaz(Sistem, IstatistiklerOptOutputFileName, CurrentRunIndex, TotalRunCount);
    }

    public void SetCurrentIndex(dynamic Sistem, int CurrentRunIndex)
    {
        this.CurrentRunIndex = CurrentRunIndex;
    }

    public void SetTotalRunCount(dynamic Sistem, int TotalRunCount)
    {
        this.TotalRunCount = TotalRunCount;
    }

    public void SetInputParams(dynamic Sistem, int Index, string Value)
    {
        InputParams[Index] = Value;
    }

    public CSystemWrapper ReadParamsFromFile(dynamic Sistem, string FileName)
    {
        if (File.Exists(FileName))
        {
            string[] readLines = File.ReadAllLines(FileName);
            this.bUseParamsFromInputFile = Convert.ToInt32(readLines[0]) == 1 ? true : false;

            if (this.bUseParamsFromInputFile)
            {
                this.CurrentRunIndex = Convert.ToInt32(readLines[1]);
                this.TotalRunCount = Convert.ToInt32(readLines[2]);
                this.bOptEnabled = Convert.ToInt32(readLines[3]) == 1 ? true : false;
                this.bIdealGetiriHesapla = Convert.ToInt32(readLines[4]) == 1 ? true : false;
                this.bIstatistikleriHesapla = Convert.ToInt32(readLines[5]) == 1 ? true : false;
                this.bIstatistikleriEkranaYaz = Convert.ToInt32(readLines[6]) == 1 ? true : false;
                this.bGetiriIstatistikleriEkranaYaz = Convert.ToInt32(readLines[7]) == 1 ? true : false;
                this.bIstatistikleriDosyayaYaz = Convert.ToInt32(readLines[8]) == 1 ? true : false;
                this.bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz = Convert.ToInt32(readLines[9]) == 1 ? true : false;
                this.bOptimizasyonIstatistikleriniDosyayaYaz = Convert.ToInt32(readLines[10]) == 1 ? true : false;
                this.bSinyalleriEkranaCiz = Convert.ToInt32(readLines[11]) == 1 ? true : false;
                this.ParamsInputFileName = readLines[12];
                this.IstatistiklerOutputFileName = readLines[13];
                this.IstatistiklerOptOutputFileName = readLines[14];
            }

            if (this.bUseParamsFromInputFile)
            {
                int k = 0;

                InputParams[k++] = readLines[15];
                InputParams[k++] = readLines[16];
                InputParams[k++] = readLines[17];

                for (int i = k; i < InputParamsCount; i++)
                    InputParams[i] = "";
            }
        }

        return this;
    }

    public CSystemWrapper UpdateSistemParametreleri(dynamic Sistem)
    {
        for (int i = 0; i < 20; i++)
            Sistem.Parametreler[i] = InputParams[i];

        return this;
    }

    public CSystemWrapper WriteParamsToFile(dynamic Sistem, string FileName, bool bUseParamsFromInputFile, int CurrentRunIndex, int TotalRunCount,
        bool bOptEnabled, bool bIdealGetiriHesapla, bool bIstatistikleriHesapla, bool bIstatistikleriEkranaYaz,
        bool bGetiriIstatistikleriEkranaYaz, bool bIstatistikleriDosyayaYaz, bool bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz,
        bool bOptimizasyonIstatistikleriniDosyayaYaz, bool bSinyalleriEkranaCiz, string ParamsInputFileName, string IstatistiklerOutputFileName, string IstatistiklerOptOutputFileName )
    {
        string strOne = "1";
        string strZero = "0";
        string str = "";

        var writeLines = new List<string>();

        writeLines.Clear();

        str = bUseParamsFromInputFile ? strOne : strZero;                                  writeLines.Add( str );
        str = CurrentRunIndex.ToString();                                                  writeLines.Add( str );
        str = TotalRunCount.ToString();                                                    writeLines.Add( str );
        str = bOptEnabled ? strOne : strZero;                                              writeLines.Add( str );

        str = bIdealGetiriHesapla ? strOne : strZero;                                      writeLines.Add( str );
        str = bIstatistikleriHesapla ? strOne : strZero;                                   writeLines.Add( str );
        str = bIstatistikleriEkranaYaz ? strOne : strZero;                                 writeLines.Add( str );
        str = bGetiriIstatistikleriEkranaYaz ? strOne : strZero;                           writeLines.Add( str );

        str = bIstatistikleriDosyayaYaz ? strOne : strZero;                                writeLines.Add( str );
        str = bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz ? strOne : strZero;     writeLines.Add( str );
        str = bOptimizasyonIstatistikleriniDosyayaYaz ? strOne : strZero;                  writeLines.Add( str );
        str = bSinyalleriEkranaCiz ? strOne : strZero;                                     writeLines.Add( str );

        str = ParamsInputFileName;                                                         writeLines.Add( str );
        str = IstatistiklerOutputFileName;                                                 writeLines.Add( str );
        str = IstatistiklerOptOutputFileName;                                              writeLines.Add( str );

        File.WriteAllLines(FileName, writeLines, Encoding.UTF8);

        return this;
    }

    public string CreateOptimizasyonLogString(dynamic Sistem, int Counter, int TotalCount, dynamic OptParam1 = null, dynamic OptParam2 = null, dynamic OptParam3 = null, dynamic OptParam4 = null, dynamic OptParam5 = null, dynamic OptParam6 = null)
    {

        Dictionary<string, dynamic> IstatistiklerNew = myTrader.Statistics.IstatistiklerNew;

        string delimiter = ";";

        string str = "";

        List<string> strList = new List<string>();

        strList.Clear();

        str = String.Format("{0} {1} ", Counter, delimiter);                                            strList.Add( str );
        str = String.Format("{0} {1} ", TotalCount, delimiter);                                         strList.Add( str );

        if (OptParam1 != null)
            str = String.Format("{0} {1} ", OptParam1, delimiter);                                      strList.Add( str );

        if (OptParam2 != null)
            str = String.Format("{0} {1} ", OptParam2, delimiter);                                      strList.Add( str );

        if (OptParam3 != null)
            str = String.Format("{0} {1} ", OptParam3, delimiter);                                      strList.Add( str );

        if (OptParam4 != null)
            str = String.Format("{0} {1} ", OptParam4, delimiter);                                      strList.Add( str );

        if (OptParam5 != null)
            str = String.Format("{0} {1} ", OptParam5, delimiter);                                      strList.Add( str );

        if (OptParam6 != null)
            str = String.Format("{0} {1} ", OptParam6, delimiter);                                      strList.Add( str );

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

        str = "";

        for (int i = 0; i < strList.Count; i++)
            str = str + string.Format("  {0} ", strList[i]);

        str = str.Trim();

        return str;
    }
    
    public void IslemZamanFiltresiUygula(dynamic Sistem, int BarIndex, ref  bool Al, ref  bool Sat, ref  bool FlatOl, ref  bool PasGec, ref  bool KarAl, ref  bool ZararKes, int FilterMode = 3)
    {
    	bool useTimeFiltering = myTrader.Signals.TimeFilteringEnabled ? true : false;            
        if (useTimeFiltering)
        {
	        int checkResult = 0;        
	        bool isTradeEnabled = false;
	        bool isPozKapatEnabled = false;
	                
	        myTrader.IslemZamanFiltresiUygula(Sistem, BarIndex, FilterMode, ref isTradeEnabled, ref isPozKapatEnabled, ref checkResult);
	        
	        if (!isTradeEnabled) Al = false;
	        if (!isTradeEnabled) Sat = false;
	        if (isPozKapatEnabled) FlatOl = true; 
        }
    }
    
    public void IslemZamanFiltresiUygula(dynamic Sistem, int BarIndex, int FilterMode = 3)
    {
    	bool useTimeFiltering = myTrader.Signals.TimeFilteringEnabled ? true : false;            
        if (useTimeFiltering)
        {
	        int checkResult = 0;        
	        bool isTradeEnabled = false;
	        bool isPozKapatEnabled = false;
	                
	        myTrader.IslemZamanFiltresiUygula(Sistem, BarIndex, FilterMode, ref isTradeEnabled, ref isPozKapatEnabled, ref checkResult);
	        
	        if (!isTradeEnabled) Al = false;
	        if (!isTradeEnabled) Sat = false;
	        if (isPozKapatEnabled) FlatOl = true; 
        }       
    }

    public void IslemZamanFiltresiUygula_Bunu_Kullanma(dynamic Sistem, int BarIndex, int mode1 = 0)
    {
        int i = BarIndex;
        var mySystem = this;
        int checkResult = 0;
        bool isTradeEnabled = false;
        bool isPozKapatEnabled = false;

        var startDateTime = mySystem.GetTrader(Sistem).StartDateTimeStr;
        var stopDateTime  = mySystem.GetTrader(Sistem).StopDateTimeStr;
        var startDate     = mySystem.GetTrader(Sistem).StartDateStr;
        var stopDate      = mySystem.GetTrader(Sistem).StopDateStr;
        var startTime     = mySystem.GetTrader(Sistem).StartTimeStr;
        var stopTime      = mySystem.GetTrader(Sistem).StopTimeStr;
        var nowDateTime   = DateTime.Now.ToString("dd.MM.yyyy HH:mm:ss");
        var nowDate       = DateTime.Now.ToString("dd.MM.yyyy");
        var nowTime       = DateTime.Now.ToString("HH:mm:ss");

        bool useTimeFiltering = mySystem.GetTrader(Sistem).Signals.TimeFilteringEnabled ? true : false;
        int mode = mode1;
        if (useTimeFiltering && 1 == 0)
        {
            if (i == Sistem.BarSayisi-1)
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
                s += "  " + ("FilterMode = " + mode.ToString()).ToString() + Environment.NewLine;
                s += "  " + "CSystemWrapper::IslemZamanFiltresiUygula".ToString() + Environment.NewLine;

                Sistem.Mesaj(s.ToString());
            }

            if (mode == 0)
            {
            }
            else if (mode == 1)
            {
                if (mySystem.GetTimeUtils(Sistem).CheckBarTimeWith(Sistem, i, startTime) >= 0 && mySystem.GetTimeUtils(Sistem).CheckBarTimeWith(Sistem, i, stopTime) < 0)
                {
                    isTradeEnabled = true;
                }
                else if (mySystem.GetTimeUtils(Sistem).CheckBarTimeWith(Sistem, i, startTime) < 0)
                {
                    if (!mySystem.GetTrader(Sistem).IsSonYonF(Sistem))
                        isPozKapatEnabled = true;
                }
                else if (mySystem.GetTimeUtils(Sistem).CheckBarTimeWith(Sistem, i, stopTime) >= 0)
                {
                    if (!mySystem.GetTrader(Sistem).IsSonYonF(Sistem))
                        isPozKapatEnabled = true;
                }

                if (!isTradeEnabled) Al = false;
                if (!isTradeEnabled) Sat = false;
                if (isPozKapatEnabled) FlatOl = true;
            }
            else if (mode == 2)
            {
                if (mySystem.GetTimeUtils(Sistem).CheckBarDateWith(Sistem, i, startDate) >= 0 && mySystem.GetTimeUtils(Sistem).CheckBarDateWith(Sistem, i, stopDate) < 0)
                {
                    isTradeEnabled = true;
                }
                else if (mySystem.GetTimeUtils(Sistem).CheckBarDateWith(Sistem, i, startDate) < 0)
                {
                    if (!mySystem.GetTrader(Sistem).IsSonYonF(Sistem))
                        isPozKapatEnabled = true;
                }
                else if (mySystem.GetTimeUtils(Sistem).CheckBarDateWith(Sistem, i, stopDate) >= 0)
                {
                    if (!mySystem.GetTrader(Sistem).IsSonYonF(Sistem))
                        isPozKapatEnabled = true;
                }

                if (!isTradeEnabled) Al = false;
                if (!isTradeEnabled) Sat = false;
                if (isPozKapatEnabled) FlatOl = true;
            }
            else if (mode == 3)
            {
                if (mySystem.GetTimeUtils(Sistem).CheckBarDateTimeWith(Sistem, i, startDateTime) >= 0 && mySystem.GetTimeUtils(Sistem).CheckBarDateTimeWith(Sistem, i, stopDateTime) < 0)
                {
                    isTradeEnabled = true;
                }
                else if (mySystem.GetTimeUtils(Sistem).CheckBarDateTimeWith(Sistem, i, startDateTime) < 0)
                {
                    if (!mySystem.GetTrader(Sistem).IsSonYonF(Sistem))
                        isPozKapatEnabled = true;
                }
                else if (mySystem.GetTimeUtils(Sistem).CheckBarDateTimeWith(Sistem, i, stopDateTime) >= 0)
                {
                    if (!mySystem.GetTrader(Sistem).IsSonYonF(Sistem))
                        isPozKapatEnabled = true;
                }

                if (!isTradeEnabled) Al = false;
                if (!isTradeEnabled) Sat = false;
                if (isPozKapatEnabled) FlatOl = true;
            }
            else if (mode == 4)
            {
                if (mySystem.GetTimeUtils(Sistem).CheckBarTimeWith(Sistem, i, startTime) >= 0)
                {
                    isTradeEnabled = true;
                }
                else if (mySystem.GetTimeUtils(Sistem).CheckBarTimeWith(Sistem, i, startTime) < 0)
                {
                    if (!mySystem.GetTrader(Sistem).IsSonYonF(Sistem))
                        isPozKapatEnabled = true;
                }

                if (!isTradeEnabled) Al = false;
                if (!isTradeEnabled) Sat = false;
                if (isPozKapatEnabled) FlatOl = true;
            }
            else if (mode == 5)
            {
                if (mySystem.GetTimeUtils(Sistem).CheckBarDateWith(Sistem, i, startDate) >= 0)
                {
                    isTradeEnabled = true;
                }
                else if (mySystem.GetTimeUtils(Sistem).CheckBarDateWith(Sistem, i, startDate) < 0)
                {
                    if (!mySystem.GetTrader(Sistem).IsSonYonF(Sistem))
                        isPozKapatEnabled = true;
                }

                if (!isTradeEnabled) Al = false;
                if (!isTradeEnabled) Sat = false;
                if (isPozKapatEnabled) FlatOl = true;
            }
            else if (mode == 6)
            {
                if (mySystem.GetTimeUtils(Sistem).CheckBarDateTimeWith(Sistem, i, startDateTime) >= 0)
                {
                    isTradeEnabled = true;
                }
                else if (mySystem.GetTimeUtils(Sistem).CheckBarDateTimeWith(Sistem, i, startDateTime) < 0)
                {
                    if (!mySystem.GetTrader(Sistem).IsSonYonF(Sistem))
                        isPozKapatEnabled = true;
                }

                if (!isTradeEnabled) Al = false;
                if (!isTradeEnabled) Sat = false;
                if (isPozKapatEnabled) FlatOl = true;
            }
        }
    }
}