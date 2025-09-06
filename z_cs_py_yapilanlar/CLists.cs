public class CLists
{
    public List<string> YonList;
    public List<float> SeviyeList;
    public List<float> SinyalList;
    public List<float> KarZararPuanList;
    public List<float> KarZararFiyatList;
    public List<float> KarZararFiyatYuzdeList;
    public List<float> KarAlList;
    public List<float> IzleyenStopList;

    public List<float> IslemSayisiList;
    public List<float> AlisSayisiList;
    public List<float> SatisSayisiList;
    public List<float> FlatSayisiList;
    public List<float> PassSayisiList;

    public List<float> KontratSayisiList;
    public List<float> VarlikAdedSayisiList;
    public List<float> KomisyonVarlikAdedSayisiList;
    public List<float> KomisyonIslemSayisiList;
    public List<float> KomisyonFiyatList;
    public List<float> KardaBarSayisiList;
    public List<float> ZarardaBarSayisiList;

    public List<float> BarIndexList;

    public List<float> ZeroLevel { get; set; }

    public List<float> BakiyePuanList;
    public List<float> BakiyeFiyatList;
    public List<float> GetiriPuanList;
    public List<float> GetiriFiyatList;
    public List<float> GetiriPuanYuzdeList;
    public List<float> GetiriFiyatYuzdeList;

    public List<float> BakiyePuanNetList;
    public List<float> BakiyeFiyatNetList;
    public List<float> GetiriPuanNetList;
    public List<float> GetiriFiyatNetList;
    public List<float> GetiriPuanYuzdeNetList;
    public List<float> GetiriFiyatYuzdeNetList;

    public List<float> GetiriKz;
    public List<float> GetiriKzNet;

    public List<float> GetiriKzSistem;
    public List<float> GetiriKzNetSistem;

    public List<float> EmirKomutList { get; set; }
    public List<float> EmirStatusList { get; set; }

    ~CLists()
    {

    }

    public CLists()
    {

    }

    public CLists Initialize(dynamic Sistem)
    {
        return this;
    }

    public CLists Reset(dynamic Sistem)
    {
        return this;
    }

    public int CreateLists(dynamic Sistem, int BarCount)
    {
        int result = 0;

        YonList = new List<string>(new string[BarCount]);               // boş liste oluştur
        SeviyeList = Sistem.Liste(BarCount, 0);                         // boş liste oluştur
        SinyalList = Sistem.Liste(BarCount, 0);                         // boş liste oluştur
        KarZararPuanList = Sistem.Liste(BarCount, 0);                   // boş liste oluştur
        KarZararFiyatList = Sistem.Liste(BarCount, 0);                  // boş liste oluştur
        KarZararFiyatYuzdeList = Sistem.Liste(BarCount, 0);             // boş liste oluştur
        KarAlList = Sistem.Liste(BarCount, 0);                          // boş liste oluştur
        IzleyenStopList = Sistem.Liste(BarCount, 0);                    // boş liste oluştur

        IslemSayisiList = Sistem.Liste(BarCount, 0);                    // boş liste oluştur
        AlisSayisiList = Sistem.Liste(BarCount, 0);                     // boş liste oluştur
        SatisSayisiList = Sistem.Liste(BarCount, 0);                    // boş liste oluştur
        FlatSayisiList = Sistem.Liste(BarCount, 0);                     // boş liste oluştur
        PassSayisiList = Sistem.Liste(BarCount, 0);                     // boş liste oluştur

        KontratSayisiList = Sistem.Liste(BarCount, 0);                  // boş liste oluştur
        VarlikAdedSayisiList = Sistem.Liste(BarCount, 0);               // boş liste oluştur
        KomisyonVarlikAdedSayisiList = Sistem.Liste(BarCount, 0);       // boş liste oluştur
        KomisyonIslemSayisiList = Sistem.Liste(BarCount, 0);            // boş liste oluştur
        KomisyonFiyatList = Sistem.Liste(BarCount, 0);                  // boş liste oluştur
        KardaBarSayisiList = Sistem.Liste(BarCount, 0);                 // boş liste oluştur
        ZarardaBarSayisiList = Sistem.Liste(BarCount, 0);               // boş liste oluştur

        BarIndexList = Sistem.Liste(BarCount, 0);                       // boş liste oluştur

        ZeroLevel = Sistem.Liste(BarCount, 0);                          // boş liste oluştur

        BakiyePuanList = Sistem.Liste(BarCount, 0);                     // boş liste oluştur
        BakiyeFiyatList = Sistem.Liste(BarCount, 0);                    // boş liste oluştur
        GetiriPuanList = Sistem.Liste(BarCount, 0);                     // boş liste oluştur
        GetiriFiyatList = Sistem.Liste(BarCount, 0);                    // boş liste oluştur
        GetiriPuanYuzdeList = Sistem.Liste(BarCount, 0);                // boş liste oluştur
        GetiriFiyatYuzdeList = Sistem.Liste(BarCount, 0);               // boş liste oluştur

        BakiyePuanNetList = Sistem.Liste(BarCount, 0);                  // boş liste oluştur
        BakiyeFiyatNetList = Sistem.Liste(BarCount, 0);                 // boş liste oluştur
        GetiriPuanNetList = Sistem.Liste(BarCount, 0);                  // boş liste oluştur
        GetiriFiyatNetList = Sistem.Liste(BarCount, 0);                 // boş liste
        GetiriPuanYuzdeNetList = Sistem.Liste(BarCount, 0);             // boş liste oluştur
        GetiriFiyatYuzdeNetList = Sistem.Liste(BarCount, 0);            // boş liste oluştur

        GetiriKz = Sistem.Liste(BarCount, 0);                           // boş liste oluştur
        GetiriKzNet = Sistem.Liste(BarCount, 0);                        // boş liste oluştur

        GetiriKzSistem = Sistem.Liste(BarCount, 0);                     // boş liste oluştur
        GetiriKzNetSistem = Sistem.Liste(BarCount, 0);                  // boş liste oluştur

        EmirKomutList = Sistem.Liste(BarCount, 0);                      // boş liste oluştur
        EmirStatusList = Sistem.Liste(BarCount, 0);                     // boş liste oluştur

        return result;
    }
}