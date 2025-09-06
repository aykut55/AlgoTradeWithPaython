public class CSignals
{
    public bool Al = false;
    public bool Sat = false;
    public bool FlatOl = false;
    public bool PasGec = false;
    public bool KarAl = false;
    public bool ZararKes = false;

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

    // Asagidaki degiskenler dogrudan kullanici tarafindan yazilan sistem kodu içerisinde
    // setlenecek-resetlenecek. CTrader sinifinda herhangi bir işleme tabi tutulmayacak.
    // Degisken tanimlama zahmetinden kurtulmak icin buraya koyuldu
    public bool KarAlEnabled = false;
    public bool ZararKesEnabled = false;
    public bool KarAlindi = false;
    public bool ZararKesildi = false;
    public bool FlatOlundu = false;
    public bool PozAcilabilir = false;
    public bool PozAcildi = false;
    public bool PozKapatilabilir = false;
    public bool PozKapatildi = false;
    public bool PozAcilabilirAlis = false;
    public bool PozAcilabilirSatis = false;
    public bool PozAcildiAlis = false;
    public bool PozAcildiSatis = false;
    public bool GunSonuPozKapatEnabled = false;
    public bool GunSonuPozKapatildi = false;
    public bool TimeFilteringEnabled = false;

    ~CSignals()
    {

    }

    public CSignals()
    {

    }

    public CSignals Initialize(dynamic Sistem)
    {
        return this;
    }

    public CSignals Reset(dynamic Sistem)
    {
        Al = false;
        Sat = false;
        FlatOl = false;
        PasGec = false;
        KarAl = false;
        ZararKes = false;

        Sinyal = "";
        SonYon = "F";   // Komisyon hesaplarken F olmasi onem arz etti
        PrevYon = "F";  // Komisyon hesaplarken F olmasi onem arz etti

        SonFiyat = 0f;
        SonAFiyat = 0f;
        SonSFiyat = 0f;
        SonFFiyat = 0f;
        SonPFiyat = 0f;

        PrevFiyat = 0f;
        PrevAFiyat = 0f;
        PrevSFiyat = 0f;
        PrevFFiyat = 0f;
        PrevPFiyat = 0f;

        SonBarNo = 0;
        SonABarNo = 0;
        SonSBarNo = 0;
        SonFBarNo = 0;
        SonPBarNo = 0;

        PrevBarNo = 0;
        PrevABarNo = 0;
        PrevSBarNo = 0;
        PrevFBarNo = 0;
        PrevPBarNo = 0;

        EmirKomut = 0f;
        EmirStatus = 0f;

        return this;
    }
}