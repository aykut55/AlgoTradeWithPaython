public class CFlags
{
    public bool BakiyeGuncelle = false;
    public bool KomisyonGuncelle = false;
    public bool DonguSonuIstatistikGuncelle = false;
    public bool KomisyonuDahilEt = true;
    public bool KaymayiDahilEt = false;

    public bool AnlikKarZararHesaplaEnabled = false;
    public bool KarAlYuzdeHesaplaEnabled = false;
    public bool IzleyenStopYuzdeHesaplaEnabled = false;
    public bool ZararKesYuzdeHesaplaEnabled = false;
    public bool KarAlSeviyeHesaplaEnabled = false;
    public bool ZararKesSeviyeHesaplaEnabled = false;

    public bool AGerceklesti = false;
    public bool SGerceklesti = false;
    public bool FGerceklesti = false;
    public bool PGerceklesti = false;

    public bool IdealGetiriHesapla = false;
    public bool IdealGetiriHesaplandi = false;

    ~CFlags()
    {

    }

    public CFlags()
    {

    }

    public CFlags Initialize(dynamic Sistem)
    {
        return this;
    }

    public CFlags Reset(dynamic Sistem)
    {
        BakiyeGuncelle = false;
        KomisyonGuncelle = false;
        DonguSonuIstatistikGuncelle = false;
        KomisyonuDahilEt = true;
        KaymayiDahilEt = false;

        AnlikKarZararHesaplaEnabled = false;
        KarAlYuzdeHesaplaEnabled = false;
        IzleyenStopYuzdeHesaplaEnabled = false;
        ZararKesYuzdeHesaplaEnabled = false;
        KarAlSeviyeHesaplaEnabled = false;
        ZararKesSeviyeHesaplaEnabled = false;

        AGerceklesti = false;
        SGerceklesti = false;
        FGerceklesti = false;
        PGerceklesti = false;

        IdealGetiriHesapla = true;
        IdealGetiriHesaplandi = false;

        return this;
    }
}