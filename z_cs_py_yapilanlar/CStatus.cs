public class CStatus
{
    public int IslemSayisi = 0;

    public int KazandiranIslemSayisi = 0;
    public int KaybettirenIslemSayisi = 0;
    public int NotrIslemSayisi = 0;

    public int KazandiranAlisSayisi = 0;
    public int KaybettirenAlisSayisi = 0;
    public int NotrAlisSayisi = 0;

    public int KazandiranSatisSayisi = 0;
    public int KaybettirenSatisSayisi = 0;
    public int NotrSatisSayisi = 0;

    public int AlisSayisi = 0;
    public int SatisSayisi = 0;
    public int FlatSayisi = 0;
    public int PassSayisi = 0;
    public int KarAlSayisi = 0;
    public int ZararKesSayisi = 0;

    public int AlKomutSayisi = 0;
    public int SatKomutSayisi = 0;
    public int PasGecKomutSayisi = 0;
    public int KarAlKomutSayisi = 0;
    public int ZararKesKomutSayisi = 0;
    public int FlatOlKomutSayisi = 0;

    public int KardaBarSayisi = 0;
    public int ZarardaBarSayisi = 0;

    public float KarZararFiyat = 0f;
    public float KarZararPuan = 0f;
    public float KarZararFiyatYuzde = 0f;

    public int KomisyonIslemSayisi = 0;

    public int KomisyonVarlikAdedSayisi = 1;
    public float KomisyonCarpan = 0f;
    public float KomisyonFiyat = 0f;

    public float KaymaMiktari = 0f;

    public int VarlikAdedSayisi = 0;

    public int VarlikAdedCarpani = 0;

    public int KontratSayisi = 0;

    public int HisseSayisi = 0;

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
    public float GetiriKzSistem = 0f;
    public float GetiriKzNetSistem = 0f;

    public string GetiriFiyatTipi = "TL";

    public float NetKarPuan = 0f;
    public float ToplamKarPuan = 0f;
    public float ToplamZararPuan = 0f;

    public float NetKarFiyat = 0f;
    public float ToplamKarFiyat = 0f;
    public float ToplamZararFiyat = 0f;

    ~CStatus()
    {

    }

    public CStatus()
    {

    }

    public CStatus Initialize(dynamic Sistem)
    {
        return this;
    }

    public CStatus Reset(dynamic Sistem)
    {
        IslemSayisi = 0;

        KazandiranIslemSayisi = 0;
        KaybettirenIslemSayisi = 0;
        NotrIslemSayisi = 0;

        KazandiranAlisSayisi = 0;
        KaybettirenAlisSayisi = 0;
        NotrAlisSayisi = 0;

        KazandiranSatisSayisi = 0;
        KaybettirenSatisSayisi = 0;
        NotrSatisSayisi = 0;

        AlisSayisi = 0;
        SatisSayisi = 0;
        FlatSayisi = 0;
        PassSayisi = 0;
        KarAlSayisi = 0;
        ZararKesSayisi = 0;

        AlKomutSayisi = 0;
        SatKomutSayisi = 0;
        PasGecKomutSayisi = 0;
        KarAlKomutSayisi = 0;
        ZararKesKomutSayisi = 0;
        FlatOlKomutSayisi = 0;

        KardaBarSayisi = 0;
        ZarardaBarSayisi = 0;

        KarZararFiyat = 0f;
        KarZararPuan = 0f;
        KarZararFiyatYuzde = 0f;

        KomisyonIslemSayisi = 0;

        KomisyonVarlikAdedSayisi = 0;
        KomisyonCarpan = 0f;
        KomisyonFiyat = 0f;

        KaymaMiktari = 0f;

        VarlikAdedSayisi = 0;

        VarlikAdedCarpani = 0;

        KontratSayisi = 0;

        HisseSayisi = 0;

        IlkBakiyeFiyat = 0f;
        IlkBakiyePuan = 0f;
        BakiyeFiyat = 0f;
        BakiyePuan = 0f;
        GetiriFiyat = 0f;
        GetiriPuan = 0f;
        GetiriFiyatYuzde = 0f;
        GetiriPuanYuzde = 0f;
        BakiyeFiyatNet = 0f;
        BakiyePuanNet = 0f;
        GetiriFiyatNet = 0f;
        GetiriPuanNet = 0f;
        GetiriFiyatYuzdeNet = 0f;
        GetiriPuanYuzdeNet = 0f;
        GetiriKz = 0f;
        GetiriKzNet = 0f;
        GetiriKzSistem = 0f;
        GetiriKzNetSistem = 0f;

        GetiriFiyatTipi = "TL";

        NetKarPuan = 0f;
        ToplamKarPuan = 0f;
        ToplamZararPuan = 0f;

        NetKarFiyat = 0f;
        ToplamKarFiyat = 0f;
        ToplamZararFiyat = 0f;

        return this;
    }
}