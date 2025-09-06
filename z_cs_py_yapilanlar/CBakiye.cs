public class CBakiye
{
    CTrader Trader = null;

    ~CBakiye()
    {

    }

    public CBakiye()
    {

    }

    public CBakiye Initialize(dynamic Sistem, CTrader Trader)
    {
        this.Trader = Trader;

        return this;
    }

    public CBakiye Reset(dynamic Sistem)
    {
        return this;
    }

    public int Hesapla(dynamic Sistem, int BarIndex)
    {
        int result = 0;

        int i = BarIndex;

        // Bakiye (Puan)
        Trader.Lists.BakiyePuanList[i] = Trader.Status.BakiyePuan + Trader.Lists.KarZararPuanList[i];
        Trader.Lists.GetiriPuanList[i] = Trader.Lists.BakiyePuanList[i] - Trader.Status.IlkBakiyePuan;

        if (Trader.Flags.BakiyeGuncelle)
        {
            Trader.Status.BakiyePuan = Trader.Lists.BakiyePuanList[i];
            Trader.Status.GetiriPuan = Trader.Lists.GetiriPuanList[i];

            if (Trader.Lists.KarZararPuanList[i] >= 0)
                Trader.Status.ToplamKarPuan += Trader.Lists.KarZararPuanList[i];
            else if (Trader.Lists.KarZararPuanList[i] < 0)
                Trader.Status.ToplamZararPuan += Trader.Lists.KarZararPuanList[i];

            Trader.Status.NetKarPuan = Trader.Status.ToplamKarPuan + Trader.Status.ToplamZararPuan;
        }

        // Bakiye (Fiyat)
        Trader.Lists.BakiyeFiyatList[i] = Trader.Status.BakiyeFiyat + Trader.Lists.KarZararFiyatList[i];
        Trader.Lists.GetiriFiyatList[i] = Trader.Lists.BakiyeFiyatList[i] - Trader.Status.IlkBakiyeFiyat;

        if (Trader.Flags.BakiyeGuncelle)
        {
            Trader.Status.BakiyeFiyat = Trader.Lists.BakiyeFiyatList[i];
            Trader.Status.GetiriFiyat = Trader.Lists.GetiriFiyatList[i];

            if (Trader.Lists.KarZararFiyatList[i] >= 0)
                Trader.Status.ToplamKarFiyat += Trader.Lists.KarZararFiyatList[i];
            else if (Trader.Lists.KarZararFiyatList[i] < 0)
                Trader.Status.ToplamZararFiyat += Trader.Lists.KarZararFiyatList[i];

            Trader.Status.NetKarFiyat = Trader.Status.ToplamKarFiyat + Trader.Status.ToplamZararFiyat;
        }

        if (Trader.Status.IlkBakiyePuan != 0f)
            Trader.Lists.GetiriPuanYuzdeList[i] = 100.0f * Trader.Lists.GetiriPuanList[i] / Convert.ToSingle(Trader.Status.IlkBakiyePuan);
        else
            Trader.Lists.GetiriPuanYuzdeList[i] = 0f;

        if (Trader.Status.IlkBakiyeFiyat != 0f)
            Trader.Lists.GetiriFiyatYuzdeList[i] = 100.0f * Trader.Lists.GetiriFiyatList[i] / Convert.ToSingle(Trader.Status.IlkBakiyeFiyat);
        else
            Trader.Lists.GetiriFiyatYuzdeList[i] = 0f;

        if (Trader.Flags.BakiyeGuncelle)
        {
            Trader.Status.GetiriPuanYuzde = Trader.Lists.GetiriPuanYuzdeList[i];
            Trader.Status.GetiriFiyatYuzde = Trader.Lists.GetiriFiyatYuzdeList[i];
        }

        float k = Trader.Status.KomisyonCarpan != 0.0f ? 1.0f : 0.0f;

        Trader.Lists.GetiriFiyatNetList[i] = Trader.Lists.GetiriFiyatList[i] - Trader.Lists.KomisyonFiyatList[i] * k;
        Trader.Lists.BakiyeFiyatNetList[i] = Trader.Lists.GetiriFiyatNetList[i] + Trader.Status.IlkBakiyeFiyat;

        Trader.Lists.GetiriFiyatYuzdeNetList[i] = 0f;
        if (Trader.Status.IlkBakiyeFiyat != 0f)
            Trader.Lists.GetiriFiyatYuzdeNetList[i] = 100.0f * Trader.Lists.GetiriFiyatNetList[i] / Convert.ToSingle(Trader.Status.IlkBakiyeFiyat);

        Trader.Lists.GetiriKz[i] = Trader.Lists.GetiriFiyatList[i] / Trader.Status.VarlikAdedSayisi;
        Trader.Lists.GetiriKzNet[i] = Trader.Lists.GetiriFiyatNetList[i] / Trader.Status.VarlikAdedSayisi;

        if (i == Trader.BarCount-1)
        {
            Trader.Status.BakiyeFiyat         = Trader.Lists.BakiyeFiyatList[Trader.Lists.BakiyeFiyatList.Count-1];
            Trader.Status.GetiriFiyat         = Trader.Lists.GetiriFiyatList[Trader.Lists.GetiriFiyatList.Count-1];
            Trader.Status.GetiriKz            = Trader.Lists.GetiriKz[Trader.Lists.GetiriKz.Count-1];
            Trader.Status.GetiriFiyatYuzde    = Trader.Lists.GetiriFiyatYuzdeList[Trader.Lists.GetiriFiyatYuzdeList.Count-1];

            Trader.Status.BakiyeFiyatNet      = Trader.Lists.BakiyeFiyatNetList[Trader.Lists.BakiyeFiyatNetList.Count-1];
            Trader.Status.GetiriFiyatNet      = Trader.Lists.GetiriFiyatNetList[Trader.Lists.GetiriFiyatNetList.Count-1];
            Trader.Status.GetiriKzNet         = Trader.Lists.GetiriKzNet[Trader.Lists.GetiriKzNet.Count-1];
            Trader.Status.GetiriFiyatYuzdeNet = Trader.Lists.GetiriFiyatYuzdeNetList[Trader.Lists.GetiriFiyatYuzdeNetList.Count-1];

            Trader.Status.BakiyePuan          = Trader.Lists.BakiyePuanList[Trader.Lists.BakiyePuanList.Count-1];
            Trader.Status.GetiriPuan          = Trader.Lists.GetiriPuanList[Trader.Lists.GetiriPuanList.Count-1];
            Trader.Status.BakiyePuanNet       = Trader.Lists.BakiyePuanNetList[Trader.Lists.BakiyePuanNetList.Count-1];
            Trader.Status.GetiriPuanNet       = Trader.Lists.GetiriPuanNetList[Trader.Lists.GetiriPuanNetList.Count-1]; ;

            Trader.Status.GetiriPuanYuzdeNet  = Trader.Lists.GetiriPuanYuzdeNetList[Trader.Lists.GetiriPuanYuzdeNetList.Count-1]; ;
        }

        return result;
    }
}