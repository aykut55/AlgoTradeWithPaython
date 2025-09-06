public class CKarZarar
{
    public dynamic V { get; set; }
    CTrader Trader = null;

    ~CKarZarar()
    {

    }

    public CKarZarar()
    {

    }

    public CKarZarar Initialize(dynamic Sistem, CTrader Trader)
    {
        this.Trader = Trader;
        this.V = Trader.V;

        return this;
    }

    public CKarZarar Reset(dynamic Sistem)
    {
        return this;
    }

    public int AnlikKarZararHesapla(dynamic Sistem, int BarIndex, string Type = "C")
    {
        int result = 0;

        int i = BarIndex;

        float AnlikFiyat = V[i].Close;

        bool AnlikKarZararHesaplaEnabled = Trader.Flags.AnlikKarZararHesaplaEnabled;

        string SonYon = Trader.Signals.SonYon;

        float SonFiyat = Trader.Signals.SonFiyat;

        int VarlikAdedSayisi = Trader.VarlikManager.VarlikAdedSayisi;

        if (AnlikKarZararHesaplaEnabled)
        {
            if (Type != "C")
            {
                if (Type == "O") AnlikFiyat = V[i].Open;
                else if (Type == "H") AnlikFiyat = V[i].High;
                else if (Type == "L") AnlikFiyat = V[i].Low;
            }

            if (SonYon == "A")
            {
                Trader.Status.KarZararPuan = (AnlikFiyat - SonFiyat);
                Trader.Status.KarZararFiyat = Trader.Status.KarZararPuan* VarlikAdedSayisi;

                Trader.Lists.KarZararPuanList[i] = Trader.Status.KarZararPuan;
                Trader.Lists.KarZararFiyatList[i] = Trader.Status.KarZararFiyat;

                Trader.Status.KarZararFiyatYuzde = 100.0f * Trader.Status.KarZararPuan / SonFiyat;
                Trader.Lists.KarZararFiyatYuzdeList[i] = Trader.Status.KarZararFiyatYuzde;
            }
            else if (SonYon == "S")
            {
                Trader.Status.KarZararPuan = (SonFiyat - AnlikFiyat);
                Trader.Status.KarZararFiyat = Trader.Status.KarZararPuan* VarlikAdedSayisi;

                Trader.Lists.KarZararPuanList[i] = Trader.Status.KarZararPuan;
                Trader.Lists.KarZararFiyatList[i] = Trader.Status.KarZararFiyat;

                Trader.Status.KarZararFiyatYuzde = 100.0f * Trader.Status.KarZararPuan / SonFiyat;
                Trader.Lists.KarZararFiyatYuzdeList[i] = Trader.Status.KarZararFiyatYuzde;
            }
            else if (SonYon == "F")
            {

            }

            if (Trader.Status.KarZararPuan > 0)
            {
                Trader.Status.KardaBarSayisi++;
                Trader.Status.ZarardaBarSayisi--;
            }
            else if (Trader.Status.KarZararPuan == 0)
            {
                Trader.Status.KardaBarSayisi = 0;
                Trader.Status.ZarardaBarSayisi = 0;
            }
            else
            {
                Trader.Status.KardaBarSayisi--;
                Trader.Status.ZarardaBarSayisi++;
            }
        }

        return result;
    }

}