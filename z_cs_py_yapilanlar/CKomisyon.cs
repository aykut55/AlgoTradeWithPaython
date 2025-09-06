public class CKomisyon
{
    CTrader Trader = null;

    ~CKomisyon()
    {

    }

    public CKomisyon()
    {

    }

    public CKomisyon Initialize(dynamic Sistem, CTrader Trader)
    {
        this.Trader = Trader;

        return this;
    }

    public CKomisyon Reset(dynamic Sistem)
    {
        return this;
    }

    public int Hesapla(dynamic Sistem, int BarIndex)
    {
        int result = 0;

        int i = BarIndex;

        Trader.Status.KomisyonFiyat = Trader.Lists.KomisyonIslemSayisiList[i] * Convert.ToSingle(Trader.Status.KomisyonCarpan) * Trader.Status.KomisyonVarlikAdedSayisi;
        Trader.Lists.KomisyonFiyatList[i] = Trader.Status.KomisyonFiyat;

        return result;
    }
}