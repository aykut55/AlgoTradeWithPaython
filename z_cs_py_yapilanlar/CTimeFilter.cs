public class CTimeFilter
{
    CTrader Trader = null;

    ~CTimeFilter()
    {

    }

    public CTimeFilter()
    {

    }

    public CTimeFilter Initialize(dynamic Sistem, CTrader Trader)
    {
        this.Trader = Trader;

        return this;
    }

    public CTimeFilter Reset(dynamic Sistem)
    {
        return this;
    }
}