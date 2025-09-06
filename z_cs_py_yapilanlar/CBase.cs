public class CBase
{
    public int Id { get; set; }

    /*public int SystemId { get; set; }
    public string SystemName { get; set; }
    public string SymbolName { get; set; }
    public string SymbolPeriod { get; set; }*/

    public dynamic V          { get; set; }
    public List<float> Open   { get; set; }
    public List<float> High   { get; set; }
    public List<float> Low    { get; set; }
    public List<float> Close  { get; set; }
    public List<float> Volume { get; set; }
    public List<float> Lot    { get; set; }
    public int BarCount       { get; set; }
    public int LastBarIndex   { get; set; }

    ~CBase()
    {

    }

    public CBase()
    {

    }

    public void ShowMessage(dynamic Sistem, string Message)
    {
        Sistem.Mesaj(Message);
    }

    public void SetData(dynamic Sistem, dynamic V, dynamic Open, dynamic High, dynamic Low, dynamic Close, dynamic Volume, dynamic Lot)
    {
        this.V = V;
        this.Open = Open;
        this.High = High;
        this.Low = Low;
        this.Close = Close;
        this.Volume = Volume;
        this.Lot = Lot;
        this.BarCount = V.Count;
        this.LastBarIndex = V.Count-1;
    }

    /*
    // Sadece SetData'yi kullanmaya karar verdim
    public int SetGrafikVerileri(dynamic Sistem, dynamic GrafikVerileri)
    {
        return parseData(Sistem, GrafikVerileri);
    }

    private int parseData(dynamic Sistem, dynamic Data)
    {
        int result = 0;

        try
        {
            Open = Sistem.GrafikFiyatOku(Data, "Acilis");
            High = Sistem.GrafikFiyatOku(Data, "Yuksek");
            Low = Sistem.GrafikFiyatOku(Data, "Dusuk");
            Close = Sistem.GrafikFiyatOku(Data, "Kapanis");
            Volume = Sistem.GrafikFiyatOku(Data, "Hacim");
            Lot = Sistem.GrafikFiyatOku(Data, "Lot");

            V = Data;
            BarCount = V.Count;
            LastBarIndex = V.Count - 1;

            result = 0;
        }
        catch (Exception error)
        {
            result = -1;
        }

        return result;
    }*/
}