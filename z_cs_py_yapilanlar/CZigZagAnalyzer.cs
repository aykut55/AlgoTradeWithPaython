public class CZigZagAnalyzer : CBase
{
    public float ZigZagParam = 0f;
    public List<float> ZigZag = null;
    public List<string> EmirList = null;
    public List< string > LogMessageList = null;
    public List<float> PercentageChangeList = null;
    string delimiter = ";";

    ~CZigZagAnalyzer()
    {

    }

    public CZigZagAnalyzer()
    {
        ZigZag = new List<float> ();
        EmirList = new List<string> ();
        LogMessageList = new List<string> ();
        PercentageChangeList = new List<float> ();
    }

    public CZigZagAnalyzer Initialize(dynamic Sistem, dynamic V, dynamic Open, dynamic High, dynamic Low, dynamic Close, dynamic Volume, dynamic Lot)
    {
        SetData(Sistem, V, Open, High, Low, Close, Volume, Lot);

        return this;
    }

    public CZigZagAnalyzer Reset(dynamic Sistem)
    {
        ZigZag = null;

        EmirList.Clear();

        LogMessageList.Clear();

        return this;
    }

    public void Calculate(dynamic Sistem, List<float> Source, float Percentage)
    {
        ZigZagParam = Percentage;
        ZigZag = Sistem.ZigZagPercent(Source, Percentage);
    }

    public void FillEmirList(dynamic Sistem)
    {
        var Emir = "";
        var BarIndex = 0;
        bool ListeyeEkle = false;

        EmirList.Clear();

        for (int i = 0; i < BarCount; i++)
        {
            EmirList.Add("");

            if (i < 2) continue;

            bool al1 = true;

            al1 = al1 && ( ZigZag[i-2] > ZigZag[i-1] ) && ( ZigZag[i-1] < ZigZag[i] );

            bool sat1 = true;

            sat1 = sat1 && ( ZigZag[i-2] < ZigZag[i-1] ) && ( ZigZag[i-1] > ZigZag[i] );

            if (al1)  { Emir = "A"; BarIndex = i-1; EmirList[BarIndex] = Emir; ListeyeEkle = true; }

            if (sat1) { Emir = "S"; BarIndex = i-1; EmirList[BarIndex] = Emir; ListeyeEkle = true; }

            if (ListeyeEkle)
            {
                ListeyeEkle = false;
            }
        }
    }

    public void CalculatePercentageChange(dynamic Sistem)
    {
        var Emir = "";
        var BarIndex = 0;
        bool ListeyeEkle = false;

        var PrevValue = ZigZag[0];
        var PercentageChange = 0f;
        PercentageChangeList = Sistem.Liste(BarCount, 0f);

        for (int i = 0; i < BarCount; i++)
        {
            PercentageChange = (ZigZag[i] - PrevValue) * 100f / PrevValue;

            PercentageChangeList[i] = PercentageChange;

            if (i < 1) continue;

            bool al1 = true;

            bool sat1 = true;

            al1 = al1 && ( EmirList[i] == "A" );

            sat1 = sat1 && ( EmirList[i] == "S" );

            if (al1)  { PrevValue = ZigZag[i]; }

            if (sat1) { PrevValue = ZigZag[i]; }
        }
    }

    public string CreateLogMessage(dynamic Sistem, int BarIndex, float ZigZagValue, string SistemYon, float Param1 = 0.0f, float Param2 = 0.0f, float Param3 = 0.0f)
    {
        int i = BarIndex;

        string str = String.Format("{0} {1,-5} {2} {3, 10} {4} {5, 10} {6}  {7, 10} {8} {9, 10} {10} {11, 10} {12} {13, 10} {14} ",  delimiter, i, delimiter, ZigZagValue.ToString("0.00"), delimiter, SistemYon, delimiter, BarIndex, delimiter, Param1.ToString("0.00"), delimiter, Param2.ToString("0.00"), delimiter, Param3.ToString("0.00"), delimiter);  //

        return str;
    }

    public void AddLogMessage(dynamic Sistem, string LogMessage)
    {
        LogMessageList.Add(LogMessage);
    }

    public void UpdateLogMessage(dynamic Sistem, int BarIndex, string LogMessage)
    {
        int i = BarIndex;

        LogMessageList[i] = LogMessage;
    }

    public void WriteToFile(dynamic Sistem, string FileName)
    {
        CFileUtils myFileUtils = new CFileUtils();

        CTimeUtils myTimeUtils = new CTimeUtils();

        myTimeUtils.Initialize(Sistem, V, Open, High, Low, Close, Volume, Lot);

        myTimeUtils.GecenZamanBilgileriniAl(Sistem);

        string aciklama1 = "...";
        string aciklama2 = "...";
        string aciklama3 = "...";
        string aciklama4 = "...";
        string aciklama5 = "...";
        string aciklama6 = "...";

        string logFileFullName = FileName.Trim();
    }

}