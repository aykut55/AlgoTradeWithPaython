public class CIndicatorManager : CBase
{
    public List< string > ParamsList = new List< string > ();    
    public List< List<float> > ValuesList = new List< List<float> >(); 
    
    public List< string > MaYontemList = new List< string > { "Exp", "HullMA", "Simple", "TimeSeries", "Triangular", "Variable", "Volume", "Weighted", "Wilder", "ZeroLag" };
    
    public List< int > MaPeriyodList = new List< int > ();
    public List< int > MaPeriyodListFiboNumbers = new List< int > { 3, 5, 8, 13, 21, 34, 55, 89, 144, 233 };
    public List< int > MaPeriyodListCommonNumbers = new List< int > { 5, 10, 15, 20, 30, 45, 50, 100, 200, 500, 1000 };

    public List< string > MaParamsList = new List< string > ();
    public List< List<float> > MaList = new List< List<float> >();
    public Dictionary<string, dynamic> MaDictionary = new Dictionary<string, dynamic>();

    public List< string > RsiParamsList = new List< string > ();
    public List< List<float> > RsiList = new List< List<float> >();

    public List< string > MacdParamsList = new List< string > ();    
    public List< List<float> > MacdList = new List< List<float> >(); 
    
    public List< string > StochasticFastParamsList = new List< string > ();    
    public List< List<float> > StochasticFastList = new List< List<float> >(); 
    
    public List< string > StochasticSlowParamsList = new List< string > ();    
    public List< List<float> > StochasticSlowList = new List< List<float> >(); 
    
    public List< string > StochasticRSIParamsList = new List< string > ();    
    public List< List<float> > StochasticRSIList = new List< List<float> >(); 
    
    public List< string > StochasticOscParamsList = new List< string > ();    
    public List< List<float> > StochasticOscList = new List< List<float> >(); 
    
    public List< string > TomaParamsList = new List< string > ();    
    public List< List<float> > TomaList = new List< List<float> >();    
    
    ~CIndicatorManager()
    {

    }

    public CIndicatorManager()
    {

    }

    public CIndicatorManager Initialize(dynamic Sistem, dynamic V, dynamic Open, dynamic High, dynamic Low, dynamic Close, dynamic Volume, dynamic Lot)
    {
        SetData(Sistem, V, Open, High, Low, Close, Volume, Lot);

        return this;
    }

    public CIndicatorManager Reset(dynamic Sistem)
    {
        MaParamsList.Clear();
        
        MaList.Clear();
        
        MaDictionary.Clear();

        RsiParamsList.Clear();
        
        RsiList.Clear();

        MacdParamsList.Clear();
        
        MacdList.Clear();
                
        StochasticFastParamsList.Clear();
        
        StochasticFastList.Clear();
                
        StochasticSlowParamsList.Clear();
        
        StochasticSlowList.Clear();
                
        StochasticRSIParamsList.Clear();
        
        StochasticRSIList.Clear();
                
        StochasticOscParamsList.Clear();
        
        StochasticOscList.Clear();
        
        TomaParamsList.Clear();
        
        TomaList.Clear();

        return this;
    }

    public dynamic CalculateMA(dynamic Sistem, dynamic Source, string Method, int Periyod)
    {
        return Sistem.MA(Source, Method, Periyod);
    }

    public dynamic CalculateMA2(dynamic Sistem, dynamic Source, string Method, int Periyod)
    {
        return Sistem.MA2(Source, Method, Periyod);
    }

    public dynamic CalculateMA3(dynamic Sistem, dynamic Source, string Method, int Periyod)
    {
        return Sistem.MA3(Source, Method, Periyod);
    }

    public dynamic CalculateMAM(dynamic Sistem, dynamic Source, string Method = "Weighted")
    {
        return Sistem.MAM(Source, Method, 3, 5, 8, 13, 21, 34, 55, 89);
    }

    public dynamic CalculateMAM(dynamic Sistem, dynamic Source, string Method, List<int> PeriyodList)
    {
        var ma = Source;

        for (int i = 0; i < PeriyodList.Count; i++)
        {
            ma = Sistem.MA(ma, Method, PeriyodList[i]);
        }

        return ma;
    }

    public dynamic FillMaList(dynamic Sistem, dynamic Source, string Method, List<int> PeriyodList)
    {
        return FillMaList(Sistem, Source, new List<string> {Method}, PeriyodList);      
    }

    public dynamic FillMaList(dynamic Sistem, dynamic Source, List<string> MethodList, List<int> PeriyodList)
    {
        for (int i = 0; i < MethodList.Count; i++)
        {
            for (int j = 0; j < PeriyodList.Count; j++)
            {
                var maParam = String.Format("{0}_{1}", MethodList[i], PeriyodList[j]);
                        
                MaParamsList.Add( maParam );
                
                MaList.Add( Sistem.MA(Source, MethodList[i], PeriyodList[j]) );
            }
        }
        
        return MaList;
    }    
    
    public void ClearMaList(dynamic Sistem)
    {
        MaParamsList.Clear();
        
        MaList.Clear();
    }
    
    public dynamic CalculateRSI(dynamic Sistem, dynamic Source, int Periyod)
    {
        return Sistem.RSI(Source, Periyod);
    }

    public dynamic FillRsiList(dynamic Sistem, dynamic Source, List<int> PeriyodList)
    {
        for (int i = 0; i < PeriyodList.Count; i++)
        {
            var rsiParam = String.Format("{0}", PeriyodList[i]);
                    
            RsiParamsList.Add( rsiParam );
                    
            RsiList.Add( Sistem.RSI(Source, PeriyodList[i]) );
        }

        return RsiList;
    }

    public void ClearRsiList(dynamic Sistem)
    {
        RsiParamsList.Clear();
        
        RsiList.Clear();
    }
    
    public dynamic CalculateMACD(dynamic Sistem, dynamic Source, int PeriyodFast, int PeriyodSlow)
    {
        return Sistem.MACD(Source, PeriyodFast, PeriyodSlow);
    }

    public dynamic FillMacdList(dynamic Sistem, dynamic Source, List<int> PeriyodFastList, List<int> PeriyodSlowList)
    {
        for (int i = 0; i < PeriyodFastList.Count; i++)
        {
            for (int j = 0; j < PeriyodSlowList.Count; j++)
            {
                var macdParam = String.Format("{0}_{1}", PeriyodFastList[i], PeriyodSlowList[j]);
                        
                MacdParamsList.Add( macdParam );
                        
                MacdList.Add( Sistem.MACD(Source, PeriyodFastList[i], PeriyodSlowList[j]) );
            }
        }

        return MacdList;
    }

    public void ClearMacdList(dynamic Sistem)
    {
        MacdParamsList.Clear();
        
        MacdList.Clear();        
    }

    public dynamic CalculateStochasticFast(dynamic Sistem, dynamic Source, int Periyod1, int Periyod2)
    {
        return Sistem.StochasticFast(Source, Periyod1, Periyod2);
    }

    public dynamic FillStochasticFastList(dynamic Sistem, dynamic Source, List<int> Periyod1List, List<int> Periyod2List)
    {
        for (int i = 0; i < Periyod1List.Count; i++)
        {
            for (int j = 0; j < Periyod2List.Count; j++)
            {
                var stochasticFastParam = String.Format("{0}_{1}", Periyod1List[i], Periyod2List[j]);
                        
                StochasticFastParamsList.Add( stochasticFastParam );
                        
                StochasticFastList.Add( Sistem.StochasticFast(Source, Periyod1List[i], Periyod2List[j]) );
            }
        }

        return MacdList;
    }

    public void ClearStochasticFastList(dynamic Sistem)
    {
        StochasticFastParamsList.Clear();
        
        StochasticFastList.Clear();        
    }
    
    public dynamic CalculateStochasticSlow(dynamic Sistem, dynamic Source, int Periyod1, int Periyod2)
    {
        return Sistem.StochasticSlow(Source, Periyod1, Periyod2);
    }

    public dynamic FillStochasticSlowList(dynamic Sistem, dynamic Source, List<int> Periyod1List, List<int> Periyod2List)
    {
        for (int i = 0; i < Periyod1List.Count; i++)
        {
            for (int j = 0; j < Periyod2List.Count; j++)
            {
                var stochasticSlowParam = String.Format("{0}_{1}", Periyod1List[i], Periyod2List[j]);
                        
                StochasticSlowParamsList.Add( stochasticSlowParam );
                        
                StochasticSlowList.Add( Sistem.StochasticSlow(Source, Periyod1List[i], Periyod2List[j]) );
            }
        }

        return MacdList;
    }

    public void ClearStochasticSlowList(dynamic Sistem)
    {
        StochasticSlowParamsList.Clear();
        
        StochasticSlowList.Clear();            
    }
    
    public dynamic CalculateStochasticRSI(dynamic Sistem, dynamic Source, int Periyod)
    {
        return Sistem.StochasticRSI(Source, Periyod);
    }

    public dynamic FillStochasticRSIList(dynamic Sistem, dynamic Source, List<int> PeriyodList)
    {
        for (int i = 0; i < PeriyodList.Count; i++)
        {
            var stochasticRSIParam = String.Format("{0}", PeriyodList[i]);
                    
            StochasticRSIParamsList.Add( stochasticRSIParam );
                    
            StochasticRSIList.Add( Sistem.StochasticRSI(Source, PeriyodList[i]) );
        }

        return RsiList;
    }

    public void ClearStochasticRSIList(dynamic Sistem)
    {
        StochasticRSIParamsList.Clear();
        
        StochasticRSIList.Clear();  
    }

    public dynamic CalculateStochasticOsc(dynamic Sistem, dynamic Source, int Periyod1, int Periyod2)
    {
        return Sistem.StochasticOsc(Source, Periyod1, Periyod2);
    }

    public dynamic FillStochasticOscList(dynamic Sistem, dynamic Source, List<int> Periyod1List, List<int> Periyod2List)
    {
        for (int i = 0; i < Periyod1List.Count; i++)
        {
            for (int j = 0; j < Periyod2List.Count; j++)
            {
                var stochasticOscParam = String.Format("{0}_{1}", Periyod1List[i], Periyod2List[j]);
                        
                StochasticOscParamsList.Add( stochasticOscParam );
                        
                StochasticOscList.Add( Sistem.StochasticOsc(Source, Periyod1List[i], Periyod2List[j]) );
            }
        }

        return MacdList;
    }

    public void ClearStochasticOscList(dynamic Sistem)
    {
        StochasticOscParamsList.Clear();
        
        StochasticOscList.Clear();        
    }

    public dynamic CalculateTOMA(dynamic Sistem, dynamic Source, int Periyod, float Percentage, string Method)
    {
        return Sistem.TOMA(Source, Periyod, Percentage, Method);
    }   
    
    public dynamic FillTomaList(dynamic Sistem, dynamic Source, List<int> PeriyodList, List<float> PercentageList, string Method)
    {
        return FillTomaList(Sistem, Source, PeriyodList, PercentageList, new List<string> {Method} );
    }    
    
    public dynamic FillTomaList(dynamic Sistem, dynamic Source, List<int> PeriyodList, List<float> PercentageList, List<string> MethodList)
    {
        for (int i = 0; i < PeriyodList.Count; i++)
        {
            for (int j = 0; j < PercentageList.Count; j++)
            {
                for (int k = 0; k < MethodList.Count; k++)
                {
                    var tomaParam = String.Format("{0}_{1}_{2}", PeriyodList[i], PercentageList[j], MethodList[k]);
                            
                    TomaParamsList.Add( tomaParam );
                    
                    TomaList.Add( Sistem.TOMA(Source, PeriyodList[i], PercentageList[j], MethodList[k]) );
                }
            }
        }

        return TomaList;
    }  
    
    public void ClearTomaList(dynamic Sistem)
    {
        TomaParamsList.Clear();
        
        TomaList.Clear();
    } 
}