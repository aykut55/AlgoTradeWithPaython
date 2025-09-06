public class CUtils
    {
        ~CUtils()
        {

        }

        public CUtils()
        {

        }

        public CUtils Initialize(dynamic Sistem)
        {
            return this;
        }

        public CUtils Reset(dynamic Sistem)
        {
            return this;
        }

        #region BirimCevrimler

        public bool GetBool(dynamic Value)
        {
            return Convert.ToBoolean(Value);
        }

        public int GetInteger16(dynamic Value)
        {
            return Convert.ToInt16(Value);
        }

        public int GetInteger32(dynamic Value)
        {
            return Convert.ToInt32(Value);
        }

        public int GetInteger64(dynamic Value)
        {
            return Convert.ToInt64(Value);
        }

        public int GetInteger(dynamic Value)
        {
            return GetInteger32(Value);
        }

        public float GetFloat(dynamic Value)
        {
            return Convert.ToSingle(Value);
        }

        public double GetDouble(dynamic Value)
        {
            return Convert.ToDouble(Value);
        }

        public bool GetBool(string Value, string trueValue = "1")
        {
            return Value == trueValue ? true : false;
        }

        public bool GetBool(int Value, int trueValue = 1)
        {
            return Value == trueValue ? true : false;
        }

        #endregion BirimCevrimler

        #region Kesisimler

        public T GetMax<T>(T lhs, T rhs)
        {
            return Comparer<T>.Default.Compare(lhs, rhs) > 0 ? lhs : rhs;
        }

        public bool YukariKesti(dynamic Sistem, int i, List<int> ListX, List<int> ListY, bool esitlikDahil = true)
        {
            bool res = false;

            if (esitlikDahil)
            {
                res = ListX[i - 1] < ListY[i - 1] && ListX[i] >= ListY[i];
            }
            else
            {
                res = ListX[i - 1] < ListY[i - 1] && ListX[i] > ListY[i];
            }

            return res;
        }

        public bool YukariKesti(dynamic Sistem, int i, List<float> ListX, List<float> ListY, bool esitlikDahil = true)
        {
            bool res = false;

            if (esitlikDahil)
            {
                res = ListX[i - 1] < ListY[i - 1] && ListX[i] >= ListY[i];
            }
            else
            {
                res = ListX[i - 1] < ListY[i - 1] && ListX[i] > ListY[i];
            }

            return res;
        }

        public bool YukariKesti(dynamic Sistem, int i, List<double> ListX, List<double> ListY, bool esitlikDahil = true)
        {
            bool res = false;

            if (esitlikDahil)
            {
                res = ListX[i - 1] < ListY[i - 1] && ListX[i] >= ListY[i];
            }
            else
            {
                res = ListX[i - 1] < ListY[i - 1] && ListX[i] > ListY[i];
            }

            return res;
        }

        public bool AsagiKesti(dynamic Sistem, int i, List<int> ListX, List<int> ListY, bool esitlikDahil = true)
        {
            bool res = false;

            if (esitlikDahil)
            {
                res = ListX[i - 1] > ListY[i - 1] && ListX[i] <= ListY[i];
            }
            else
            {
                res = ListX[i - 1] > ListY[i - 1] && ListX[i] < ListY[i];
            }

            return res;
        }

        public bool AsagiKesti(dynamic Sistem, int i, List<float> ListX, List<float> ListY, bool esitlikDahil = true)
        {
            bool res = false;

            if (esitlikDahil)
            {
                res = ListX[i - 1] > ListY[i - 1] && ListX[i] <= ListY[i];
            }
            else
            {
                res = ListX[i - 1] > ListY[i - 1] && ListX[i] < ListY[i];
            }

            return res;
        }

        public bool AsagiKesti(dynamic Sistem, int i, List<double> ListX, List<double> ListY, bool esitlikDahil = true)
        {
            bool res = false;

            if (esitlikDahil)
            {
                res = ListX[i - 1] > ListY[i - 1] && ListX[i] <= ListY[i];
            }
            else
            {
                res = ListX[i - 1] > ListY[i - 1] && ListX[i] < ListY[i];
            }

            return res;
        }

        public bool YukariKesti(dynamic Sistem, int i, List<int> ListX, int Seviye, bool esitlikDahil = true)
        {
            bool res = false;

            if (esitlikDahil)
            {
                res = ListX[i - 1] < Seviye && ListX[i] >= Seviye;
            }
            else
            {
                res = ListX[i - 1] < Seviye && ListX[i] > Seviye;
            }

            return res;
        }

        public bool YukariKesti(dynamic Sistem, int i, List<float> ListX, float Seviye, bool esitlikDahil = true)
        {
            bool res = false;

            if (esitlikDahil)
            {
                res = ListX[i - 1] < Seviye && ListX[i] >= Seviye;
            }
            else
            {
                res = ListX[i - 1] < Seviye && ListX[i] > Seviye;
            }

            return res;
        }

        public bool YukariKesti(dynamic Sistem, int i, List<double> ListX, double Seviye, bool esitlikDahil = true)
        {
            bool res = false;

            if (esitlikDahil)
            {
                res = ListX[i - 1] < Seviye && ListX[i] >= Seviye;
            }
            else
            {
                res = ListX[i - 1] < Seviye && ListX[i] > Seviye;
            }

            return res;
        }

        public bool AsagiKesti(dynamic Sistem, int i, List<int> ListX, int Seviye, bool esitlikDahil = true)
        {
            bool res = false;

            if (esitlikDahil)
            {
                res = ListX[i - 1] > Seviye && ListX[i] <= Seviye;
            }
            else
            {
                res = ListX[i - 1] > Seviye && ListX[i] < Seviye;
            }

            return res;
        }

        public bool AsagiKesti(dynamic Sistem, int i, List<float> ListX, float Seviye, bool esitlikDahil = true)
        {
            bool res = false;

            if (esitlikDahil)
            {
                res = ListX[i - 1] > Seviye && ListX[i] <= Seviye;
            }
            else
            {
                res = ListX[i - 1] > Seviye && ListX[i] < Seviye;
            }

            return res;
        }

        public bool AsagiKesti(dynamic Sistem, int i, List<double> ListX, double Seviye, bool esitlikDahil = true)
        {
            bool res = false;

            if (esitlikDahil)
            {
                res = ListX[i - 1] > Seviye && ListX[i] <= Seviye;
            }
            else
            {
                res = ListX[i - 1] > Seviye && ListX[i] < Seviye;
            }

            return res;
        }

        #endregion Kesisimler

        #region LogicIslemler

        // List - List
        public bool Buyuk(dynamic Sistem, int i, List<int> ListX, List<int> ListY)
        {
            bool res = false;

            res = ListX[i] > ListY[i];

            return res;
        }

        public bool Buyuk(dynamic Sistem, int i, List<float> ListX, List<float> ListY)
        {
            bool res = false;

            res = ListX[i] > ListY[i];

            return res;
        }

        public bool Buyuk(dynamic Sistem, int i, List<double> ListX, List<double> ListY)
        {
            bool res = false;

            res = ListX[i] > ListY[i];

            return res;
        }

        public bool Kucuk(dynamic Sistem, int i, List<int> ListX, List<int> ListY)
        {
            bool res = false;

            res = ListX[i] < ListY[i];

            return res;
        }

        public bool Kucuk(dynamic Sistem, int i, List<float> ListX, List<float> ListY)
        {
            bool res = false;

            res = ListX[i] < ListY[i];

            return res;
        }

        public bool Kucuk(dynamic Sistem, int i, List<double> ListX, List<double> ListY)
        {
            bool res = false;

            res = ListX[i] < ListY[i];

            return res;
        }

        public bool BuyukEsit(dynamic Sistem, int i, List<int> ListX, List<int> ListY)
        {
            bool res = false;

            res = ListX[i] >= ListY[i];

            return res;
        }

        public bool BuyukEsit(dynamic Sistem, int i, List<float> ListX, List<float> ListY)
        {
            bool res = false;

            res = ListX[i] >= ListY[i];

            return res;
        }

        public bool BuyukEsit(dynamic Sistem, int i, List<double> ListX, List<double> ListY)
        {
            bool res = false;

            res = ListX[i] >= ListY[i];

            return res;
        }

        public bool KucukEsit(dynamic Sistem, int i, List<int> ListX, List<int> ListY)
        {
            bool res = false;

            res = ListX[i] <= ListY[i];

            return res;
        }

        public bool KucukEsit(dynamic Sistem, int i, List<float> ListX, List<float> ListY)
        {
            bool res = false;

            res = ListX[i] <= ListY[i];

            return res;
        }

        public bool KucukEsit(dynamic Sistem, int i, List<double> ListX, List<double> ListY)
        {
            bool res = false;

            res = ListX[i] <= ListY[i];

            return res;
        }

        public bool Esit(dynamic Sistem, int i, List<int> ListX, List<int> ListY)
        {
            bool res = false;

            res = ListX[i] == ListY[i];

            return res;
        }

        public bool Esit(dynamic Sistem, int i, List<float> ListX, List<float> ListY)
        {
            bool res = false;

            res = ListX[i] == ListY[i];

            return res;
        }

        public bool Esit(dynamic Sistem, int i, List<double> ListX, List<double> ListY)
        {
            bool res = false;

            res = ListX[i] == ListY[i];

            return res;
        }

        public bool EsitDegil(dynamic Sistem, int i, List<int> ListX, List<int> ListY)
        {
            bool res = false;

            res = ListX[i] != ListY[i];

            return res;
        }

        public bool EsitDegil(dynamic Sistem, int i, List<float> ListX, List<float> ListY)
        {
            bool res = false;

            res = ListX[i] != ListY[i];

            return res;
        }

        public bool EsitDegil(dynamic Sistem, int i, List<double> ListX, List<double> ListY)
        {
            bool res = false;

            res = ListX[i] != ListY[i];

            return res;
        }

        // List - Seviye
        public bool Buyuk(dynamic Sistem, int i, List<int> ListX, int Seviye)
        {
            bool res = false;

            res = ListX[i] > Seviye;

            return res;
        }

        public bool Buyuk(dynamic Sistem, int i, List<float> ListX, float Seviye)
        {
            bool res = false;

            res = ListX[i] > Seviye;

            return res;
        }

        public bool Buyuk(dynamic Sistem, int i, List<double> ListX, double Seviye)
        {
            bool res = false;

            res = ListX[i] > Seviye;

            return res;
        }

        public bool Kucuk(dynamic Sistem, int i, List<int> ListX, int Seviye)
        {
            bool res = false;

            res = ListX[i] < Seviye;

            return res;
        }

        public bool Kucuk(dynamic Sistem, int i, List<float> ListX, float Seviye)
        {
            bool res = false;

            res = ListX[i] < Seviye;

            return res;
        }

        public bool Kucuk(dynamic Sistem, int i, List<double> ListX, double Seviye)
        {
            bool res = false;

            res = ListX[i] < Seviye;

            return res;
        }

        public bool BuyukEsit(dynamic Sistem, int i, List<int> ListX, int Seviye)
        {
            bool res = false;

            res = ListX[i] >= Seviye;

            return res;
        }

        public bool BuyukEsit(dynamic Sistem, int i, List<float> ListX, float Seviye)
        {
            bool res = false;

            res = ListX[i] >= Seviye;

            return res;
        }

        public bool BuyukEsit(dynamic Sistem, int i, List<double> ListX, double Seviye)
        {
            bool res = false;

            res = ListX[i] >= Seviye;

            return res;
        }

        public bool KucukEsit(dynamic Sistem, int i, List<int> ListX, int Seviye)
        {
            bool res = false;

            res = ListX[i] <= Seviye;

            return res;
        }

        public bool KucukEsit(dynamic Sistem, int i, List<float> ListX, float Seviye)
        {
            bool res = false;

            res = ListX[i] <= Seviye;

            return res;
        }

        public bool KucukEsit(dynamic Sistem, int i, List<double> ListX, double Seviye)
        {
            bool res = false;

            res = ListX[i] <= Seviye;

            return res;
        }

        public bool Esit(dynamic Sistem, int i, List<int> ListX, int Seviye)
        {
            bool res = false;

            res = ListX[i] == Seviye;

            return res;
        }

        public bool Esit(dynamic Sistem, int i, List<float> ListX, float Seviye)
        {
            bool res = false;

            res = ListX[i] == Seviye;

            return res;
        }

        public bool Esit(dynamic Sistem, int i, List<double> ListX, double Seviye)
        {
            bool res = false;

            res = ListX[i] == Seviye;

            return res;
        }

        public bool EsitDegil(dynamic Sistem, int i, List<int> ListX, int Seviye)
        {
            bool res = false;

            res = ListX[i] != Seviye;

            return res;
        }

        public bool EsitDegil(dynamic Sistem, int i, List<float> ListX, float Seviye)
        {
            bool res = false;

            res = ListX[i] != Seviye;

            return res;
        }

        public bool EsitDegil(dynamic Sistem, int i, List<double> ListX, double Seviye)
        {
            bool res = false;

            res = ListX[i] != Seviye;

            return res;
        }

        // List - List
        public bool Ustunde(dynamic Sistem, int i, List<int> ListX, List<int> ListY)
        {
            return Buyuk(Sistem, i, ListX, ListY);
        }

        public bool Ustunde(dynamic Sistem, int i, List<float> ListX, List<float> ListY)
        {
            return Buyuk(Sistem, i, ListX, ListY);
        }

        public bool Ustunde(dynamic Sistem, int i, List<double> ListX, List<double> ListY)
        {
            return Buyuk(Sistem, i, ListX, ListY);
        }

        public bool Altinda(dynamic Sistem, int i, List<int> ListX, List<int> ListY)
        {
            return Kucuk(Sistem, i, ListX, ListY);
        }

        public bool Altinda(dynamic Sistem, int i, List<float> ListX, List<float> ListY)
        {
            return Kucuk(Sistem, i, ListX, ListY);
        }

        public bool Altinda(dynamic Sistem, int i, List<double> ListX, List<double> ListY)
        {
            return Kucuk(Sistem, i, ListX, ListY);
        }

        // List - Seviye
        public bool Ustunde(dynamic Sistem, int i, List<int> ListX, int Seviye)
        {
            return Buyuk(Sistem, i, ListX, Seviye);
        }

        public bool Ustunde(dynamic Sistem, int i, List<float> ListX, float Seviye)
        {
            return Buyuk(Sistem, i, ListX, Seviye);
        }

        public bool Ustunde(dynamic Sistem, int i, List<double> ListX, double Seviye)
        {
            return Buyuk(Sistem, i, ListX, Seviye);
        }

        public bool Altinda(dynamic Sistem, int i, List<int> ListX, int Seviye)
        {
            return Kucuk(Sistem, i, ListX, Seviye);
        }

        public bool Altinda(dynamic Sistem, int i, List<float> ListX, float Seviye)
        {
            return Kucuk(Sistem, i, ListX, Seviye);
        }

        public bool Altinda(dynamic Sistem, int i, List<double> ListX, double Seviye)
        {
            return Kucuk(Sistem, i, ListX, Seviye);
        }

        // List - List
        public bool Buyuk(dynamic Sistem, int i, List<int> List1, List<int> List2, List<int> List3, List<int> List4 = null, List<int> List5 = null, List<int> List6 = null, List<int> List7 = null, List<int> List8 = null, List<int> List9 = null, List<int> List10 = null)
        {
            bool res = false;

            res = ( List1[i] > List2[i] ) && ( List2[i] > List3[i] );
            if (List4 != null)
            {
                res = res && ( List3[i] > List4[i] );
                if (List5 != null)
                {
                    res = res && ( List4[i] > List5[i] );
                    if (List6 != null)
                    {
                        res = res && ( List5[i] > List6[i] );
                        if (List7 != null)
                        {
                            res = res && ( List6[i] > List7[i] );
                            if (List8 != null)
                            {
                                res = res && ( List7[i] > List8[i] );
                                if (List9 != null)
                                {
                                    res = res && ( List8[i] > List9[i] );
                                    if (List10 != null)
                                    {
                                        res = res && ( List9[i] > List10[i] );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return res;
        }

        public bool Buyuk(dynamic Sistem, int i, List<float> List1, List<float> List2, List<float> List3, List<float> List4 = null, List<float> List5 = null, List<float> List6 = null, List<float> List7 = null, List<float> List8 = null, List<float> List9 = null, List<float> List10 = null)
        {
            bool res = false;

            res = ( List1[i] > List2[i] ) && ( List2[i] > List3[i] );
            if (List4 != null)
            {
                res = res && ( List3[i] > List4[i] );
                if (List5 != null)
                {
                    res = res && ( List4[i] > List5[i] );
                    if (List6 != null)
                    {
                        res = res && ( List5[i] > List6[i] );
                        if (List7 != null)
                        {
                            res = res && ( List6[i] > List7[i] );
                            if (List8 != null)
                            {
                                res = res && ( List7[i] > List8[i] );
                                if (List9 != null)
                                {
                                    res = res && ( List8[i] > List9[i] );
                                    if (List10 != null)
                                    {
                                        res = res && ( List9[i] > List10[i] );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return res;
        }

        public bool Buyuk(dynamic Sistem, int i, List<double> List1, List<double> List2, List<double> List3, List<double> List4 = null, List<double> List5 = null, List<double> List6 = null, List<double> List7 = null, List<double> List8 = null, List<double> List9 = null, List<double> List10 = null)
        {
            bool res = false;

            res = ( List1[i] > List2[i] ) && ( List2[i] > List3[i] );
            if (List4 != null)
            {
                res = res && ( List3[i] > List4[i] );
                if (List5 != null)
                {
                    res = res && ( List4[i] > List5[i] );
                    if (List6 != null)
                    {
                        res = res && ( List5[i] > List6[i] );
                        if (List7 != null)
                        {
                            res = res && ( List6[i] > List7[i] );
                            if (List8 != null)
                            {
                                res = res && ( List7[i] > List8[i] );
                                if (List9 != null)
                                {
                                    res = res && ( List8[i] > List9[i] );
                                    if (List10 != null)
                                    {
                                        res = res && ( List9[i] > List10[i] );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return res;
        }

        public bool Kucuk(dynamic Sistem, int i, List<int> List1, List<int> List2, List<int> List3, List<int> List4 = null, List<int> List5 = null, List<int> List6 = null, List<int> List7 = null, List<int> List8 = null, List<int> List9 = null, List<int> List10 = null)
        {
            bool res = false;

            res = ( List1[i] < List2[i] ) && ( List2[i] < List3[i] );
            if (List4 != null)
            {
                res = res && ( List3[i] < List4[i] );
                if (List5 != null)
                {
                    res = res && ( List4[i] < List5[i] );
                    if (List6 != null)
                    {
                        res = res && ( List5[i] < List6[i] );
                        if (List7 != null)
                        {
                            res = res && ( List6[i] < List7[i] );
                            if (List8 != null)
                            {
                                res = res && ( List7[i] < List8[i] );
                                if (List9 != null)
                                {
                                    res = res && ( List8[i] < List9[i] );
                                    if (List10 != null)
                                    {
                                        res = res && ( List9[i] < List10[i] );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return res;
        }

        public bool Kucuk(dynamic Sistem, int i, List<float> List1, List<float> List2, List<float> List3, List<float> List4 = null, List<float> List5 = null, List<float> List6 = null, List<float> List7 = null, List<float> List8 = null, List<float> List9 = null, List<float> List10 = null)
        {
            bool res = false;

            res = ( List1[i] < List2[i] ) && ( List2[i] < List3[i] );
            if (List4 != null)
            {
                res = res && ( List3[i] < List4[i] );
                if (List5 != null)
                {
                    res = res && ( List4[i] < List5[i] );
                    if (List6 != null)
                    {
                        res = res && ( List5[i] < List6[i] );
                        if (List7 != null)
                        {
                            res = res && ( List6[i] < List7[i] );
                            if (List8 != null)
                            {
                                res = res && ( List7[i] < List8[i] );
                                if (List9 != null)
                                {
                                    res = res && ( List8[i] < List9[i] );
                                    if (List10 != null)
                                    {
                                        res = res && ( List9[i] < List10[i] );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return res;
        }

        public bool Kucuk(dynamic Sistem, int i, List<double> List1, List<double> List2, List<double> List3, List<double> List4 = null, List<double> List5 = null, List<double> List6 = null, List<double> List7 = null, List<double> List8 = null, List<double> List9 = null, List<double> List10 = null)
        {
            bool res = false;

            res = ( List1[i] < List2[i] ) && ( List2[i] < List3[i] );
            if (List4 != null)
            {
                res = res && ( List3[i] < List4[i] );
                if (List5 != null)
                {
                    res = res && ( List4[i] < List5[i] );
                    if (List6 != null)
                    {
                        res = res && ( List5[i] < List6[i] );
                        if (List7 != null)
                        {
                            res = res && ( List6[i] < List7[i] );
                            if (List8 != null)
                            {
                                res = res && ( List7[i] < List8[i] );
                                if (List9 != null)
                                {
                                    res = res && ( List8[i] < List9[i] );
                                    if (List10 != null)
                                    {
                                        res = res && ( List9[i] < List10[i] );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return res;
        }

        #endregion LogicIslemler

        #region MathIslemler

        public List<int> ListeTopla(dynamic Sistem, List<int> Source, int K)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] + K;
            }

            return result;
        }

        public List<float> ListeTopla(dynamic Sistem, List<float> Source, float K)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] + K;
            }

            return result;
        }

        public List<double> ListeTopla(dynamic Sistem, List<double> Source, double K)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] + K;
            }

            return result;
        }

        public List<int> ListeCikar(dynamic Sistem, List<int> Source, int K)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] - K;
            }

            return result;
        }

        public List<float> ListeCikar(dynamic Sistem, List<float> Source, float K)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] - K;
            }

            return result;
        }

        public List<double> ListeCikar(dynamic Sistem, List<double> Source, double K)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] - K;
            }

            return result;
        }

        public List<int> ListeCarp(dynamic Sistem, List<int> Source, int K)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] * K;
            }

            return result;
        }

        public List<float> ListeCarp(dynamic Sistem, List<float> Source, float K)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] * K;
            }

            return result;
        }

        public List<double> ListeCarp(dynamic Sistem, List<double> Source, double K)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] * K;
            }

            return result;
        }

        public List<int> ListeBol(dynamic Sistem, List<int> Source, int K)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] / K;
            }

            return result;
        }

        public List<float> ListeBol(dynamic Sistem, List<float> Source, float K)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] / K;
            }

            return result;
        }

        public List<double> ListeBol(dynamic Sistem, List<double> Source, double K)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] / K;
            }

            return result;
        }

        public List<int> ListeYuzde(dynamic Sistem, List<int> Source, int K, int Offset = 0)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] * K / 100 + Offset;
            }

            return result;
        }

        public List<float> ListeYuzde(dynamic Sistem, List<float> Source, float K, float Offset = 0.0f)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] * K / 100.0f + Offset;
            }

            return result;
        }

        public List<double> ListeYuzde(dynamic Sistem, List<double> Source, double K, double Offset = 0.0)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] * K / 100.0+ Offset;
            }

            return result;
        }

        public List<int> ListeYuzde(dynamic Sistem, List<int> Source, int K, List<int> OffsetList = null)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] * K / 100;
                if (OffsetList != null)
                    result[i] = result[i] + OffsetList[i];
            }

            return result;
        }

        public List<float> ListeYuzde(dynamic Sistem, List<float> Source, float K, List<float> OffsetList = null)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] * K / 100.0f;
                if (OffsetList != null)
                    result[i] = result[i] + OffsetList[i];
            }

            return result;
        }

        public List<double> ListeYuzde(dynamic Sistem, List<double> Source, double K, List<double> OffsetList = null)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source[i] * K / 100.0;
                if (OffsetList != null)
                    result[i] = result[i] + OffsetList[i];
            }

            return result;
        }

        public List<int> ListeTopla(dynamic Sistem, List<int> Source1, List<int> Source2)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source1[i] + Source2[i];
            }

            return result;
        }

        public List<float> ListeTopla(dynamic Sistem, List<float> Source1, List<float> Source2)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source1[i] + Source2[i];
            }

            return result;
        }

        public List<double> ListeTopla(dynamic Sistem, List<double> Source1, List<double> Source2)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source1[i] + Source2[i];
            }

            return result;
        }

        public List<int> ListeCikar(dynamic Sistem, List<int> Source1, List<int> Source2)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source1[i] - Source2[i];
            }

            return result;
        }

        public List<float> ListeCikar(dynamic Sistem, List<float> Source1, List<float> Source2)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source1[i] - Source2[i];
            }

            return result;
        }

        public List<double> ListeCikar(dynamic Sistem, List<double> Source1, List<double> Source2)
        {
            var result = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                result[i] = Source1[i] - Source2[i];
            }

            return result;
        }

        #endregion MathIslemler

        #region ListOperations

        public List<int> ListeElemanlariniResetle(dynamic Sistem, int begIndex, int barCount, List<int> list, int lowerLimit, int upperLimit)
        {
            int value = 0;
            return ListeElemanlariniSetEt(Sistem, begIndex, barCount, list, lowerLimit, upperLimit, value);
        }

        public List<float> ListeElemanlariniResetle(dynamic Sistem, int begIndex, int barCount, List<float> list, float lowerLimit, float upperLimit)
        {
            float value = 0;
            return ListeElemanlariniSetEt(Sistem, begIndex, barCount, list, lowerLimit, upperLimit, value);
        }

        public List<double> ListeElemanlariniResetle(dynamic Sistem, int begIndex, int barCount, List<double> list, double lowerLimit, double upperLimit)
        {
            double value = 0.0;
            return ListeElemanlariniSetEt(Sistem, begIndex, barCount, list, lowerLimit, upperLimit, value);
        }

        public List<int> ListeElemanlariniSetEt(dynamic Sistem, int begIndex, int barCount, List<int> list, int lowerLimit, int upperLimit, int newValue = 0)
        {
            List<int> newList = Sistem.Liste(list.Count, 0);

            for (int i = begIndex; i < barCount; i++)
            {
                if (lowerLimit <= list[i] && list[i] <= upperLimit)
                {
                    newList[i] = newValue;
                }
                else
                {
                    newList[i] = list[i];
                }
            }

            return newList;
        }

        public List<float> ListeElemanlariniSetEt(dynamic Sistem, int begIndex, int barCount, List<float> list, float lowerLimit, float upperLimit, float newValue = 0.0f)
        {
            List<float> newList = Sistem.Liste(list.Count, 0);

            for (int i = begIndex; i < barCount; i++)
            {
                if (lowerLimit <= list[i] && list[i] <= upperLimit)
                {
                    newList[i] = newValue;
                }
                else
                {
                    newList[i] = list[i];
                }
            }

            return newList;
        }

        public List<double> ListeElemanlariniSetEt(dynamic Sistem, int begIndex, int barCount, List<double> list, double lowerLimit, double upperLimit, double newValue = 0.0)
        {
            List<double> newList = new List<double>(list.Count);

            for (int i = begIndex; i < barCount; i++)
            {
                if (lowerLimit <= list[i] && list[i] <= upperLimit)
                {
                    newList[i] = newValue;
                }
                else
                {
                    newList[i] = list[i];
                }
            }

            return newList;
        }

        public List<T> GetNewList<T>(dynamic Sistem, int Count, T Value)
        {
            List<T> list = new List<T>( new T[Count] );

            for (int i = 0; i < Count; i++)
            {
                list[i] = Value;
            }

            return list;
        }

        public List<float> GetZeroLevelList(dynamic Sistem, int Count)
        {
            //List<float> ZeroLevel = Sistem.Liste(Count, 0);
            List<float> ZeroLevel = GetNewList(Sistem, Count, 0.0f);

            return ZeroLevel;
        }

        public List<T> GetNewList<T>(dynamic Sistem, List<T> SrcList)
        {
            int Count = SrcList.Count;

            List<T> list = new List<T>( new T[Count] );

            for (int i = 0; i < Count; i++)
            {
                list[i] = SrcList[i];
            }

            return list;
        }

        #endregion ListOperations

        #region Kairi

        public List<float> GetDistanceListAsPercentage(dynamic Sistem, List<float> Source, List<float> High, List<float> Low)
        {
            var Distance = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                Distance[i] = (Source[i] - Low[i]) * 100f / (High[i] - Low[i]);
            }

            return Distance;
        }

        public List<float> GetDistanceListAsPercentage(dynamic Sistem, List<float> Src, List<float> Ref)
        {
            var Distance = Sistem.Liste(Sistem.BarSayisi, 0);

            for (int i = 0; i < Sistem.BarSayisi; i++)
            {
                Distance[i] = (Src[i] - Ref[i]) / Ref[i] * 100;
            }

            return Distance;
        }

        public List<float> GetKairi(dynamic Sistem, List<float> Src, List<float> Ref)
        {
            return GetDistanceListAsPercentage(Sistem, Src, Ref);
        }

        public List<float> GetKairi(dynamic Sistem, List<float> Src, int MaPeriyot, string MaYontem = "Simple")
        {
            return GetDistanceListAsPercentage(Sistem, Src, Sistem.MA(Src, MaYontem, MaPeriyot));
        }

        #endregion Kairi

        #region SinyalSiralamalari

        public List<float> SortTheSignals2(dynamic Sistem)
        {
            var Distance = Sistem.Liste(Sistem.BarSayisi, 0);

            return Distance;
        }

        public string GetTheListSortStatus(dynamic Sistem, int i, List<double> List1, List<double> List2, List<double> List3 = null, List<double> List4 = null, List<double> List5 = null, List<double> List6 = null, List<double> List7 = null, List<double> List8 = null, List<double> List9 = null, List<double> List10 = null)
        {
            string res = "01-02-03-04-05-06-07-08-09-10";

            Dictionary<string, double> statusDict = new Dictionary<string, double>();

            statusDict.Add("01", List1[i]);
            statusDict.Add("02", List2[i]);
            if (List3 != null)
                statusDict.Add("03", List3[i]);
            if (List4 != null)
                statusDict.Add("04", List4[i]);
            if (List5 != null)
                statusDict.Add("05", List5[i]);
            if (List6 != null)
                statusDict.Add("06", List6[i]);
            if (List7 != null)
                statusDict.Add("07", List7[i]);
            if (List8 != null)
                statusDict.Add("08", List8[i]);
            if (List9 != null)
                statusDict.Add("09", List9[i]);
            if (List10 != null)
                statusDict.Add("10", List10[i]);

            var sortedStatusDict = from entry in statusDict orderby entry.Value ascending select entry;

            foreach (var entry in sortedStatusDict)
            {
                res = res + entry.Key + "-";
            }

            return res;
        }

        #endregion SinyalSiralamalari
        
        public dynamic CreateLevels(dynamic Sistem, int MaxLevelNumber = 100, bool UseFloatingKeys = false)
        {
            int BarCount = Sistem.BarSayisi;
                        
            dynamic results = null;
            
            if (UseFloatingKeys)
            {
                Dictionary<float, List<float> > Levels = new Dictionary<float, List<float> >();
                
                for (int i = 0; i <= MaxLevelNumber; i++) {
                    float key = Convert.ToSingle(i);
                    Levels[key] = Sistem.Liste(BarCount, i);
                } 
                
                results = Levels; 
            }
            else
            {
                Dictionary<int, List<float> > Levels = new Dictionary<int, List<float> >();
                
                for (int i = 0; i <= MaxLevelNumber; i++) {
                    int key = i;
                    Levels[key] = Sistem.Liste(BarCount, i);
                } 
                
                results = Levels; 
            }
            
            return results;
        }
                
    }