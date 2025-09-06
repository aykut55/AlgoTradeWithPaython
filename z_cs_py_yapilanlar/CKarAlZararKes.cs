public class CKarAlZararKes
{
    CTrader Trader = null;

    ~CKarAlZararKes()
    {

    }

    public CKarAlZararKes()
    {

    }

    public CKarAlZararKes Initialize(dynamic Sistem, CTrader Trader)
    {
        this.Trader = Trader;

        return this;
    }

    public CKarAlZararKes Reset(dynamic Sistem)
    {
        return this;
    }

    // Type 1 (Kar Al, Zarar Kes)
    public int KarAlYuzdeHesapla(dynamic Sistem, int BarIndex, float KarAlYuzdesi, List<float> Ref)
    {
        int result = 0;

        int i = BarIndex;

        if (Trader.Flags.KarAlYuzdeHesaplaEnabled)
        {
            Trader.Lists.KarAlList[i] = Sistem.KarAlYuzde(KarAlYuzdesi, i);
            if (Trader.Lists.KarAlList[i] == 0) Trader.Lists.KarAlList[i] = Ref[i];

            // result = 1  :  Alis yonunda KarAl Aktif oldu
            // result = -1 : Satis yonunda KarAl Aktif oldu
                 if ( Trader.IsSonYonA(Sistem) &&  (Ref[i] > Trader.Lists.KarAlList[i]) ) result = 1;
            else if ( Trader.IsSonYonS(Sistem) &&  (Ref[i] < Trader.Lists.KarAlList[i]) ) result = -1;
        }

        return result;
    }

    public int IzleyenStopYuzdeHesapla(dynamic Sistem, int BarIndex, float IzleyenStopYuzdesi, List<float> Ref)
    {
        int result = 0;

        int i = BarIndex;

        if (Trader.Flags.IzleyenStopYuzdeHesaplaEnabled)
        {
            Trader.Lists.IzleyenStopList[i] = Sistem.IzleyenStopYuzde(IzleyenStopYuzdesi, i);
            if (Trader.Lists.IzleyenStopList[i] == 0) Trader.Lists.IzleyenStopList[i] = Ref[i];

            // result = 1  :  Alis yonunda StopOl Aktif oldu
            // result = -1 : Satis yonunda StopOl Aktif oldu
                 if ( Trader.IsSonYonA(Sistem) &&  (Ref[i] < Trader.Lists.IzleyenStopList[i]) ) result = 1;
            else if ( Trader.IsSonYonS(Sistem) &&  (Ref[i] > Trader.Lists.IzleyenStopList[i]) ) result = -1;
        }

        return result;
    }

    public int KarAlYuzdeHesapla(dynamic Sistem, int BarIndex, float KarAlYuzdesi = 2.0f)
    {
        return KarAlYuzdeHesapla(Sistem, BarIndex, KarAlYuzdesi, Trader.Close);
    }

    public int IzleyenStopYuzdeHesapla(dynamic Sistem, int BarIndex, float IzleyenStopYuzdesi = 1.0f)
    {
        return IzleyenStopYuzdeHesapla(Sistem, BarIndex, IzleyenStopYuzdesi, Trader.Close);
    }

    // Type 2 (Kar Al, Zarar Kes)
    public int SonFiyataGoreKarAlYuzdeHesapla(dynamic Sistem, int BarIndex, float KarAlYuzdesi = 2f)
    {
        int result = 0;

        int i = BarIndex;

        if (Trader.Flags.KarAlYuzdeHesaplaEnabled)
        {
            // result = 1  :  Alis yonunda KarAl Aktif oldu
            // result = -1 : Satis yonunda KarAl Aktif oldu
                 if ( Trader.IsSonYonA(Sistem) &&  ( Trader.Close[i] > Trader.Signals.SonFiyat * (1.0 + KarAlYuzdesi * 0.01f) ) ) result = 1;
            else if ( Trader.IsSonYonS(Sistem) &&  ( Trader.Close[i] < Trader.Signals.SonFiyat * (1.0 - KarAlYuzdesi * 0.01f) ) ) result = -1;
        }

        return result;
    }

    public int SonFiyataGoreZararKesYuzdeHesapla(dynamic Sistem, int BarIndex, float ZararKesYuzdesi = -1f)
    {
        int result = 0;

        int i = BarIndex;

        float ZararKesYuzdesi_ = -1f * ZararKesYuzdesi;

        if (Trader.Flags.ZararKesYuzdeHesaplaEnabled)
        {
            // result = 1  :  Alis yonunda StopOl Aktif oldu
            // result = -1 : Satis yonunda StopOl Aktif oldu
                 if ( Trader.IsSonYonA(Sistem) &&  ( Trader.Close[i] < Trader.Signals.SonFiyat * (1.0 - ZararKesYuzdesi_ * 0.01f) )  ) result = 1;
            else if ( Trader.IsSonYonS(Sistem) &&  ( Trader.Close[i] > Trader.Signals.SonFiyat * (1.0 + ZararKesYuzdesi_ * 0.01f) )  ) result = -1;
        }

        return result;
    }

    // Type 3 (Kar Al, Zarar Kes)
    public int SonFiyataGoreKarAlYuzdeHesapla(dynamic Sistem, int BarIndex, int SeviyeBas = 2, int SeviyeSon = 10, float Carpan = 0.01f)
    {
        int result = 0;

        int i = BarIndex;

        if (Trader.Flags.KarAlYuzdeHesaplaEnabled)
        {
            CUtils myUtils = new CUtils();

            bool _karAl = false;

            if (Trader.IsSonYonA(Sistem))
            {
                for (int m = SeviyeBas; m < SeviyeSon; m++)
                {
                    _karAl = _karAl || myUtils.AsagiKesti(Sistem, i, Trader.Close, Trader.Signals.SonFiyat * (1.0f + m * Carpan) );
                    if (_karAl)
                        break;
                }

                if (_karAl) result = 1;
            }

            else if (Trader.IsSonYonS(Sistem))
            {
                for (int m = SeviyeBas; m < SeviyeSon; m++)
                {
                    _karAl = _karAl || myUtils.YukariKesti(Sistem, i, Trader.Close, Trader.Signals.SonFiyat * (1.0f - m * Carpan) );
                    if (_karAl)
                        break;
                }

                if (_karAl) result = -1;
            }

            // result = 1  :  Alis yonunda KarAl Aktif oldu
            // result = -1 : Satis yonunda KarAl Aktif oldu
        }

        return result;
    }

    public int SonFiyataGoreZararKesYuzdeHesapla(dynamic Sistem, int BarIndex, int SeviyeBas = -2, int SeviyeSon = -10, float Carpan = 0.01f)
    {
        int result = 0;

        int i = BarIndex;

        int SeviyeBas_ = -1 * SeviyeBas;

        int SeviyeSon_ = -1 * SeviyeSon;

        if (Trader.Flags.ZararKesYuzdeHesaplaEnabled)
        {
            CUtils myUtils = new CUtils();

            bool _zararKes = false;

            if (Trader.IsSonYonA(Sistem))
            {
                for (int m = SeviyeBas_; m < SeviyeSon_; m++)
                {
                    _zararKes = _zararKes || myUtils.AsagiKesti(Sistem, i, Trader.Close, Trader.Signals.SonFiyat * (1.0f - m * Carpan) );
                    if (_zararKes)
                        break;
                }

                if (_zararKes) result = 1;
            }

            else if (Trader.IsSonYonS(Sistem))
            {
                for (int m = SeviyeBas_; m < SeviyeSon_; m++)
                {
                    _zararKes = _zararKes || myUtils.YukariKesti(Sistem, i, Trader.Close, Trader.Signals.SonFiyat * (1.0f + m * Carpan) );
                    if (_zararKes)
                        break;
                }

                if (_zararKes) result = -1;
            }

            // result = 1  :  Alis yonunda StopOl Aktif oldu
            // result = -1 : Satis yonunda StopOl Aktif oldu
        }

        return result;
    }

    // Type 4 (Kar Al, Zarar Kes)
    public int SonFiyataGoreKarAlSeviyeHesapla(dynamic Sistem, int BarIndex, float KarAlSeviyesi = 2000.0f)
    {
        int result = 0;

        int i = BarIndex;

        if (Trader.Flags.KarAlSeviyeHesaplaEnabled)
        {
            // result = 1  :  KarAl Aktif oldu
            CUtils myUtils = new CUtils();
            result = myUtils.YukariKesti(Sistem, i, Trader.Lists.KarZararFiyatList, KarAlSeviyesi) ? 1 : 0;
            // veya
            // result = ( KarZararFiyatList[i - 1] > KarAlSeviyesi && KarZararFiyatList[i] <= KarAlSeviyesi ) ? 1 : 0;
        }

        return result;
    }

    public int SonFiyataGoreZararKesSeviyeHesapla(dynamic Sistem, int BarIndex, float ZararKesSeviyesi = -1000.0f)
    {
        int result = 0;

        int i = BarIndex;

        if (Trader.Flags.ZararKesSeviyeHesaplaEnabled)
        {
            // result = 1  :  ZararKes Aktif oldu
            CUtils myUtils = new CUtils();
            result = myUtils.AsagiKesti(Sistem, i, Trader.Lists.KarZararFiyatList, ZararKesSeviyesi) ? 1 : 0;
            // veya
            //result = ( KarZararFiyatList[i - 1] > ZararKesSeviyesi && KarZararFiyatList[i] <= ZararKesSeviyesi ) ? 1 : 0;
        }

        return result;
    }

    // Type 5 (Kar Al, Zarar Kes)
    public int SonFiyataGoreKarAlSeviyeHesapla(dynamic Sistem, int BarIndex, int SeviyeBas = 5, int SeviyeSon = 50, int Carpan = 1000)
    {
        int result = 0;

        int i = BarIndex;

        if (Trader.Flags.KarAlSeviyeHesaplaEnabled)
        {
            CUtils myUtils = new CUtils();

            bool _karAl = false;
            for (int m = SeviyeBas; m < SeviyeSon; m++)
            {
                _karAl = _karAl || myUtils.AsagiKesti(Sistem, i, Trader.Lists.KarZararFiyatList, m*Carpan);
                if (_karAl)
                    break;
            }

            if (_karAl) result = 1;     // :  KarAl Aktif oldu
        }

        return result;
    }

    public int SonFiyataGoreZararKesSeviyeHesapla(dynamic Sistem, int BarIndex, int SeviyeBas = -1, int SeviyeSon = -10, int Carpan = 1000)
    {
        int result = 0;

        int i = BarIndex;

        if (Trader.Flags.ZararKesSeviyeHesaplaEnabled)
        {
            CUtils myUtils = new CUtils();

            bool _zararKes = false;
            for (int m = SeviyeBas; m > SeviyeSon; m--)
            {
                _zararKes = _zararKes || myUtils.AsagiKesti(Sistem, i, Trader.Lists.KarZararFiyatList, m*Carpan);
                if (_zararKes)
                    break;
            }

            if (_zararKes) result = 1;     // :  ZararKes Aktif oldu
        }

        return result;
    }

//    //Kullanim Ornekleri
//
//    //Yontem 1
//    if (myTrader.KarAlEnabled == true)
//        myTrader.KarAl = myTrader.KarAlYuzdeHesapla(Sistem, i, 0.7f) != 0 ? true : false;
//
//    if (myTrader.ZararKesEnabled == true)
//        myTrader.ZararKes = myTrader.IzleyenStopYuzdeHesapla(Sistem, i, 0.3f) != 0 ? true : false;
//
//    //Yontem 2
//    if (myTrader.KarAlEnabled == true)
//        myTrader.KarAl = myTrader.SonFiyataGoreKarAlYuzdeHesapla(Sistem, i, 1.0f) != 0 ? true : false;
//
//    if (myTrader.ZararKesEnabled == true)
//        myTrader.ZararKes = myTrader.SonFiyataGoreZararKesYuzdeHesapla(Sistem, i, -0.5f) != 0 ? true : false;
//
//    //Yontem 3
//    if (myTrader.KarAlEnabled == true)
//        myTrader.KarAl = myTrader.SonFiyataGoreKarAlYuzdeHesapla(Sistem, i, 2, 10) != 0 ? true : false;
//
//    if (myTrader.ZararKesEnabled == true)
//        myTrader.ZararKes = myTrader.SonFiyataGoreZararKesYuzdeHesapla(Sistem, i, -2, -10) != 0 ? true : false;
//
//    //Yontem 4
//    if (myTrader.KarAlEnabled == true)
//        myTrader.KarAl = myTrader.SonFiyataGoreKarAlSeviyeHesapla(Sistem, i, 2000f) != 0 ? true : false;
//
//    if (myTrader.ZararKesEnabled == true)
//        myTrader.ZararKes = myTrader.SonFiyataGoreZararKesSeviyeHesapla(Sistem, i, -1500) != 0 ? true : false;
//
//    //Yontem 5
//    if (myTrader.KarAlEnabled == true)
//        myTrader.KarAl = myTrader.SonFiyataGoreKarAlSeviyeHesapla(Sistem, i, 5, 50) != 0 ? true : false;
//
//    if (myTrader.ZararKesEnabled == true)
//        myTrader.ZararKes = myTrader.SonFiyataGoreZararKesSeviyeHesapla(Sistem, i, -3, -5) != 0 ? true : false;
}