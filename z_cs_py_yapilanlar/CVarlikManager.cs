public class CVarlikManager
{
    public float IlkBakiyeFiyat   { get; set; }
    public float IlkBakiyePuan    { get; set; }
    public float KomisyonCarpan   { get; set; }
    public bool  KomisyonuDahilEt { get; set; }
    public float KaymaMiktari     { get; set; }
    public bool  KaymayiDahilEt   { get; set; }
    public int KomisyonVarlikAdedSayisi { get; set; }
    public int VarlikAdedSayisi         { get; set; }
    public int VarlikAdedCarpani        { get; set; }
    public int KontratSayisi            { get; set; }
    public int HisseSayisi              { get; set; }

    public string GetiriFiyatTipi       { get; set; }

    private int MarketIndex = 0;

    ~CVarlikManager()
    {

    }

    public CVarlikManager()
    {

    }

    public CVarlikManager Initialize(dynamic Sistem)
    {
        SetKomisyonParams(Sistem);
        SetKaymaParams(Sistem);
        SetBakiyeParams(Sistem);

        Reset(Sistem);

        return this;
    }

    public CVarlikManager Reset(dynamic Sistem)
    {
        return this;
    }

    public CVarlikManager SetKomisyonParams(dynamic Sistem, float KomisyonCarpan = 3.0f)
    {
        this.KomisyonCarpan = KomisyonCarpan;
        this.KomisyonuDahilEt = KomisyonCarpan == 0.0f ? false : true;

        return this;
    }

    public CVarlikManager SetKaymaParams(dynamic Sistem, float KaymaMiktari  = 0.0f)
    {
        this.KaymaMiktari = KaymaMiktari;
        this.KaymayiDahilEt = KaymaMiktari == 0.0f ? false : true;

        return this;
    }

    public CVarlikManager SetBakiyeParams(dynamic Sistem, float IlkBakiye = 100000.0f, float IlkBakiyePuan = 0.0f)
    {
        this.IlkBakiyeFiyat = IlkBakiye;
        this.IlkBakiyePuan = IlkBakiyePuan;

        return this;
    }

    // Bist Hisse
    public CVarlikManager SetKontratParamsBistHisse(dynamic Sistem, int HisseSayisi = 1000, int VarlikAdedCarpani = 1)
    {
        this.HisseSayisi = HisseSayisi;
        this.VarlikAdedCarpani = VarlikAdedCarpani;
        this.VarlikAdedSayisi = HisseSayisi * VarlikAdedCarpani;
        this.KomisyonVarlikAdedSayisi = HisseSayisi;
        this.GetiriFiyatTipi = "TL";

        return this;
    }

    // Viop Endeks
    public CVarlikManager SetKontratParamsViopEndeks(dynamic Sistem, int KontratSayisi = 1, int VarlikAdedCarpani = 10)
    {
        this.KontratSayisi = KontratSayisi;
        this.VarlikAdedCarpani = VarlikAdedCarpani;
        this.VarlikAdedSayisi = KontratSayisi * VarlikAdedCarpani;
        this.KomisyonVarlikAdedSayisi = KontratSayisi;
        this.GetiriFiyatTipi = "TL";

        return this;
    }

    // Viop Hisse
    public CVarlikManager SetKontratParamsViopHisse(dynamic Sistem, int KontratSayisi = 1, int VarlikAdedCarpani = 100)
    {
        this.KontratSayisi = KontratSayisi;
        this.VarlikAdedCarpani = VarlikAdedCarpani;
        this.VarlikAdedSayisi = KontratSayisi * VarlikAdedCarpani;
        this.KomisyonVarlikAdedSayisi = KontratSayisi;
        this.GetiriFiyatTipi = "TL";

        return this;
    }

    // Viop Dolar
    public CVarlikManager SetKontratParamsViopDolar(dynamic Sistem, int KontratSayisi = 1, int VarlikAdedCarpani = 1000)
    {
        this.KontratSayisi = KontratSayisi;
        this.VarlikAdedCarpani = VarlikAdedCarpani;
        this.VarlikAdedSayisi = KontratSayisi * VarlikAdedCarpani;
        this.KomisyonVarlikAdedSayisi = KontratSayisi;
        this.GetiriFiyatTipi = "TL";

        return this;
    }

    // Viop Euro
    public CVarlikManager SetKontratParamsViopEuro(dynamic Sistem, int KontratSayisi = 1, int VarlikAdedCarpani = 1000)
    {
        this.KontratSayisi = KontratSayisi;
        this.VarlikAdedCarpani = VarlikAdedCarpani;
        this.VarlikAdedSayisi = KontratSayisi * VarlikAdedCarpani;
        this.KomisyonVarlikAdedSayisi = KontratSayisi;
        this.GetiriFiyatTipi = "TL";

        return this;
    }

    // Viop Gram Altin
    public CVarlikManager SetKontratParamsViopGramAltin(dynamic Sistem, int KontratSayisi = 1, int VarlikAdedCarpani = 1)
    {
        this.KontratSayisi = KontratSayisi;
        this.VarlikAdedCarpani = VarlikAdedCarpani;
        this.VarlikAdedSayisi = KontratSayisi * VarlikAdedCarpani;
        this.KomisyonVarlikAdedSayisi = KontratSayisi;
        this.GetiriFiyatTipi = "TL";

        return this;
    }
/*
    // Viop Ons Altin - Yapilacak
    public CVarlikManager SetKontratParamsViopOnsAltin(dynamic Sistem, int KontratSayisi = 1, int VarlikAdedCarpani = 1)
    {
        this.KontratSayisi = KontratSayisi;
        this.VarlikAdedCarpani = VarlikAdedCarpani;
        this.VarlikAdedSayisi = KontratSayisi * VarlikAdedCarpani;
        this.KomisyonVarlikAdedSayisi = KontratSayisi;
        this.GetiriFiyatTipi = "...";

        return this;
    }

    // Viop Ons Gumus - Yapilacak
    public CVarlikManager SetKontratParamsViopOnsGumus(dynamic Sistem, int KontratSayisi = 1, int VarlikAdedCarpani = 1)
    {
        this.KontratSayisi = KontratSayisi;
        this.VarlikAdedCarpani = VarlikAdedCarpani;
        this.VarlikAdedSayisi = KontratSayisi * VarlikAdedCarpani;
        this.KomisyonVarlikAdedSayisi = KontratSayisi;
        this.GetiriFiyatTipi = "...";

        return this;
    }
*/
    // Banka Dolar
    public CVarlikManager SetKontratParamsBankaDolar(dynamic Sistem, int KontratSayisi = 1, int VarlikAdedCarpani = 1)
    {
        this.KontratSayisi = KontratSayisi;
        this.VarlikAdedCarpani = VarlikAdedCarpani;
        this.VarlikAdedSayisi = KontratSayisi * VarlikAdedCarpani;
        this.KomisyonVarlikAdedSayisi = KontratSayisi;
        this.GetiriFiyatTipi = "TL";

        return this;
    }

    // Banka Euro
    public CVarlikManager SetKontratParamsBankaEuro(dynamic Sistem, int KontratSayisi = 1, int VarlikAdedCarpani = 1)
    {
        this.KontratSayisi = KontratSayisi;
        this.VarlikAdedCarpani = VarlikAdedCarpani;
        this.VarlikAdedSayisi = KontratSayisi * VarlikAdedCarpani;
        this.KomisyonVarlikAdedSayisi = KontratSayisi;
        this.GetiriFiyatTipi = "TL";

        return this;
    }

    // Banka Gram Altin
    public CVarlikManager SetKontratParamsBankaGramAltin(dynamic Sistem, int KontratSayisi = 1, int VarlikAdedCarpani = 1)
    {
        this.KontratSayisi = KontratSayisi;
        this.VarlikAdedCarpani = VarlikAdedCarpani;
        this.VarlikAdedSayisi = KontratSayisi * VarlikAdedCarpani;
        this.KomisyonVarlikAdedSayisi = KontratSayisi;
        this.GetiriFiyatTipi = "TL";

        return this;
    }

    // Fx Onst Altin (Micro - 0.01 Lot ve katlari)
    public CVarlikManager SetKontratParamsFxOnsAltinMicro(dynamic Sistem, int KontratSayisi = 1, int VarlikAdedCarpani = 1)
    {
        this.KontratSayisi = KontratSayisi;
        this.VarlikAdedCarpani = VarlikAdedCarpani;
        this.VarlikAdedSayisi = KontratSayisi * VarlikAdedCarpani;
        this.KomisyonVarlikAdedSayisi = KontratSayisi;
        this.GetiriFiyatTipi = "$";

        return this;
    }
/*
    // Fx Parite (Micro - 0.01 Lot ve katlari) - Yapilacak
    public CVarlikManager SetKontratParamsFxPariteMicro(dynamic Sistem, int KontratSayisi = 1, int VarlikAdedCarpani = 1)
    {
        this.KontratSayisi = KontratSayisi;
        this.VarlikAdedCarpani = VarlikAdedCarpani;
        this.VarlikAdedSayisi = KontratSayisi * VarlikAdedCarpani;
        this.KomisyonVarlikAdedSayisi = KontratSayisi;
        this.GetiriFiyatTipi = "$";

        return this;
    }

    // Fx Endeks (Micro - 0.01 Lot ve katlari) - Yapilacak
    public CVarlikManager SetKontratParamsFxEndeksMicro(dynamic Sistem, int KontratSayisi = 1, int VarlikAdedCarpani = 1)
    {
        this.KontratSayisi = KontratSayisi;
        this.VarlikAdedCarpani = VarlikAdedCarpani;
        this.VarlikAdedSayisi = KontratSayisi * VarlikAdedCarpani;
        this.KomisyonVarlikAdedSayisi = KontratSayisi;
        this.GetiriFiyatTipi = "$";

        return this;
    }
*/
/*
        public CVarlikManager SetParamsLike(dynamic Sistem, string SembolKodu, bool KomisyonuDahilEt = true)
        {
            //iDeal SİSTEM, ROBOT, ALGO Yazım için Yardım Kılavuzu
            //var Sembol = "FX'EURUSD";
            //var Sembol = "IMKBH'GARAN";
            //var Sembol1 = "VIP'VIP-X030";
            //var Sembol2 = "IMKBX'XU030";
            //var Sembol3 = "VIP'VIP-X030-T";
            //var Sembol = "IMKBH'ISCTR";
            //var Sembol = "VIP'VIP-THYAO"; // ÖRNEK SEMBOL KODU
            //var EmirSembol =  "VIP'F_THYAO1118"; // BURAYA SEMBOLÜNÜZÜN AÇIK YAZILIŞI
            //var Sembol = "VIP'F_XU0300813S0";
            //var Sembol1 = "FX'USDTRY";
            //var Sembol2 = "FX'EURUSD";
            //Sistem.SonFiyat("IMKBX'XU100"),
            //var GrafikSembolu  = "VIP'VIP-X030-T";
            //var EmirSembol1    = "VIP'F_XU0300622";

            int KontratSayisi = 10; int VarlikAdedCarpani = 10; float KomisyonCarpan = 3f; float KaymaMiktari = 0f; float IlkBakiye = 100000.0f;

            List<string> SembolKoduList = new List<string>(); for (int i = 0; i < 50; i++) SembolKoduList.Add("");
            SembolKoduList[0] = "VIP'VIP-X030-T";
            SembolKoduList[1] = "VIP'VIP-X030";
            SembolKoduList[2] = "VIP-X030-T";
            SembolKoduList[3] = "VIP-X030";
            SembolKoduList[4] = "VIP'VIP-THYAO";
            SembolKoduList[5] = "VIP-THYAO";
            SembolKoduList[6] = "VIP'VIP-USD";
            SembolKoduList[7] = "VIP-USD";
            SembolKoduList[8] = "VIP'VIP-GOZ-T";
            SembolKoduList[9] = "VIP-GOZ-T";
            SembolKoduList[10] = "VIP'VIP-GLD";
            SembolKoduList[11] = "VIP-GLD";
            SembolKoduList[12] = "IMKBH'THYAO";
            SembolKoduList[13] = "THYAO";
            SembolKoduList[14] = "FX-EURUSD";
            SembolKoduList[15] = "FX-EURUSD";
            SembolKoduList[16] = "SPOT-USDTRY";
            SembolKoduList[17] = "SPOT-USDTRY";
            SembolKoduList[18] = "SPOT-ALTIN";
            SembolKoduList[19] = "SPOT-ALTIN";

            string sembolKodu = "";
            for (int i = 0; i < SembolKoduList.Count; i++)
            {
                if (SembolKoduList[i] == SembolKodu) {
                    sembolKodu = SembolKodu;
                    break;
                }
            }

            if (sembolKodu != "")
            {
                // secim = 0 : Viop X030-T, secim = 1 : Viop X030
                if (sembolKodu == SembolKoduList[0] || sembolKodu == SembolKoduList[1] || sembolKodu == SembolKoduList[2] || sembolKodu == SembolKoduList[3])
                {
                    KontratSayisi = 10;
                    VarlikAdedCarpani = 10;
                    KomisyonCarpan = 3f;
                    KaymaMiktari = 0f;
                    IlkBakiye = 100000.0f;
                }
                // secim = 2 : Viop THYAO, GARAN, ISCTR
                else if (sembolKodu == SembolKoduList[4] || sembolKodu == SembolKoduList[5])
                {
                    KontratSayisi = 10;
                    VarlikAdedCarpani = 100;
                    KomisyonCarpan = 1f;
                    KaymaMiktari = 0f;
                    IlkBakiye = 100000.0f;
                }
                // secim = 3 : viop USDTRY, EURTRY, EURUSD
                else if (sembolKodu == SembolKoduList[6] || sembolKodu == SembolKoduList[7])
                {
                    KontratSayisi = 10;
                    VarlikAdedCarpani = 1000;
                    KomisyonCarpan = 1f;
                    KaymaMiktari = 0f;
                    IlkBakiye = 100000.0f;
                }
                // secim = 4 : viop Ons Altın, Ons Gumus
                else if (sembolKodu == SembolKoduList[8] || sembolKodu == SembolKoduList[9])
                {
                    KontratSayisi = 10;
                    VarlikAdedCarpani = 1;
                    KomisyonCarpan = 1f;
                    KaymaMiktari = 0f;
                    IlkBakiye = 100000.0f;
                }
                // secim = 5 : viop Gram Altın
                else if (sembolKodu == SembolKoduList[10] || sembolKodu == SembolKoduList[11])
                {
                    KontratSayisi = 10;
                    VarlikAdedCarpani = 1;
                    KomisyonCarpan = 1f;
                    KaymaMiktari = 0f;
                    IlkBakiye = 100000.0f;
                }
                // secim = 8 : Hisse
                else if (sembolKodu == SembolKoduList[12] || sembolKodu == SembolKoduList[13])
                {
                    KontratSayisi = 1000;
                    VarlikAdedCarpani = 1;
                    KomisyonCarpan = 1f;
                    KaymaMiktari = 0f;
                    IlkBakiye = 100000.0f;
                }
                // secim = 9 : Fx
                else if (sembolKodu == SembolKoduList[14] || sembolKodu == SembolKoduList[15])
                {

                }
                // secim = 10 : spot USDTRY, EURTRY, EURUSD
                else if (sembolKodu == SembolKoduList[16] || sembolKodu == SembolKoduList[17])
                {

                }
                // secim = 4 : spot Altın, spot Gumus
                else if (sembolKodu == SembolKoduList[18] || sembolKodu == SembolKoduList[19])
                {

                }
            }
*/


/*
// secim = 0 : Viop X030-T
// secim = 1 : Viop X030
// secim = 2 : Viop THYAO, GARAN, ISCTR
// secim = 3 : viop USDTRY, EURTRY, EURUSD
// secim = 4 : viop Ons Altın, Ons Gumus
// secim = 5 : viop Gram Altın
// secim = 8 : Hisse
// secim = 9 : Fx

var paraCinsi = "TL";
if (secim == 4) paraCinsi = "$";

var sozlesmeCinsi = "Kontrat";
if (secim == 8) sozlesmeCinsi = "Hisse";
*/


        /*
        public CVarlikManager SetParams(dynamic Sistem, int KontratSayisi = 1, int VarlikAdedCarpani = 1, float KomisyonCarpan = 0.0f, float KaymaMiktari  = 0.0f, float IlkBakiye = 100000.0f)
        {
            SetKontratParams(Sistem, KontratSayisi, VarlikAdedCarpani);
            SetKomisyonParams(Sistem, KomisyonCarpan);
            SetKaymaParams(Sistem, KaymaMiktari);
            SetBakiyeParams(Sistem, IlkBakiye);

            return this;
        }

        public CVarlikManager SetKontratParams(dynamic Sistem, int KontratSayisi = 1, int VarlikAdedCarpani = 10)
        {
            this.KontratSayisi = KontratSayisi;
            this.VarlikAdedSayisi = KontratSayisi * VarlikAdedCarpani;
            this.KomisyonVarlikAdedSayisi = KontratSayisi;

            return this;
        }
        */

        /* Kullanim Ornegi
        // Bist Hisse
        myVarlik.SetKontratParamsBistHisse(Sistem, HisseSayisi = 1000, VarlikAdedCarpani = 1).SetKomisyonParams(Sistem, KomisyonCarpan = 1.0f);
        // Viop Endeks
        myVarlik.SetKontratParamsViopEndeks(Sistem, KontratSayisi = 10, VarlikAdedCarpani = 10).SetKomisyonParams(Sistem, KomisyonCarpan = 3.0f);
        // Viop Hisse
        myVarlik.SetKontratParamsViopHisse(Sistem, KontratSayisi = 1, VarlikAdedCarpani = 100).SetKomisyonParams(Sistem, KomisyonCarpan = 3.0f);
        // Viop Dolar
        myVarlik.SetKontratParamsViopDolar(Sistem, KontratSayisi = 1, VarlikAdedCarpani = 1000).SetKomisyonParams(Sistem, KomisyonCarpan = 3.0f);
        // Viop Euro
        myVarlik.SetKontratParamsViopEuro(Sistem, KontratSayisi = 1, VarlikAdedCarpani = 1000).SetKomisyonParams(Sistem, KomisyonCarpan = 3.0f);
        // Viop Gram Altin
        myVarlik.SetKontratParamsViopGramAltin(Sistem, KontratSayisi = 1, VarlikAdedCarpani = 1).SetKomisyonParams(Sistem, KomisyonCarpan = 3.0f);
        // Banka Dolar
        myVarlik.SetKontratParamsBankaDolar(Sistem, KontratSayisi = 1, VarlikAdedCarpani = 1).SetKomisyonParams(Sistem, KomisyonCarpan = 3.0f);
        // Banka Euro
        myVarlik.SetKontratParamsBankaEuro(Sistem, KontratSayisi = 1, VarlikAdedCarpani = 1).SetKomisyonParams(Sistem, KomisyonCarpan = 3.0f);
        // Banka Gram Altin
        myVarlik.SetKontratParamsBankaGramAltin(Sistem, KontratSayisi = 1, VarlikAdedCarpani = 1).SetKomisyonParams(Sistem, KomisyonCarpan = 3.0f);
        // Fx Onst Altin (Micro - 0.01 Lot ve katlari)
        myVarlik.SetKontratParamsFxOnsAltinMicro(Sistem, KontratSayisi = 1, VarlikAdedCarpani = 1).SetKomisyonParams(Sistem, KomisyonCarpan = 0.0f);
        */
}