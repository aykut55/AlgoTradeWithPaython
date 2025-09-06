public class CBirlesikSistemManager
    {
        public float IlkBakiyeFiyat = 100000.0f;
        public float KomisyonCarpan = 3.0f;
        public int VarlikAdedCarpani = 1;

        public float SonBakiyeFiyat = 0.0f;
        public float SonBakiyeFiyatNet = 0.0f;
        public float Komisyon = 0.0f;
        public float GetiriFiyat = 0.0f;
        public float GetiriFiyatNet = 0.0f;

        ~CBirlesikSistemManager()
        {
        }

        public CBirlesikSistemManager()
        {
        }

        public int Initialize(dynamic Sistem)
        {
            Reset(Sistem);

            return 0;
        }

        public int Reset(dynamic Sistem)
        {
            return 0;
        }

        public void SistemleriCalistir(dynamic Sistem, List<string> ParametreList)
        {
            var str = "";
            int colNum = 0; int rowNum = 0; int panelNo = 1;
            int[] colNums = { 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160 };
            int[] rowNums = { 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400 };
            var Renk = Color.Gold;
            var RenkPoz = Color.Gold;

            var V = Sistem.GrafikVerileri;
            var C = Sistem.GrafikFiyatOku(V, "Kapanis");

            for (int i = 1; i < V.Count; i++)
            {
                if (V[i].Date.Day != V[i - 1].Date.Day) Sistem.DikeyCizgiEkle(i, Color.DimGray, 2, 2);
            }

            var SistemList = new List<string>();
            var PeriyotList = new List<string>();
            var LotList = new List<float>();

            for (int i = 0; i < ParametreList.Count; i++)
            {
                var FieldArray = ParametreList[i].Split(',');

                SistemList.Add(FieldArray[0].Trim());

                PeriyotList.Add(FieldArray[1].Trim());

                LotList.Add(Convert.ToSingle(FieldArray[2].Trim()));
            }

            var ViopData = Sistem.GrafikVerileri;
            var TarihDictionary = new Dictionary<DateTime, int>();

            for (int i = 0; i < ViopData.Count; i++)
            {
                TarihDictionary[ViopData[i].Date] = i;
            }

            List<List<string>> Yonler = new List<List<string>>();

            for (int i = 0; i < ParametreList.Count; i++)
            {
                var BosList = new List<string>();

                for (int j = 0; j < ViopData.Count; j++)
                    BosList.Add("");

                Yonler.Add(BosList);
            }

            for (int i = 0; i < ParametreList.Count; i++)
            {
                var SembolSistem = Sistem.SistemGetir(SistemList[i], Sistem.Sembol, PeriyotList[i]);

                if (SembolSistem == null) continue;

                for (int j = 0; j < SembolSistem.GrafikVerileri.Count; j++)
                {
                    var Tarih = SembolSistem.GrafikVerileri[j].Date;
                    if (TarihDictionary.ContainsKey(Tarih))
                        Yonler[i][TarihDictionary[Tarih]] = SembolSistem.Yon[j];
                }
            }

            var SonPozDictionary = new Dictionary<string, int>();
            var PozList = Sistem.Liste(0);
            for (int i = 0; i < Yonler.Count; i++)
            {
                var SonPozStr = "";
                for (int j = V.Count - 1; j > 0; j--)
                {
                    if (Yonler[i][j] != "")
                    {
                        SonPozStr = Yonler[i][j];
                        break;
                    }
                }

                int SonPozLot = 0;
                if (SonPozStr == "A")
                    SonPozLot = Convert.ToInt32(LotList[i]);
                else if (SonPozStr == "S")
                    SonPozLot = -Convert.ToInt32(LotList[i]);
                SonPozDictionary[SistemList[i]] = SonPozLot;

                float Poz = 0;
                for (int j = 0; j < V.Count; j++)
                {
                    if (Yonler[i][j] == "A")
                        Poz = LotList[i];
                    else if (Yonler[i][j] == "S")
                        Poz = -LotList[i];
                    else if (Yonler[i][j] == "F")
                        Poz = 0;

                    PozList[j] += Convert.ToInt32(Poz);
                }
            }

            Sistem.Cizgiler[0].Deger = PozList;
            Sistem.Cizgiler[0].Aciklama = "PozList";

            Sistem.Cizgiler[1].Deger = Sistem.Liste(0);
            Sistem.Cizgiler[1].Aciklama = "ZeroList";

            Sistem.DolguEkle(0, 1, Color.FromArgb(120, 0, 255, 0), Color.FromArgb(120, 255, 0, 0));

            int Counter = 0;
            foreach (var item in SonPozDictionary)
            {
                RenkPoz = Color.Gold;
                if (item.Value > 0)
                    RenkPoz = Color.LimeGreen;
                else if (item.Value < 0)
                    RenkPoz = Color.Red;
                Counter++;

                str = string.Format("{0} : {1}\t", Math.Abs(item.Value).ToString("0"), item.Key);
                Sistem.ZeminYazisiEkle(str, panelNo, colNums[colNum], rowNums[rowNum++], RenkPoz, "Tahoma", 12);
            }

            var SonYon = "";
            for (int i = 0; i < V.Count; i++)
            {
                if (PozList[i] > 0 && SonYon != "A")
                    Sistem.Yon[i] = "A";
                else if (PozList[i] < 0 && SonYon != "S")
                    Sistem.Yon[i] = "S";
                else if (PozList[i] == 0 && SonYon != "F")
                    Sistem.Yon[i] = "F";

                if (Sistem.Yon[i] != "")
                    SonYon = Sistem.Yon[i];
            }

            Sistem.GetiriHesapla("01/01/1900", 0.0f);
            Sistem.GetiriMaxDDHesapla("01/10/1900", "01/01/2031");

            var Kasa = 0.0f;
            var KZList = Sistem.Liste(0);
            for (int i = 1; i < V.Count; i++)
            {
                if (PozList[i] != PozList[i - 1])
                    Kasa += -(PozList[i] - PozList[i - 1]) * C[i];
                KZList[i] = Kasa + (PozList[i] * C[i]);
            }

            Sistem.Cizgiler[2].Deger = KZList;
            Sistem.Cizgiler[2].Aciklama = "KZList";

            var GetiriKZGunSonu = Sistem.Liste(0);
            GetiriKZGunSonu[GetiriKZGunSonu.Count - 1] = KZList[KZList.Count - 1];
            for (int i = KZList.Count - 2; i >= 0; i--)
            {
                GetiriKZGunSonu[i] = GetiriKZGunSonu[i + 1];
                if (V[i].Date.Day != V[i + 1].Date.Day)
                    GetiriKZGunSonu[i] = KZList[i];
            }

            Sistem.Cizgiler[3].Deger = KZList;
            Sistem.Cizgiler[3].Aciklama = "GetiriKZ";

            var Sure = (DateTime.Now - V[0].Date).TotalDays / 30.4;

            var DateGunBarNo = 0;
            for (int i = V.Count - 2; i > 0; i--)
            {
                if (V[i].Date.Day != V[V.Count - 1].Date.Day)
                {
                    DateGunBarNo = i;
                    break;
                }
            }
            var GetiriGun = Math.Round((KZList[KZList.Count - 1] - KZList[DateGunBarNo]), 1);

            var Date1Ay = DateTime.Now.AddDays(-30);
            var Date1AyBarNo = 0;
            for (int i = V.Count - 1; i > 0; i--)
            {
                if (V[i].Date <= Date1Ay)
                {
                    Date1AyBarNo = i;
                    break;
                }
            }
            var Getiri1Ay = Math.Round((KZList[KZList.Count - 1] - KZList[Date1AyBarNo]), 1);

            var Date2Ay = DateTime.Now.AddDays(-60);
            var Date2AyBarNo = 0;
            for (int i = V.Count - 1; i > 0; i--)
            {
                if (V[i].Date <= Date2Ay)
                {
                    Date2AyBarNo = i;
                    break;
                }
            }
            var Getiri2Ay = Math.Round((KZList[KZList.Count - 1] - KZList[Date2AyBarNo]), 1);

            var Date3Ay = DateTime.Now.AddDays(-90);
            var Date3AyBarNo = 0;
            for (int i = V.Count - 1; i > 0; i--)
            {
                if (V[i].Date <= Date3Ay)
                {
                    Date3AyBarNo = i;
                    break;
                }
            }
            var Getiri3Ay = Math.Round((KZList[KZList.Count - 1] - KZList[Date3AyBarNo]), 1);

            var GetiriKZAy = Sistem.Liste(0);
            for (int i = 1; i < V.Count; i++)
            {
                if (V[i].Date.Month == V[i - 1].Date.Month)
                    GetiriKZAy[i] = GetiriKZAy[i - 1];
                else
                    GetiriKZAy[i] = KZList[i - 1];
            }

            Sistem.Cizgiler[4].Deger = GetiriKZAy;
            Sistem.Cizgiler[4].Aciklama = "GetiriKZAy";

            Renk = Color.Gold;

            str = string.Format("Süre : {0} Ay\t", Sure.ToString("0.0"));
            Sistem.ZeminYazisiEkle(str, panelNo, colNums[colNum], rowNums[rowNum++], Renk, "Tahoma", 12);

            str = string.Format("Getiri : {0}\t", KZList[KZList.Count - 1].ToString("0.000"));
            Sistem.ZeminYazisiEkle(str, panelNo, colNums[colNum], rowNums[rowNum++], Renk, "Tahoma", 12);

            str = string.Format("30 Gün : {0}\t", Getiri1Ay.ToString("0.000"));
            Sistem.ZeminYazisiEkle(str, panelNo, colNums[colNum], rowNums[rowNum++], Renk, "Tahoma", 12);

            str = string.Format("60 Gün : {0}\t", Getiri2Ay.ToString("0.000"));
            Sistem.ZeminYazisiEkle(str, panelNo, colNums[colNum], rowNums[rowNum++], Renk, "Tahoma", 12);

            str = string.Format("90 Gün : {0}\t", Getiri3Ay.ToString("0.000"));
            Sistem.ZeminYazisiEkle(str, panelNo, colNums[colNum], rowNums[rowNum++], Renk, "Tahoma", 12);

            if (GetiriGun > 0)
                Renk = Color.LimeGreen;
            else if (GetiriGun < 0)
                Renk = Color.Red;

            str = string.Format("Bu Gün : {0}\t", GetiriGun.ToString("0.000"));
            Sistem.ZeminYazisiEkle(str, panelNo, colNums[colNum], rowNums[rowNum++], Renk, "Tahoma", 12);

            Sistem.Cizgiler[5].Deger = GetiriKZGunSonu;
            Sistem.Cizgiler[5].Aciklama = "GetiriKZGun";

            var GetiriKZSaatSonu = Sistem.Liste(0);
            GetiriKZSaatSonu[GetiriKZSaatSonu.Count - 1] = KZList[KZList.Count - 1];
            for (int i = KZList.Count - 2; i >= 0; i--)
            {
                GetiriKZSaatSonu[i] = GetiriKZSaatSonu[i + 1];
                if (V[i].Date.Hour != V[i + 1].Date.Hour)
                    GetiriKZSaatSonu[i] = KZList[i];
            }

            Sistem.Cizgiler[6].Deger = GetiriKZSaatSonu;
            Sistem.Cizgiler[6].Aciklama = "GetiriKZSaatSonu";

            var IslemSayisiList = Sistem.Liste(0);
            var KomisyonIslemSayisiList = Sistem.Liste(0);
            int IslemSayisi = 0;
            int KomisyonIslemSayisi = 0;
            int IslemSayisiA = 0;
            int IslemSayisiS = 0;
            int IslemSayisiF = 0;
            SonYon = "";
            for (int i = 1; i < V.Count; i++)
            {
                if (PozList[i] > 0 && SonYon != "A")
                {
                    Sistem.Yon[i] = "A";

                    IslemSayisiA++;
                }
                else if (PozList[i] < 0 && SonYon != "S")
                {
                    Sistem.Yon[i] = "S";

                    IslemSayisiS++;
                }
                else if (PozList[i] == 0 && SonYon != "F")
                {
                    Sistem.Yon[i] = "F";

                    IslemSayisiF++;
                }

                if (PozList[i] != PozList[i - 1]) { 
                    IslemSayisi++;
                    KomisyonIslemSayisi += Math.Abs( PozList[i] - PozList[i-1] );
                }

                IslemSayisiList[i] = IslemSayisi;
                KomisyonIslemSayisiList[i] = KomisyonIslemSayisi;

                if (Sistem.Yon[i] != "")
                    SonYon = Sistem.Yon[i];
            }

            Sistem.Cizgiler[7].Deger = IslemSayisiList;
            Sistem.Cizgiler[7].Aciklama = "IslemSayisiList";

            Sistem.Cizgiler[8].Deger = KomisyonIslemSayisiList;
            Sistem.Cizgiler[8].Aciklama = "KomisyonIslemSayisiList";

            var SonPoz = PozList[PozList.Count - 1];
            Renk = Color.Gold;
            if (SonPoz > 0)
                Renk = Color.LimeGreen;
            else if (SonPoz < 0)
                Renk = Color.Red;

            str = string.Format("Poz : {0}\t", SonPoz.ToString("0"));
            Sistem.ZeminYazisiEkle(str, panelNo, colNums[colNum], rowNums[rowNum++], Renk, "Tahoma", 12);

            Renk = Color.Gold;
            str = string.Format("Profit Factor : {0}\t", Sistem.ProfitFactor.ToString("0.000"));
            Sistem.ZeminYazisiEkle(str, panelNo, colNums[colNum], rowNums[rowNum++], Renk, "Tahoma", 12);

            str = string.Format("MaxDD : {0}\t", Sistem.GetiriMaxDD.ToString("0.000"));
            Sistem.ZeminYazisiEkle(str, panelNo, colNums[colNum], rowNums[rowNum++], Renk, "Tahoma", 12);

            str = string.Format("MaxDD Tarih : {0}\t", Sistem.GetiriMaxDDTarih.ToString());
            Sistem.ZeminYazisiEkle(str, panelNo, colNums[colNum], rowNums[rowNum++], Renk, "Tahoma", 12);

            str = string.Format("IslemSayisi : {0}\t", IslemSayisi);
            Sistem.ZeminYazisiEkle(str, panelNo, colNums[colNum], rowNums[rowNum++], Renk, "Tahoma", 12);

            Komisyon = KomisyonIslemSayisi * KomisyonCarpan;

            GetiriFiyat = KZList[KZList.Count - 1] * VarlikAdedCarpani;
            SonBakiyeFiyat = IlkBakiyeFiyat + GetiriFiyat;

            GetiriFiyatNet = GetiriFiyat - Komisyon;
            SonBakiyeFiyatNet = IlkBakiyeFiyat + GetiriFiyatNet;

            str = string.Format("KomisyonIslemSayisi : {0} 	 Komisyon : {1}", KomisyonIslemSayisi, Komisyon.ToString("0.00"));
            Sistem.ZeminYazisiEkle(str, panelNo, colNums[colNum], rowNums[rowNum++], Renk, "Tahoma", 12);

            str = string.Format("{0} : {1} 	 {2, -19} : {3} 	 {4, -20} : {5} ", "IlkBakiye", IlkBakiyeFiyat.ToString("0.00"), "Getiri", GetiriFiyat.ToString("0.00"), "SonBakiye", SonBakiyeFiyat.ToString("0.00"));
            Sistem.ZeminYazisiEkle(str, panelNo, colNums[colNum], rowNums[rowNum++], Renk, "Tahoma", 12);

            str = string.Format("{0} : {1} 	 {2, -15} : {3} 	 {4, -16} : {5} ", "IlkBakiye", IlkBakiyeFiyat.ToString("0.00"), "Getiri (Net)", GetiriFiyatNet.ToString("0.00"), "SonBakiye (Net)", SonBakiyeFiyatNet.ToString("0.00"));
            Sistem.ZeminYazisiEkle(str, panelNo, colNums[colNum], rowNums[rowNum++], Renk, "Tahoma", 12);

        }

        public int SetParams(dynamic Sistem, int VarlikAdedCarpani = 1, float KomisyonCarpan = 0.0f, float IlkBakiye = 100000.0f)
        {
            int result = 0;

            this.IlkBakiyeFiyat = IlkBakiye;
            this.KomisyonCarpan = KomisyonCarpan;
            this.VarlikAdedCarpani = VarlikAdedCarpani;

            return result;
        }
    }
