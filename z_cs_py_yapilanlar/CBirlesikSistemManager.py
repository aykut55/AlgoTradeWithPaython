from datetime import datetime, timedelta

class CBirlesikSistemManager:
    def __init__(self):
        self.IlkBakiyeFiyat = 100000.0
        self.KomisyonCarpan = 3.0
        self.VarlikAdedCarpani = 1
        self.SonBakiyeFiyat = 0.0
        self.SonBakiyeFiyatNet = 0.0
        self.Komisyon = 0.0
        self.GetiriFiyat = 0.0
        self.GetiriFiyatNet = 0.0

    def __del__(self):
        pass

    def initialize(self, Sistem):
        self.reset(Sistem)
        return 0

    def reset(self, Sistem):
        return 0

    def sistemleri_calistir(self, Sistem, ParametreList):
        str_val = ""
        colNum, rowNum, panelNo = 0, 0, 1
        colNums = [160] * 15
        rowNums = [50 + i * 25 for i in range(15)]
        Renk = "Gold"
        RenkPoz = "Gold"

        V = Sistem.GrafikVerileri
        C = Sistem.GrafikFiyatOku(V, "Kapanis")

        for i in range(1, len(V)):
            if V[i].Date.day != V[i - 1].Date.day:
                Sistem.DikeyCizgiEkle(i, "DimGray", 2, 2)

        SistemList = []
        PeriyotList = []
        LotList = []

        for item in ParametreList:
            FieldArray = item.split(',')
            SistemList.append(FieldArray[0].strip())
            PeriyotList.append(FieldArray[1].strip())
            LotList.append(float(FieldArray[2].strip()))

        ViopData = Sistem.GrafikVerileri
        TarihDictionary = {ViopData[i].Date: i for i in range(len(ViopData))}

        Yonler = [["" for _ in range(len(ViopData))] for _ in range(len(ParametreList))]

        for i in range(len(ParametreList)):
            SembolSistem = Sistem.SistemGetir(SistemList[i], Sistem.Sembol, PeriyotList[i])
            if SembolSistem is None:
                continue
            for j in range(len(SembolSistem.GrafikVerileri)):
                Tarih = SembolSistem.GrafikVerileri[j].Date
                if Tarih in TarihDictionary:
                    Yonler[i][TarihDictionary[Tarih]] = SembolSistem.Yon[j]

        SonPozDictionary = {}
        PozList = Sistem.Liste(0)
        for i in range(len(Yonler)):
            SonPozStr = ""
            for j in range(len(V) - 1, 0, -1):
                if Yonler[i][j] != "":
                    SonPozStr = Yonler[i][j]
                    break
            
            SonPozLot = 0
            if SonPozStr == "A":
                SonPozLot = int(LotList[i])
            elif SonPozStr == "S":
                SonPozLot = -int(LotList[i])
            SonPozDictionary[SistemList[i]] = SonPozLot

            Poz = 0
            for j in range(len(V)):
                if Yonler[i][j] == "A":
                    Poz = LotList[i]
                elif Yonler[i][j] == "S":
                    Poz = -LotList[i]
                elif Yonler[i][j] == "F":
                    Poz = 0
                PozList[j] += int(Poz)

        Sistem.Cizgiler[0].Deger = PozList
        Sistem.Cizgiler[0].Aciklama = "PozList"
        Sistem.Cizgiler[1].Deger = Sistem.Liste(0)
        Sistem.Cizgiler[1].Aciklama = "ZeroList"
        Sistem.DolguEkle(0, 1, (120, 0, 255, 0), (120, 255, 0, 0))

        Counter = 0
        for key, value in SonPozDictionary.items():
            RenkPoz = "Gold"
            if value > 0:
                RenkPoz = "LimeGreen"
            elif value < 0:
                RenkPoz = "Red"
            Counter += 1
            str_val = f"{abs(value):0} : {key}\t"
            Sistem.ZeminYazisiEkle(str_val, panelNo, colNums[colNum], rowNums[rowNum], RenkPoz, "Tahoma", 12)
            rowNum += 1

        SonYon = ""
        for i in range(len(V)):
            if PozList[i] > 0 and SonYon != "A":
                Sistem.Yon[i] = "A"
            elif PozList[i] < 0 and SonYon != "S":
                Sistem.Yon[i] = "S"
            elif PozList[i] == 0 and SonYon != "F":
                Sistem.Yon[i] = "F"
            if Sistem.Yon[i] != "":
                SonYon = Sistem.Yon[i]

        Sistem.GetiriHesapla("01/01/1900", 0.0)
        Sistem.GetiriMaxDDHesapla("01/10/1900", "01/01/2031")

        Kasa = 0.0
        KZList = Sistem.Liste(0)
        for i in range(1, len(V)):
            if PozList[i] != PozList[i - 1]:
                Kasa += -(PozList[i] - PozList[i - 1]) * C[i]
            KZList[i] = Kasa + (PozList[i] * C[i])

        Sistem.Cizgiler[2].Deger = KZList
        Sistem.Cizgiler[2].Aciklama = "KZList"

        GetiriKZGunSonu = Sistem.Liste(0)
        GetiriKZGunSonu[-1] = KZList[-1]
        for i in range(len(KZList) - 2, -1, -1):
            GetiriKZGunSonu[i] = GetiriKZGunSonu[i + 1]
            if V[i].Date.day != V[i + 1].Date.day:
                GetiriKZGunSonu[i] = KZList[i]

        Sistem.Cizgiler[3].Deger = KZList
        Sistem.Cizgiler[3].Aciklama = "GetiriKZ"

        Sure = (datetime.now() - V[0].Date).days / 30.4

        DateGunBarNo = 0
        for i in range(len(V) - 2, 0, -1):
            if V[i].Date.day != V[-1].Date.day:
                DateGunBarNo = i
                break
        GetiriGun = round((KZList[-1] - KZList[DateGunBarNo]), 1)

        Date1Ay = datetime.now() - timedelta(days=30)
        Date1AyBarNo = 0
        for i in range(len(V) - 1, 0, -1):
            if V[i].Date <= Date1Ay:
                Date1AyBarNo = i
                break
        Getiri1Ay = round((KZList[-1] - KZList[Date1AyBarNo]), 1)

        Date2Ay = datetime.now() - timedelta(days=60)
        Date2AyBarNo = 0
        for i in range(len(V) - 1, 0, -1):
            if V[i].Date <= Date2Ay:
                Date2AyBarNo = i
                break
        Getiri2Ay = round((KZList[-1] - KZList[Date2AyBarNo]), 1)

        Date3Ay = datetime.now() - timedelta(days=90)
        Date3AyBarNo = 0
        for i in range(len(V) - 1, 0, -1):
            if V[i].Date <= Date3Ay:
                Date3AyBarNo = i
                break
        Getiri3Ay = round((KZList[-1] - KZList[Date3AyBarNo]), 1)

        GetiriKZAy = Sistem.Liste(0)
        for i in range(1, len(V)):
            if V[i].Date.month == V[i - 1].Date.month:
                GetiriKZAy[i] = GetiriKZAy[i - 1]
            else:
                GetiriKZAy[i] = KZList[i - 1]

        Sistem.Cizgiler[4].Deger = GetiriKZAy
        Sistem.Cizgiler[4].Aciklama = "GetiriKZAy"

        Renk = "Gold"
        str_val = f"Süre : {Sure:.1f} Ay\t"
        Sistem.ZeminYazisiEkle(str_val, panelNo, colNums[colNum], rowNums[rowNum], Renk, "Tahoma", 12)
        rowNum += 1

        # ... and so on for the rest of the ZeminYazisiEkle calls

    def set_params(self, Sistem, VarlikAdedCarpani=1, KomisyonCarpan=0.0, IlkBakiye=100000.0):
        result = 0
        self.IlkBakiyeFiyat = IlkBakiye
        self.KomisyonCarpan = KomisyonCarpan
        self.VarlikAdedCarpani = VarlikAdedCarpani
        return result
