public class CTimeUtils : CBase
    {
        [DllImport("kernel32.dll")]
        public static extern UInt64 GetTickCount64();

        public float ToplamGecenSureAy { get; set; }
        public float ToplamGecenSureGun { get; set; }
        public float ToplamGecenSureSaat { get; set; }
        public float ToplamGecenSureDakika { get; set; }

        int counter5_90 = 0;
        int counter5_120 = 0;
        int counter5_150 = 0;
        int counter5_180 = 0;
        int counter5_240 = 0;
        int counter5_300 = 0;
        int counter5_480 = 0;

        int WatchDogCounter = 0;
        bool WatchDogCounterStarted = false;
        bool WatchDogCounterFinished = false;

        public UInt64 startTimeTick { get; set; }
        public UInt64 stopTimeTick { get; set; }
        public UInt64 currentTimeTick { get; set; }
        public UInt64 elapsedTimeTick { get; set; }
        public UInt64 CurrentTimeInMSec { get; set; }
        public UInt64 ElapsedTimeInMSec { get; set; }
        public UInt64 ExecutionTimeInMSec { get; set; }

        ~CTimeUtils()
        {
        }

        public CTimeUtils()
        {

        }

        public CTimeUtils Initialize(dynamic Sistem, dynamic V, dynamic Open, dynamic High, dynamic Low, dynamic Close, dynamic Volume, dynamic Lot)
        {
            SetData(Sistem, V, Open, High, Low, Close, Volume, Lot);

            Reset(Sistem);

            return this;
        }

        public CTimeUtils Reset(dynamic Sistem)
        {
            ToplamGecenSureAy = 0f;
            ToplamGecenSureGun = 0f;
            ToplamGecenSureSaat = 0f;
            ToplamGecenSureDakika = 0f;

            WatchDogCounter = 0;
            WatchDogCounterStarted = false;
            WatchDogCounterFinished = false;

            startTimeTick = 0;
            stopTimeTick = 0;
            currentTimeTick = 0;
            elapsedTimeTick = 0;
            CurrentTimeInMSec = 0;
            ElapsedTimeInMSec = 0;
            ExecutionTimeInMSec = 0;

            return this;
        }

        public void GecenZamanBilgileriniAl(dynamic Sistem)
        {
            var sureDakika = (DateTime.Now - V[0].Date).TotalMinutes;
            var sureSaat = (DateTime.Now - V[0].Date).TotalHours;
            var sureGun = (DateTime.Now - V[0].Date).TotalDays;
            var sureAy = sureGun / 30.4f;
            ToplamGecenSureAy = Convert.ToSingle(sureAy);           //ToplamGecenSureAy = AyList.Count;
            ToplamGecenSureGun = Convert.ToSingle(sureGun);         //ToplamGecenSureGun = GunList.Count;
            ToplamGecenSureSaat = Convert.ToSingle(sureSaat);       //ToplamGecenSureSaat = SaatList.Count;
            ToplamGecenSureDakika = Convert.ToSingle(sureDakika);   //ToplamGecenSureDakika = DakikalikList.Count;
        }

        public string GetLastExecutionTime(dynamic Sistem)
        {
            string LastExecutionTime = DateTime.Now.ToString("yyyy.MM.dd HH:mm:ss");
            return LastExecutionTime;
        }

        public bool IsYeniAy(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;
            bool yeniAy = (V[i].Date.Month != V[i - 1].Date.Month);
            return yeniAy;
        }

        public bool IsYeniHafta(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;
            bool yeniGun = (V[i - 1].Date.Day != V[i].Date.Day);
            bool yeniHafta = yeniGun && (this.GetWeekNumber(Sistem, V[i - 1].Date) != this.GetWeekNumber(Sistem, V[i].Date));
            return yeniHafta;
        }

        public bool IsYeniGun(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;
            bool yeniGun = (V[i].Date.Day != V[i - 1].Date.Day);
            return yeniGun;
        }

        public bool IsYeniSaat(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;
            bool yeniSaat = (V[i].Date.Hour != V[i - 1].Date.Hour);
            return yeniSaat;
        }

        public bool IsYeni(dynamic Sistem, int BarIndex, string Periyot)
        {
            bool result = false;
            int i = BarIndex;

            switch (Periyot)
            {
                case "5S":
                    break;

                case "10S":
                    break;

                case "15S":
                    break;

                case "1":
                    result = (V[i].Date.Minute % 1 == 0);
                    break;

                case "2":
                    result = (V[i].Date.Minute % 2 == 0);
                    break;

                case "3":
                    result = (V[i].Date.Minute % 3 == 0);
                    break;

                case "4":
                    result = (V[i].Date.Minute % 4 == 0);
                    break;

                case "5":
                    result = (V[i].Date.Minute % 5 == 0);
                    break;

                case "10":
                    result = (V[i].Date.Minute % 10 == 0);
                    break;

                case "15":
                    result = (V[i].Date.Minute % 15 == 0);
                    break;

                case "20":
                    result = (V[i].Date.Minute % 20 == 0);
                    break;

                case "30":
                    result = (V[i].Date.Minute % 30 == 0);
                    break;

                case "45":
                    result = (V[i].Date.Minute % 45 == 0);
                    break;

                case "60":
                    result = (V[i].Date.Minute % 60 == 0);
                    break;

                case "90":
                    if (V[i].Date.Minute % 5 == 0)
                    {
                        counter5_90++;
                        if (counter5_90 == 18)
                        {
                            result = true;
                            counter5_90 = 0;
                        }
                    }
                    break;

                case "120":
                    if (V[i].Date.Minute % 5 == 0)
                    {
                        counter5_120++;
                        if (counter5_120 == 24)
                        {
                            result = true;
                            counter5_120 = 0;
                        }
                    }
                    break;

                case "150":
                    if (V[i].Date.Minute % 5 == 0)
                    {
                        counter5_150++;
                        if (counter5_150 == 30)
                        {
                            result = true;
                            counter5_150 = 0;
                        }
                    }
                    break;

                case "180":
                    if (V[i].Date.Minute % 5 == 0)
                    {
                        counter5_180++;
                        if (counter5_180 == 36)
                        {
                            result = true;
                            counter5_180 = 0;
                        }
                    }
                    break;

                case "240":
                    if (V[i].Date.Minute % 5 == 0)
                    {
                        counter5_240++;
                        if (counter5_240 == 48)
                        {
                            result = true;
                            counter5_240 = 0;
                        }
                    }
                    break;

                case "300":
                    if (V[i].Date.Minute % 5 == 0)
                    {
                        counter5_300++;
                        if (counter5_300 == 60)
                        {
                            result = true;
                            counter5_300 = 0;
                        }
                    }
                    break;

                case "480":
                    if (V[i].Date.Minute % 5 == 0)
                    {
                        counter5_480++;
                        if (counter5_480 == 96)
                        {
                            result = true;
                            counter5_480 = 0;
                        }
                    }
                    break;

                case "S":
                    if (V[i].Date.Minute % 5 == 0)
                    {
                        counter5_240++;
                        if (counter5_240 == 48)
                        {
                            result = true;
                            counter5_240 = 0;
                        }
                    }
                    break;

                case "G":
                    result = IsYeniGun(Sistem, i);
                    break;

                case "H":
                    result = IsYeniHafta(Sistem, i);
                    break;

                case "M":
                    result = IsYeniAy(Sistem, i);
                    break;

                default:
                    result = false;
                    break;
            }

            return result;
        }

        public float GecenSure(dynamic Sistem, string Ref = "A")
        {
            double sure = 0f;
            if (Ref == "A")
                sure = (DateTime.Now - V[0].Date).TotalDays / 30.4;
            else if (Ref == "G")
                sure = (DateTime.Now - V[0].Date).TotalDays;
            else if (Ref == "S")
                sure = (DateTime.Now - V[0].Date).TotalHours;
            else if (Ref == "D")
                sure = (DateTime.Now - V[0].Date).TotalMinutes;
            return Convert.ToSingle(sure);
        }

        public float PeriyotCarpanHesapla(dynamic Sistem, string Periyot1, int Value1, string Periyot2)
        {
            float k1 = PeriyotCarpanHesapla(Sistem, Periyot1);
            float k2 = PeriyotCarpanHesapla(Sistem, Periyot2);
            float value2 = Value1 * k2 / k1;
            return value2;
        }

        public float PeriyotCarpanHesapla(dynamic Sistem, string Periyot)
        {
            float periyotCarpanGunluk = 1.0f;

            if (Periyot == "B")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 1 * 60;
            }
            else if (Periyot == "5S")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 1 * 12;
            }
            else if (Periyot == "10S")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 1 * 6;
            }
            else if (Periyot == "15S")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 1 * 4;
            }
            else if (Periyot == "1")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 1;
            }
            else if (Periyot == "2")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 2;
            }
            else if (Periyot == "3")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 3;
            }
            else if (Periyot == "4")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 4;
            }
            else if (Periyot == "5")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 5;
            }
            else if (Periyot == "10")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 10;
            }
            else if (Periyot == "15")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 15;
            }
            else if (Periyot == "20")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 20;
            }
            else if (Periyot == "30")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 30;
            }
            else if (Periyot == "45")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 45;
            }
            else if (Periyot == "60")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 60;
            }
            else if (Periyot == "90")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 90;
            }
            else if (Periyot == "120")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 120;
            }
            else if (Periyot == "150")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 150;
            }
            else if (Periyot == "180")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 180;
            }
            else if (Periyot == "240")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 240;
            }
            else if (Periyot == "300")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 300;
            }
            else if (Periyot == "360")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 360;
            }
            else if (Periyot == "420")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 420;
            }
            else if (Periyot == "480")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 480;
            }
            else if (Periyot == "S")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 240;
            }
            else if (Periyot == "G")
            {
                periyotCarpanGunluk = 1.0f * 8 * 60 / 480;
            }
            else
            {
                periyotCarpanGunluk = 0;
                Sistem.Mesaj("Sistem.Periyot tanımlı değerler arasından seçilmelidir! ");
            }

            return periyotCarpanGunluk;
        }

        public void StartWatchDogCounter(dynamic Sistem, int Counter)
        {
            WatchDogCounter = Counter;
            WatchDogCounterStarted = true;
            WatchDogCounterFinished = false;
        }

        public void StopWatchDogCounter(dynamic Sistem)
        {
            WatchDogCounterStarted = false;
        }

        public bool WatchDogCounterIsStarted(dynamic Sistem)
        {
            return WatchDogCounterStarted == true;
        }

        public int StepWatchDogCounter(dynamic Sistem)
        {
            WatchDogCounter--;
            if (WatchDogCounter <= 0)
            {
                WatchDogCounterFinished = true;
            }
            return WatchDogCounter;
        }

        public bool WatchDogCounterIsFinished(dynamic Sistem)
        {
            return WatchDogCounterFinished == true;
        }

        public UInt64 StartTimer(dynamic Sistem)
        {
            startTimeTick = GetTickCount64();
            return startTimeTick;
        }

        public UInt64 StopTimer(dynamic Sistem)
        {
            stopTimeTick = GetTickCount64();
            return stopTimeTick;
        }

        public UInt64 GetCurrentTimeInMSec(dynamic Sistem)
        {
            currentTimeTick = GetTickCount64();
            CurrentTimeInMSec = currentTimeTick;
            return CurrentTimeInMSec;
        }

        public UInt64 GetElapsedTimeInMSec(dynamic Sistem)
        {
            elapsedTimeTick = GetTickCount64() - startTimeTick;
            ElapsedTimeInMSec = elapsedTimeTick;
            return ElapsedTimeInMSec;
        }

        public UInt64 GetExecutionTimeInMSec(dynamic Sistem)
        {
            elapsedTimeTick = stopTimeTick - startTimeTick;
            ExecutionTimeInMSec = elapsedTimeTick;
            return ExecutionTimeInMSec;
        }

        public DateTime GetDateTime(string DateTimeStr)
        {
            DateTime dateTime = DateTime.ParseExact(DateTimeStr, "dd.MM.yyyy HH:mm:ss", System.Globalization.CultureInfo.InvariantCulture);
            return dateTime;
        }

        public DateTime GetDateTime(string DateStr, string TimeStr)
        {
            string DateTimeStr = DateStr + " " + TimeStr;
            return GetDateTime(DateTimeStr);
        }

        public DateTime GetDate(string DateStr)
        {
            DateTime dateTime = DateTime.ParseExact(DateStr, "dd.MM.yyyy", System.Globalization.CultureInfo.InvariantCulture);
            return dateTime;
        }

        public DateTime GetTime(string TimeStr)
        {
            DateTime dateTime = DateTime.ParseExact(TimeStr, "HH:mm:ss", System.Globalization.CultureInfo.InvariantCulture);
            return dateTime;
        }

        public int CompareDates(DateTime DateTime1, DateTime DateTime2)
        {
            int res = DateTime.Compare(DateTime1, DateTime2);

            return res;
        }

        public int CompareTimes(DateTime DateTime1, DateTime DateTime2)
        {
            int res = TimeSpan.Compare(DateTime1.TimeOfDay, DateTime2.TimeOfDay);

            return res;
        }

        public DateTime GetDateTime(int year, int month, int day, int hour, int minute, int second)
        {
            DateTime dateTime = new DateTime(year, month, day, hour, minute, second);
            return dateTime;
        }
        
        public string ToLongDateString(DateTime dateTime)
        {
            return dateTime.ToLongDateString();
        }

        public string ToLongTimeString(DateTime dateTime)
        {
            return dateTime.ToLongTimeString();
        }

        public string GetShortDateString(DateTime dateTime)
        {
            return dateTime.ToShortDateString();
        }

        public string GetShortTimeString(DateTime dateTime)
        {
            return dateTime.ToShortTimeString();
        }

        public DateTime GetCurrentDateTime()
        {
            DateTime dateTime = DateTime.Now;
            return dateTime;
        }
        public DateTime GetCurrentDate()
        {
            DateTime date = DateTime.Now.Date;
            return date;
        }

        public TimeSpan GetCurrentTime()
        {
            TimeSpan time = DateTime.Now.TimeOfDay;
            return time;
        }

        public DateTime GetBarDateTime(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;
            var V = Sistem.GrafikVerileri;
            return V[i].Date;
        }

        public DateTime GetBarDate(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;
            var V = Sistem.GrafikVerileri;
            return V[i].Date;
        }

        public DateTime GetBarTime(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;
            var V = Sistem.GrafikVerileri;
            return V[i].Date;
        }

        public bool IsBar(dynamic Sistem, int BarIndex, string BarZamani)
        {
            int i = BarIndex;
            var V = Sistem.GrafikVerileri;
            bool result = V[i].Date.ToShortTimeString() == BarZamani;
            return result;
        }

        public bool IsPozKapatValid(dynamic Sistem, int BarIndex, string PozKapatZamani)
        {
            int i = BarIndex;
            var V = Sistem.GrafikVerileri;
            bool result = V[i].Date.ToShortTimeString() == PozKapatZamani;
            return result;
        }

        public int CheckBarTimeWith(dynamic Sistem, int BarIndex, string BarZamani)
        {
            int i = BarIndex;
            var V = Sistem.GrafikVerileri;
            int result = CompareTimes(GetBarDateTime(Sistem, i), GetTime(BarZamani));
            return result;
        } 

        public int CheckBarDateWith(dynamic Sistem, int BarIndex, string BarZamani)
        {
            int i = BarIndex;
            var V = Sistem.GrafikVerileri;
            int result = CompareDates(GetBarDateTime(Sistem, i), GetDate(BarZamani));
            return result;
        } 

        public int CheckBarDateTimeWith(dynamic Sistem, int BarIndex, string BarZamani)
        {
            int i = BarIndex;
            var V = Sistem.GrafikVerileri;
            int result = CompareDates(GetBarDateTime(Sistem, i), GetDateTime(BarZamani));
            return result;
        }

    public bool IsGunSonuPozKapatSinyaliGeldi_SadeceAnaliz(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;
            var V = Sistem.GrafikVerileri;
            int BarCount = V.Count;
            bool result = (i < BarCount - 1 && V[i].Date.Day != V[i + 1].Date.Day);
            return result;
        }

        public bool IsAySonuPozKapatSinyaliGeldi_SadeceAnaliz(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;
            var V = Sistem.GrafikVerileri;
            int BarCount = V.Count;
            bool result = (i < BarCount - 1 && V[i].Date.Month != V[i + 1].Date.Month);
            return result;
        }

        public int GetWeekNumber(dynamic Sistem, DateTime date)
        {
            var day = (int)System.Globalization.CultureInfo.CurrentCulture.Calendar.GetDayOfWeek(date);
            return System.Globalization.CultureInfo.CurrentCulture.Calendar.GetWeekOfYear(date.AddDays(4 - (day == 0 ? 7 : day)), System.Globalization.CalendarWeekRule.FirstFourDayWeek, DayOfWeek.Monday);
        }

        public int GetWeekNumber(dynamic Sistem, int BarIndex)
        {
            int i = BarIndex;
            var V = Sistem.GrafikVerileri;
            return GetWeekNumber(Sistem, V[i].Date);
        }
}