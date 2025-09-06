public class CFileUtils
    {
        public int SistemId { get; set; }
        public string SistemAdi { get; set; }
        public string GrafikSembol { get; set; }
        public int GrafikPeriyot { get; set; }
        public int BarCount { get; set; }

        public bool LogEnabled { get; set; }
        public string LogFileName { get; set; }

        dynamic LogFileManager = null;
        bool LogFileIsOpened = false;
        string LogMessage = "";

        ~CFileUtils()
        {
            LogFileManager = null;
        }

        public CFileUtils()
        {
            LogFileManager = new CTxtFileWriter();
        }

        public int Initialize(dynamic Sistem)
        {
            SistemId = 0;
            SistemAdi = "SistemAdi";
            GrafikSembol = "GrafikSembol";
            GrafikPeriyot = 1;
            BarCount = 0;

            Reset(Sistem);

            return 0;
        }

        public CFileUtils Reset(dynamic Sistem)
        {
            return this;
        }

        public CFileUtils EnableLogging(dynamic Sistem)
        {
            LogEnabled = true;
            return this;
        }

        public CFileUtils DisableLogging(dynamic Sistem)
        {
            LogEnabled = false;
            return this;
        }

        public CFileUtils OpenLogFile(dynamic Sistem, string LogFileName, bool LastUpdatTimeEnabled = true, bool SummaryEnabled = false)
        {
            if (LogEnabled && !LogFileIsOpened)
            {
                LogFileIsOpened = true;
                LogFileManager.OpenFile(LogFileName);

                if (LogFileIsOpened)
                {
                    if (LastUpdatTimeEnabled)
                    {
                        WriteLastUpdatTimeToLogFile(Sistem);
                    }

                    if (SummaryEnabled)
                    {
                        LogMessage = String.Format("#  {0}   : {1} 	", "Bar Sayisi", BarCount);
                        LogFileManager.WriteLine(LogMessage);

                        LogMessage = String.Format("#  {0}    : {1} 	", "Sistem Id", SistemId);
                        LogFileManager.WriteLine(LogMessage);

                        LogMessage = String.Format("#  {0}   : {1} 	", "Sistem Adi", SistemAdi);
                        LogFileManager.WriteLine(LogMessage);

                        LogMessage = String.Format("#  {0}       : {1} 	", "Sembol", GrafikSembol);
                        LogFileManager.WriteLine(LogMessage);

                        LogMessage = String.Format("#  {0}      : {1} 	", "Periyot", GrafikPeriyot);
                        LogFileManager.WriteLine(LogMessage);
                    }
                }
            }

            return this;
        }

        public CFileUtils WriteLastUpdatTimeToLogFile(dynamic Sistem)
        {
            if (LogEnabled && LogFileIsOpened)
            {
                LogMessage = String.Format("#  {0}   : {1} 	", "Log Zamani", DateTime.Now.ToString("yyyy.MM.dd HH:mm:ss"));
                LogFileManager.WriteLine(LogMessage);
            }

            return this;
        }

        public CFileUtils WriteToLogFile(dynamic Sistem, string LogMessage, bool AppendNewLine = true)
        {
            if (LogEnabled && LogFileIsOpened)
            {
                if (AppendNewLine)
                    LogFileManager.WriteLine(LogMessage);
                else
                    LogFileManager.Write(LogMessage);
            }

            return this;
        }

        public CFileUtils CloseLogFile(dynamic Sistem)
        {
            if (LogEnabled && LogFileIsOpened)
            {
                LogFileManager.CloseFile();
                LogFileIsOpened = false;
            }

            return this;
        }
    }