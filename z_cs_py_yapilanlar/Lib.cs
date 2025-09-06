namespace ideal
{
    public class Lib
    {
        [DllImport("kernel32.dll")]

        public static extern UInt64 GetTickCount64();

        ~Lib()
        {

        }

        public Lib()
        {

        }

        public void ShowMessage(dynamic Sistem, string Message = "This message comes from Lib::ShowMessage() ")
        {
            Sistem.Mesaj(Message);
        }

        // #####################################################################
        CTrader [] TraderArray = new CTrader [100];
        public CTrader GetTrader(dynamic Sistem, int Index = 0)
        {
            if (Index >= 0 && Index < 100)
            {
                if (TraderArray[Index] == null)
                    TraderArray[Index] = new CTrader(Sistem, Index);

                return TraderArray[Index];
            }

            return null;
        }
        public void DeleteTrader(CTrader Trader)
        {
            if (Trader != null)
            {
                Trader = null;
            }
        }

        // #####################################################################
        CVarlikManager VarlikManager = null;
        public CVarlikManager GetVarlik(dynamic Sistem)
        {
            if (VarlikManager == null)
                VarlikManager = new CVarlikManager();

            return VarlikManager;
        }
        public void DeleteVarlik(CVarlikManager VarlikManager)
        {
            if (VarlikManager != null) { VarlikManager = null; }
        }

        // #####################################################################
        CUtils Utils = null;
        public CUtils GetUtils(dynamic Sistem)
        {
            if (Utils == null)
                Utils = new CUtils();

            return Utils;
        }
        public void DeleteUtils(CUtils Utils)
        {
            if (Utils != null) { Utils = null; }
        }

        // #####################################################################
        CTimeUtils TimeUtils = null;
        public CTimeUtils GetTimeUtils(dynamic Sistem)
        {
            if (TimeUtils == null)
                TimeUtils = new CTimeUtils();

            return TimeUtils;
        }
        public void DeleteTimeUtils(CTimeUtils TimeUtils)
        {
            if (TimeUtils != null) { TimeUtils = null; }
        }

        // #####################################################################
        CBarUtils BarUtils = null;
        public CBarUtils GetBarUtils(dynamic Sistem)
        {
            if (BarUtils == null)
                BarUtils = new CBarUtils();

            return BarUtils;
        }
        public void DeleteBarUtils(CBarUtils BarUtils)
        {
            if (BarUtils != null) { BarUtils = null; }
        }

        // #####################################################################
        CFileUtils FileUtils = null;
        public CFileUtils GetFileUtils(dynamic Sistem)
        {
            if (FileUtils == null)
                FileUtils = new CFileUtils();

            return FileUtils;
        }
        public void DeleteFileUtils(CFileUtils FileUtils)
        {
            if (FileUtils != null) { FileUtils = null; }
        }

        // #####################################################################
        CExcelFileHandler ExcelFileHandler = null;
        public CExcelFileHandler GetExcelFileHandler(dynamic Sistem)
        {
            if (ExcelFileHandler == null)
                ExcelFileHandler = new CExcelFileHandler();

            return ExcelFileHandler;
        }
        public void DeleteExcelFileHandler(CExcelFileHandler ExcelFileHandler)
        {
            if (ExcelFileHandler != null) { ExcelFileHandler = null; }
        }

        // #####################################################################
        CBirlesikSistemManager BirlesikSistemManager = null;
        public CBirlesikSistemManager GetBirlesikSistemManager(dynamic Sistem)
        {
            if (BirlesikSistemManager == null)
                BirlesikSistemManager = new CBirlesikSistemManager();

            return BirlesikSistemManager;
        }
        public void DeleteBirlesikSistemManager(CBirlesikSistemManager BirlesikSistemManager)
        {
            if (BirlesikSistemManager != null) { BirlesikSistemManager = null; }
        }

        // #####################################################################
        CSharedMemory SharedMemory = null;
        public CSharedMemory GetSharedMemory(dynamic Sistem)
        {
            if (SharedMemory == null)
                SharedMemory = new CSharedMemory();

            return SharedMemory;
        }
        public void DeleteSharedMemory(CSharedMemory SharedMemory)
        {
            if (SharedMemory != null) { SharedMemory = null; }
        }

        // #####################################################################
        CZigZagAnalyzer ZigZagAnalyzer = null;
        public CZigZagAnalyzer GetZigZagAnalyzer(dynamic Sistem)
        {
            if (ZigZagAnalyzer == null)
                ZigZagAnalyzer = new CZigZagAnalyzer();

            return ZigZagAnalyzer;
        }
        public void DeleteZigZagAnalyzer(CZigZagAnalyzer ZigZagAnalyzer)
        {
            if (ZigZagAnalyzer != null) { ZigZagAnalyzer = null; }
        }

        // #####################################################################
        CConfigManager ConfigManager = null;
        public CConfigManager GetConfig(dynamic Sistem)
        {
            if (ConfigManager == null)
                ConfigManager = new CConfigManager();

            return ConfigManager;
        }
        public void DeleteConfig(CConfigManager ConfigManager)
        {
            if (ConfigManager != null) { ConfigManager = null; }
        }

        // #####################################################################
        CSystemWrapper SystemWrapper = null;
        public CSystemWrapper GetSystemWrapper(dynamic Sistem)
        {
            if (SystemWrapper == null)
                SystemWrapper = new CSystemWrapper();

            return SystemWrapper;
        }
        public void DeleteSystemWrapper(CSystemWrapper SystemWrapper)
        {
            if (SystemWrapper != null) { SystemWrapper = null; }
        }

        // #####################################################################
        CIndicatorManager IndicatorManager = null;
        public CIndicatorManager GetIndicatorManager(dynamic Sistem)
        {
            if (IndicatorManager == null)
                IndicatorManager = new CIndicatorManager();

            return IndicatorManager;
        }
        public void DeleteIndicatorManager(CIndicatorManager IndicatorManager)
        {
            if (IndicatorManager != null) { IndicatorManager = null; }
        }
    }
}