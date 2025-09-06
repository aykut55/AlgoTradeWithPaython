public class CSharedMemory
{
    string SonOkumaZamani;
    string SonYazmaZamani;
    string SonResetZamani;
    string SonClearZamani;

    string SonOkumaZamaniKey;
    string SonYazmaZamaniKey;
    string SonResetZamaniKey;
    string SonClearZamaniKey;
    /*
    var Anahtar11 = Sistem.Name + ";" + Sistem.Sembol + ";" + Sistem.Periyot + ";" + "YAZMA ZAMANI";
    var Anahtar12 = Sistem.Name + ";" + Sistem.Sembol + ";" + Sistem.Periyot + ";" + "RESET ZAMANI";
    var Anahtar13 = Sistem.Name + ";" + Sistem.Sembol + ";" + Sistem.Periyot + ";" + "OKUMA ZAMANI";
    var Anahtar12 = Sistem.Name + ";" + Sistem.Sembol + ";" + Sistem.Periyot + ";" + "CLEAR ZAMANI";
    */
    ~CSharedMemory()
    {

    }

    public CSharedMemory()
    {

    }

    /*public CSharedMemory Initialize(dynamic Sistem)
    {
        return this;
    }

    public CSharedMemory Reset(dynamic Sistem)
    {
        return this;
    }*/

    public string CreateKey(dynamic Sistem, string Suffix)
    {
        return ( Sistem.Name + ";" + Sistem.Sembol + ";" + Sistem.Periyot + ";" + Suffix );
    }

    public bool IsMemoryNull(dynamic Sistem, string Key)
    {
        return (Sistem.NesneGetir(Key) == null);
    }

    public void ClearMemory(dynamic Sistem, string Key)
    {
        SonClearZamaniKey = Key + ";" + "CLEAR ZAMANI";
        SonClearZamani = DateTime.Now.ToString("HH:mm:ss"); //yyyy.MM.dd HH:mm:ss
        Sistem.NesneKaydet(SonClearZamaniKey, SonClearZamani);

        Sistem.NesneKaydet(Key, null);
    }

    public void ResetMemory(dynamic Sistem, string Key)
    {
        SonResetZamaniKey = Key + ";" + "RESET ZAMANI";
        SonResetZamani = DateTime.Now.ToString("HH:mm:ss"); //yyyy.MM.dd HH:mm:ss
        Sistem.NesneKaydet(SonResetZamaniKey, SonResetZamani);

        Sistem.NesneKaydet(Key, null);
    }

    public void WriteToMemory(dynamic Sistem, string Key, dynamic Value)
    {
        SonYazmaZamaniKey = Key + ";" + "YAZMA ZAMANI";
        SonYazmaZamani = DateTime.Now.ToString("HH:mm:ss"); //yyyy.MM.dd HH:mm:ss
        Sistem.NesneKaydet(SonYazmaZamaniKey, SonYazmaZamani);

        Sistem.NesneKaydet(Key, Value);
    }

    public dynamic ReadFromMemory(dynamic Sistem, string Key)
    {
        SonOkumaZamaniKey = Key + ";" + "OKUMA ZAMANI";
        SonOkumaZamani = DateTime.Now.ToString("HH:mm:ss"); //yyyy.MM.dd HH:mm:ss
        Sistem.NesneKaydet(SonOkumaZamaniKey, SonOkumaZamani);

        return Sistem.NesneGetir(Key);
    }

    public dynamic ReadFromMemoryAsString(dynamic Sistem, string Key)
    {
        return ReadFromMemory(Sistem, Key);
    }

    public dynamic ReadFromMemoryAsInteger(dynamic Sistem, string Key)
    {
        return Convert.ToInt32( ReadFromMemory(Sistem, Key) );
    }

    public dynamic ReadFromMemoryAsSingle(dynamic Sistem, string Key)
    {
        return Convert.ToSingle( ReadFromMemory(Sistem, Key) );
    }

    public dynamic ReadFromMemoryAsDouble(dynamic Sistem, string Key)
    {
        return Convert.ToDouble( ReadFromMemory(Sistem, Key) );
    }

    /*
    var LastExecutionId = 0;
    var LastExecutionKey = Sistem.Name + ";" + Sistem.Sembol + ";" + Sistem.Periyot + ";" + "LastExecutionId";

    var LastExecutionIdStr = Sistem.NesneGetir(LastExecutionKey);
    if (LastExecutionIdStr == null) {
        LastExecutionId = 0;
        Sistem.NesneKaydet(LastExecutionKey, LastExecutionId);
    }
    else {
        LastExecutionId = Convert.ToInt32(LastExecutionIdStr);
        LastExecutionId++;
        Sistem.NesneKaydet(LastExecutionKey, LastExecutionId);
    }
    */
}