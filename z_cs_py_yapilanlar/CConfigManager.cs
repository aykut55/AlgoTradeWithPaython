public class CConfigManager
{
    string SonOkumaZamani;
    string SonYazmaZamani;
    string SonResetZamani;
    string SonClearZamani;

    string SonOkumaZamaniKey;
    string SonYazmaZamaniKey;
    string SonResetZamaniKey;
    string SonClearZamaniKey;

    ~CConfigManager()
    {

    }

    public CConfigManager()
    {

    }

    public string CreateKey(dynamic Sistem, string Suffix)
    {
        return ( Sistem.Name + ";" + Sistem.Sembol + ";" + Sistem.Periyot + ";" + Suffix );
    }

    public void SetValue(dynamic Sistem, string Key, dynamic Value)
    {
        SonYazmaZamaniKey = Key + ";" + "YAZMA ZAMANI";
        SonYazmaZamani = DateTime.Now.ToString("HH:mm:ss"); //yyyy.MM.dd HH:mm:ss
        Sistem.NesneKaydet(SonYazmaZamaniKey, SonYazmaZamani);

        Sistem.NesneKaydet(Key, Value);
    }

    public dynamic GetValue(dynamic Sistem, string Key)
    {
        SonOkumaZamaniKey = Key + ";" + "OKUMA ZAMANI";
        SonOkumaZamani = DateTime.Now.ToString("HH:mm:ss"); //yyyy.MM.dd HH:mm:ss
        Sistem.NesneKaydet(SonOkumaZamaniKey, SonOkumaZamani);

        return Sistem.NesneGetir(Key);
    }
}