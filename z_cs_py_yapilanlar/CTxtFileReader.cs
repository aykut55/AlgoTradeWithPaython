public class CTxtFileReader
{
    string FileName;
    FileStream fs;
    StreamWriter sw;

    public CTxtFileReader()
    {

    }

    // while ((line = sr.ReadLine()) != null) {

    public void OpenFile(string fileName)
    {
        fs = new FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        sw = new StreamWriter(fs);
    }

    public void CloseFile()
    {
        sw.Close();
        fs.Close();
    }

    private void ReadData()
    {
        FileStream fs = new FileStream("c:\\test.txt", FileMode.Open, FileAccess.Read);
        StreamReader sr = new StreamReader(fs);
        Console.WriteLine("Program to show content of test file");
        sr.BaseStream.Seek(0, SeekOrigin.Begin);
        string str = sr.ReadLine();
        while (str != null)
        {
            Console.WriteLine(str);
            str = sr.ReadLine();
        }
        Console.ReadLine();
        sr.Close();
        fs.Close();
    }
}