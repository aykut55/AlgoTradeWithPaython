public class CTxtFileWriter
{
    string FileName;
    FileStream fs;
    StreamWriter sw;
    bool IsOpenedFlag;

    public CTxtFileWriter()
    {
        IsOpenedFlag = false;
    }

    public bool OpenFile(string fileName, bool appendEnabled = false)
    {
        fs = new FileStream(fileName, appendEnabled ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.ReadWrite);
        sw = new StreamWriter(fs);

        IsOpenedFlag = true;
        return IsOpenedFlag;
    }

    public bool CloseFile()
    {
        sw.Close();
        fs.Close();

        IsOpenedFlag = false;
        return IsOpenedFlag;
    }

    public bool IsOpened()
    {
        return this.IsOpenedFlag;
    }

    public void WriteLine(string text)
    {
        sw.WriteLine(text);
        sw.Flush();
    }

    public void Write(string text)
    {
        sw.Write(text);
        sw.Flush();
    }

    private void WriteData()
    {
        FileStream fs = new FileStream("c:\\test.txt", FileMode.Append, FileAccess.Write);
        StreamWriter sw = new StreamWriter(fs);
        Console.WriteLine("Enter the text which you want to write to the file");
        string str = Console.ReadLine();
        sw.WriteLine(str);
        sw.Flush();
        sw.Close();
        fs.Close();
    }
}