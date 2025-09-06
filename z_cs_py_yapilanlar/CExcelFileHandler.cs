public class CExcelFileHandler
{
    public CExcelFileHandler()
    {

    }

    public dynamic CreateExcelArrayObject(dynamic Sistem, int RowNum, int ColNum)
    {
        var ExcelArray = new object[RowNum, ColNum];
        return ExcelArray;
    }

    public void SetCellValue(dynamic Sistem, dynamic ExcelArray, int Row, int Col, dynamic Value)
    {
        // [0,0] ile başliyor
        ExcelArray[Row, Col] = Value;
    }

    public void SaveToExcelFile(dynamic Sistem, string FileName, dynamic ExcelArray)
    {
        Sistem.ExcelKopyala(ExcelArray, FileName);
    }

    public dynamic ReadExcelFile(dynamic Sistem, string FileName)
    {
        var ExcelArray = Sistem.ExcelOku(FileName);
        return ExcelArray;
    }

    public int GetRowNumber(dynamic Sistem, dynamic ExcelArray)
    {
        int rowNumber = ExcelArray.GetLength(0); //satırların sayısını bul
        return rowNumber;
    }

    public dynamic GetCellValue(dynamic Sistem, dynamic ExcelArray, int Row, int Col)
    {
        // [1,1] ile başliyor
        return ExcelArray[Row, Col];
    }
}