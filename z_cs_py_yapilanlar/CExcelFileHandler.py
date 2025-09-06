class CExcelFileHandler:
    def __init__(self):
        pass

    def create_excel_array_object(self, sistem, row_num, col_num):
        # Python'da 2 boyutlu bir dizi için liste içinde liste kullanılabilir.
        return [[None for _ in range(col_num)] for _ in range(row_num)]

    def set_cell_value(self, sistem, excel_array, row, col, value):
        # [0,0] tabanlı indeksleme
        excel_array[row][col] = value

    def save_to_excel_file(self, sistem, file_name, excel_array):
        sistem.excel_kopyala(excel_array, file_name)

    def read_excel_file(self, sistem, file_name):
        return sistem.excel_oku(file_name)

    def get_row_number(self, sistem, excel_array):
        return len(excel_array)

    def get_cell_value(self, sistem, excel_array, row, col):
        # C# yorumu [1,1] tabanlı olduğunu söylüyor, ancak kod [Row, Col] şeklinde.
        # Python'da listeler 0 tabanlı olduğu için bu şekilde bırakıyorum.
        return excel_array[row][col]
