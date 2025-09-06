class CTxtFileReader:
    def __init__(self):
        self.FileName = None
        self.fs = None
        self.sw = None

    def open_file(self, file_name):
        self.fs = open(file_name, 'r')

    def close_file(self):
        if self.fs:
            self.fs.close()

    def read_data(self):
        with open('c:\\test.txt', 'r') as f:
            for line in f:
                print(line.strip())

