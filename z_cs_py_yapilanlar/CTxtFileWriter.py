class CTxtFileWriter:
    def __init__(self):
        self.FileName = None
        self.fs = None
        self.sw = None
        self.IsOpenedFlag = False

    def open_file(self, file_name, append_enabled=False):
        mode = 'a' if append_enabled else 'w'
        self.fs = open(file_name, mode)
        self.IsOpenedFlag = True
        return self.IsOpenedFlag

    def close_file(self):
        if self.fs:
            self.fs.close()
        self.IsOpenedFlag = False
        return self.IsOpenedFlag

    def is_opened(self):
        return self.IsOpenedFlag

    def write_line(self, text):
        if self.fs:
            self.fs.write(text + '\n')
            self.fs.flush()

    def write(self, text):
        if self.fs:
            self.fs.write(text)
            self.fs.flush()

    def write_data(self):
        with open('c:\\test.txt', 'a') as f:
            text_to_write = "This is the text to be written to the file."
            f.write(text_to_write + '\n')
