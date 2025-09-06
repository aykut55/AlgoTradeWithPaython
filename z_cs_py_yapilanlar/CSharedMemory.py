import datetime

class CSharedMemory:
    def __init__(self):
        self.SonOkumaZamani = None
        self.SonYazmaZamani = None
        self.SonResetZamani = None
        self.SonClearZamani = None
        self.SonOkumaZamaniKey = None
        self.SonYazmaZamaniKey = None
        self.SonResetZamaniKey = None
        self.SonClearZamaniKey = None

    def __del__(self):
        pass

    def create_key(self, Sistem, Suffix):
        return f'{Sistem.Name};{Sistem.Sembol};{Sistem.Periyot};{Suffix}'

    def is_memory_null(self, Sistem, Key):
        return Sistem.NesneGetir(Key) is None

    def clear_memory(self, Sistem, Key):
        self.SonClearZamaniKey = f'{Key};CLEAR ZAMANI'
        self.SonClearZamani = datetime.datetime.now().strftime("%H:%M:%S")
        Sistem.NesneKaydet(self.SonClearZamaniKey, self.SonClearZamani)
        Sistem.NesneKaydet(Key, None)

    def reset_memory(self, Sistem, Key):
        self.SonResetZamaniKey = f'{Key};RESET ZAMANI'
        self.SonResetZamani = datetime.datetime.now().strftime("%H:%M:%S")
        Sistem.NesneKaydet(self.SonResetZamaniKey, self.SonResetZamani)
        Sistem.NesneKaydet(Key, None)

    def write_to_memory(self, Sistem, Key, Value):
        self.SonYazmaZamaniKey = f'{Key};YAZMA ZAMANI'
        self.SonYazmaZamani = datetime.datetime.now().strftime("%H:%M:%S")
        Sistem.NesneKaydet(self.SonYazmaZamaniKey, self.SonYazmaZamani)
        Sistem.NesneKaydet(Key, Value)

    def read_from_memory(self, Sistem, Key):
        self.SonOkumaZamaniKey = f'{Key};OKUMA ZAMANI'
        self.SonOkumaZamani = datetime.datetime.now().strftime("%H:%M:%S")
        Sistem.NesneKaydet(self.SonOkumaZamaniKey, self.SonOkumaZamani)
        return Sistem.NesneGetir(Key)

    def read_from_memory_as_string(self, Sistem, Key):
        return str(self.read_from_memory(Sistem, Key))

    def read_from_memory_as_integer(self, Sistem, Key):
        return int(self.read_from_memory(Sistem, Key))

    def read_from_memory_as_single(self, Sistem, Key):
        return float(self.read_from_memory(Sistem, Key))

    def read_from_memory_as_double(self, Sistem, Key):
        return float(self.read_from_memory(Sistem, Key))
