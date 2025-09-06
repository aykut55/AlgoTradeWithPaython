from datetime import datetime

class CConfigManager:
    def __init__(self):
        self.son_okuma_zamani = None
        self.son_yazma_zamani = None
        self.son_reset_zamani = None
        self.son_clear_zamani = None
        self.son_okuma_zamani_key = None
        self.son_yazma_zamani_key = None
        self.son_reset_zamani_key = None
        self.son_clear_zamani_key = None

    def __del__(self):
        pass

    def create_key(self, sistem, suffix):
        return f"{sistem.Name};{sistem.Sembol};{sistem.Periyot};{suffix}"

    def set_value(self, sistem, key, value):
        self.son_yazma_zamani_key = f"{key};YAZMA ZAMANI"
        self.son_yazma_zamani = datetime.now().strftime("%H:%M:%S")
        sistem.nesne_kaydet(self.son_yazma_zamani_key, self.son_yazma_zamani)
        sistem.nesne_kaydet(key, value)

    def get_value(self, sistem, key):
        self.son_okuma_zamani_key = f"{key};OKUMA ZAMANI"
        self.son_okuma_zamani = datetime.now().strftime("%H:%M:%S")
        sistem.nesne_kaydet(self.son_okuma_zamani_key, self.son_okuma_zamani)
        return sistem.nesne_getir(key)
