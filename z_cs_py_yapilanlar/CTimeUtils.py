from CBase import CBase
import datetime
import time

class CTimeUtils(CBase):
    def __init__(self):
        super().__init__()
        self.ToplamGecenSureAy = 0.0
        self.ToplamGecenSureGun = 0.0
        self.ToplamGecenSureSaat = 0.0
        self.ToplamGecenSureDakika = 0.0
        self.counter5_90 = 0
        self.counter5_120 = 0
        self.counter5_150 = 0
        self.counter5_180 = 0
        self.counter5_240 = 0
        self.counter5_300 = 0
        self.counter5_480 = 0
        self.WatchDogCounter = 0
        self.WatchDogCounterStarted = False
        self.WatchDogCounterFinished = False
        self.startTimeTick = 0
        self.stopTimeTick = 0
        self.currentTimeTick = 0
        self.elapsedTimeTick = 0
        self.CurrentTimeInMSec = 0
        self.ElapsedTimeInMSec = 0
        self.ExecutionTimeInMSec = 0

    def initialize(self, Sistem, V, Open, High, Low, Close, Volume, Lot):
        self.set_data(Sistem, V, Open, High, Low, Close, Volume, Lot)
        self.reset(Sistem)
        return self

    def reset(self, Sistem):
        self.ToplamGecenSureAy = 0.0
        self.ToplamGecenSureGun = 0.0
        self.ToplamGecenSureSaat = 0.0
        self.ToplamGecenSureDakika = 0.0
        self.WatchDogCounter = 0
        self.WatchDogCounterStarted = False
        self.WatchDogCounterFinished = False
        self.startTimeTick = 0
        self.stopTimeTick = 0
        self.currentTimeTick = 0
        self.elapsedTimeTick = 0
        self.CurrentTimeInMSec = 0
        self.ElapsedTimeInMSec = 0
        self.ExecutionTimeInMSec = 0
        return self

    def gecen_zaman_bilgilerini_al(self, Sistem):
        sure_dakika = (datetime.datetime.now() - self.V[0].Date).total_seconds() / 60
        sure_saat = (datetime.datetime.now() - self.V[0].Date).total_seconds() / 3600
        sure_gun = (datetime.datetime.now() - self.V[0].Date).days
        sure_ay = sure_gun / 30.4
        self.ToplamGecenSureAy = float(sure_ay)
        self.ToplamGecenSureGun = float(sure_gun)
        self.ToplamGecenSureSaat = float(sure_saat)
        self.ToplamGecenSureDakika = float(sure_dakika)

    def get_last_execution_time(self, Sistem):
        return datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")

    def is_yeni_ay(self, Sistem, BarIndex):
        i = BarIndex
        return self.V[i].Date.month != self.V[i - 1].Date.month

    def is_yeni_hafta(self, Sistem, BarIndex):
        i = BarIndex
        yeni_gun = self.V[i - 1].Date.day != self.V[i].Date.day
        return yeni_gun and self.get_week_number(Sistem, self.V[i - 1].Date) != self.get_week_number(Sistem, self.V[i].Date)

    def is_yeni_gun(self, Sistem, BarIndex):
        i = BarIndex
        return self.V[i].Date.day != self.V[i - 1].Date.day

    def is_yeni_saat(self, Sistem, BarIndex):
        i = BarIndex
        return self.V[i].Date.hour != self.V[i - 1].Date.hour

    def is_yeni(self, Sistem, BarIndex, Periyot):
        result = False
        i = BarIndex
        if Periyot == "5S":
            pass
        elif Periyot == "10S":
            pass
        elif Periyot == "15S":
            pass
        elif Periyot == "1":
            result = self.V[i].Date.minute % 1 == 0
        elif Periyot == "2":
            result = self.V[i].Date.minute % 2 == 0
        elif Periyot == "3":
            result = self.V[i].Date.minute % 3 == 0
        elif Periyot == "4":
            result = self.V[i].Date.minute % 4 == 0
        elif Periyot == "5":
            result = self.V[i].Date.minute % 5 == 0
        elif Periyot == "10":
            result = self.V[i].Date.minute % 10 == 0
        elif Periyot == "15":
            result = self.V[i].Date.minute % 15 == 0
        elif Periyot == "20":
            result = self.V[i].Date.minute % 20 == 0
        elif Periyot == "30":
            result = self.V[i].Date.minute % 30 == 0
        elif Periyot == "45":
            result = self.V[i].Date.minute % 45 == 0
        elif Periyot == "60":
            result = self.V[i].Date.minute % 60 == 0
        elif Periyot == "90":
            if self.V[i].Date.minute % 5 == 0:
                self.counter5_90 += 1
                if self.counter5_90 == 18:
                    result = True
                    self.counter5_90 = 0
        elif Periyot == "120":
            if self.V[i].Date.minute % 5 == 0:
                self.counter5_120 += 1
                if self.counter5_120 == 24:
                    result = True
                    self.counter5_120 = 0
        elif Periyot == "150":
            if self.V[i].Date.minute % 5 == 0:
                self.counter5_150 += 1
                if self.counter5_150 == 30:
                    result = True
                    self.counter5_150 = 0
        elif Periyot == "180":
            if self.V[i].Date.minute % 5 == 0:
                self.counter5_180 += 1
                if self.counter5_180 == 36:
                    result = True
                    self.counter5_180 = 0
        elif Periyot == "240":
            if self.V[i].Date.minute % 5 == 0:
                self.counter5_240 += 1
                if self.counter5_240 == 48:
                    result = True
                    self.counter5_240 = 0
        elif Periyot == "300":
            if self.V[i].Date.minute % 5 == 0:
                self.counter5_300 += 1
                if self.counter5_300 == 60:
                    result = True
                    self.counter5_300 = 0
        elif Periyot == "480":
            if self.V[i].Date.minute % 5 == 0:
                self.counter5_480 += 1
                if self.counter5_480 == 96:
                    result = True
                    self.counter5_480 = 0
        elif Periyot == "S":
            if self.V[i].Date.minute % 5 == 0:
                self.counter5_240 += 1
                if self.counter5_240 == 48:
                    result = True
                    self.counter5_240 = 0
        elif Periyot == "G":
            result = self.is_yeni_gun(Sistem, i)
        elif Periyot == "H":
            result = self.is_yeni_hafta(Sistem, i)
        elif Periyot == "M":
            result = self.is_yeni_ay(Sistem, i)
        else:
            result = False
        return result

    def gecen_sure(self, Sistem, Ref="A"):
        sure = 0.0
        if Ref == "A":
            sure = (datetime.datetime.now() - self.V[0].Date).total_seconds() / (30.4 * 24 * 3600)
        elif Ref == "G":
            sure = (datetime.datetime.now() - self.V[0].Date).days
        elif Ref == "S":
            sure = (datetime.datetime.now() - self.V[0].Date).total_seconds() / 3600
        elif Ref == "D":
            sure = (datetime.datetime.now() - self.V[0].Date).total_seconds() / 60
        return float(sure)

    def periyot_carpan_hesapla(self, Sistem, Periyot1, Value1, Periyot2):
        k1 = self.periyot_carpan_hesapla(Sistem, Periyot1)
        k2 = self.periyot_carpan_hesapla(Sistem, Periyot2)
        value2 = Value1 * k2 / k1
        return value2

    def periyot_carpan_hesapla(self, Sistem, Periyot):
        periyotCarpanGunluk = 1.0
        if Periyot == "B":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 1 * 60
        elif Periyot == "5S":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 1 * 12
        elif Periyot == "10S":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 1 * 6
        elif Periyot == "15S":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 1 * 4
        elif Periyot == "1":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 1
        elif Periyot == "2":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 2
        elif Periyot == "3":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 3
        elif Periyot == "4":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 4
        elif Periyot == "5":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 5
        elif Periyot == "10":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 10
        elif Periyot == "15":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 15
        elif Periyot == "20":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 20
        elif Periyot == "30":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 30
        elif Periyot == "45":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 45
        elif Periyot == "60":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 60
        elif Periyot == "90":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 90
        elif Periyot == "120":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 120
        elif Periyot == "150":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 150
        elif Periyot == "180":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 180
        elif Periyot == "240":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 240
        elif Periyot == "300":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 300
        elif Periyot == "360":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 360
        elif Periyot == "420":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 420
        elif Periyot == "480":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 480
        elif Periyot == "S":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 240
        elif Periyot == "G":
            periyotCarpanGunluk = 1.0 * 8 * 60 / 480
        else:
            periyotCarpanGunluk = 0
            Sistem.Mesaj("Sistem.Periyot tanımlı değerler arasından seçilmelidir! ")
        return periyotCarpanGunluk

    def start_watch_dog_counter(self, Sistem, Counter):
        self.WatchDogCounter = Counter
        self.WatchDogCounterStarted = True
        self.WatchDogCounterFinished = False

    def stop_watch_dog_counter(self, Sistem):
        self.WatchDogCounterStarted = False

    def watch_dog_counter_is_started(self, Sistem):
        return self.WatchDogCounterStarted

    def step_watch_dog_counter(self, Sistem):
        self.WatchDogCounter -= 1
        if self.WatchDogCounter <= 0:
            self.WatchDogCounterFinished = True
        return self.WatchDogCounter

    def watch_dog_counter_is_finished(self, Sistem):
        return self.WatchDogCounterFinished

    def start_timer(self, Sistem):
        self.startTimeTick = time.perf_counter_ns() // 1_000_000
        return self.startTimeTick

    def stop_timer(self, Sistem):
        self.stopTimeTick = time.perf_counter_ns() // 1_000_000
        return self.stopTimeTick

    def get_current_time_in_msec(self, Sistem):
        self.currentTimeTick = time.perf_counter_ns() // 1_000_000
        self.CurrentTimeInMSec = self.currentTimeTick
        return self.CurrentTimeInMSec

    def get_elapsed_time_in_msec(self, Sistem):
        self.elapsedTimeTick = (time.perf_counter_ns() // 1_000_000) - self.startTimeTick
        self.ElapsedTimeInMSec = self.elapsedTimeTick
        return self.ElapsedTimeInMSec

    def get_execution_time_in_msec(self, Sistem):
        self.elapsedTimeTick = self.stopTimeTick - self.startTimeTick
        self.ExecutionTimeInMSec = self.elapsedTimeTick
        return self.ExecutionTimeInMSec

    def get_date_time(self, DateTimeStr):
        return datetime.datetime.strptime(DateTimeStr, "%d.%m.%Y %H:%M:%S")

    def get_date_time_from_date_and_time(self, DateStr, TimeStr):
        DateTimeStr = DateStr + " " + TimeStr
        return self.get_date_time(DateTimeStr)

    def get_date(self, DateStr):
        return datetime.datetime.strptime(DateStr, "%d.%m.%Y")

    def get_time(self, TimeStr):
        return datetime.datetime.strptime(TimeStr, "%H:%M:%S").time()

    def compare_dates(self, DateTime1, DateTime2):
        if DateTime1 < DateTime2:
            return -1
        elif DateTime1 > DateTime2:
            return 1
        else:
            return 0

    def compare_times(self, DateTime1, DateTime2):
        if DateTime1.time() < DateTime2.time():
            return -1
        elif DateTime1.time() > DateTime2.time():
            return 1
        else:
            return 0

    def get_date_time_from_components(self, year, month, day, hour, minute, second):
        return datetime.datetime(year, month, day, hour, minute, second)

    def to_long_date_string(self, dateTime):
        return dateTime.strftime("%A, %B %d, %Y")

    def to_long_time_string(self, dateTime):
        return dateTime.strftime("%H:%M:%S")

    def get_short_date_string(self, dateTime):
        return dateTime.strftime("%m/%d/%Y")

    def get_short_time_string(self, dateTime):
        return dateTime.strftime("%I:%M %p")

    def get_current_date_time(self):
        return datetime.datetime.now()

    def get_current_date(self):
        return datetime.datetime.now().date()

    def get_current_time(self):
        return datetime.datetime.now().time()

    def get_bar_date_time(self, Sistem, BarIndex):
        i = BarIndex
        V = Sistem.GrafikVerileri
        return V[i].Date

    def get_bar_date(self, Sistem, BarIndex):
        i = BarIndex
        V = Sistem.GrafikVerileri
        return V[i].Date

    def get_bar_time(self, Sistem, BarIndex):
        i = BarIndex
        V = Sistem.GrafikVerileri
        return V[i].Date

    def is_bar(self, Sistem, BarIndex, BarZamani):
        i = BarIndex
        V = Sistem.GrafikVerileri
        return V[i].Date.strftime("%H:%M") == BarZamani

    def is_poz_kapat_valid(self, Sistem, BarIndex, PozKapatZamani):
        i = BarIndex
        V = Sistem.GrafikVerileri
        return V[i].Date.strftime("%H:%M") == PozKapatZamani

    def check_bar_time_with(self, Sistem, BarIndex, BarZamani):
        i = BarIndex
        V = Sistem.GrafikVerileri
        return self.compare_times(self.get_bar_date_time(Sistem, i), self.get_time(BarZamani))

    def check_bar_date_with(self, Sistem, BarIndex, BarZamani):
        i = BarIndex
        V = Sistem.GrafikVerileri
        return self.compare_dates(self.get_bar_date_time(Sistem, i), self.get_date(BarZamani))

    def check_bar_date_time_with(self, Sistem, BarIndex, BarZamani):
        i = BarIndex
        V = Sistem.GrafikVerileri
        return self.compare_dates(self.get_bar_date_time(Sistem, i), self.get_date_time(BarZamani))

    def is_gun_sonu_poz_kapat_sinyali_geldi_sadece_analiz(self, Sistem, BarIndex):
        i = BarIndex
        V = Sistem.GrafikVerileri
        BarCount = len(V)
        return (i < BarCount - 1 and V[i].Date.day != V[i + 1].Date.day)

    def is_ay_sonu_poz_kapat_sinyali_geldi_sadece_analiz(self, Sistem, BarIndex):
        i = BarIndex
        V = Sistem.GrafikVerileri
        BarCount = len(V)
        return (i < BarCount - 1 and V[i].Date.month != V[i + 1].Date.month)

    def get_week_number(self, Sistem, date):
        return date.isocalendar()[1]

    def get_week_number_from_bar_index(self, Sistem, BarIndex):
        i = BarIndex
        V = Sistem.GrafikVerileri
        return self.get_week_number(Sistem, V[i].Date)
