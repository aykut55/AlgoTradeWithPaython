class CBase:
    def __init__(self):
        self.Id = 0
        self.V = None
        self.Open = []
        self.High = []
        self.Low = []
        self.Close = []
        self.Volume = []
        self.Lot = []
        self.BarCount = 0
        self.LastBarIndex = 0

    def __del__(self):
        pass

    def show_message(self, Sistem, Message):
        Sistem.Mesaj(Message)

    def set_data(self, Sistem, V, Open, High, Low, Close, Volume, Lot):
        self.V = V
        self.Open = Open
        self.High = High
        self.Low = Low
        self.Close = Close
        self.Volume = Volume
        self.Lot = Lot
        self.BarCount = len(V)
        self.LastBarIndex = len(V) - 1

    # Sadece SetData'yi kullanmaya karar verdim
    # public int SetGrafikVerileri(dynamic Sistem, dynamic GrafikVerileri)
    # {
    #     return parseData(Sistem, GrafikVerileri);
    # }
    #
    # private int parseData(dynamic Sistem, dynamic Data)
    # {
    #     int result = 0;
    #
    #     try
    #     {
    #         Open = Sistem.GrafikFiyatOku(Data, "Acilis");
    #         High = Sistem.GrafikFiyatOku(Data, "Yuksek");
    #         Low = Sistem.GrafikFiyatOku(Data, "Dusuk");
    #         Close = Sistem.GrafikFiyatOku(Data, "Kapanis");
    #         Volume = Sistem.GrafikFiyatOku(Data, "Hacim");
    #         Lot = Sistem.GrafikFiyatOku(Data, "Lot");
    #
    #         V = Data;
    #         BarCount = V.Count;
    #         LastBarIndex = V.Count - 1;
    #
    #         result = 0;
    #     }
    #     catch (Exception error)
    #     {
    #         result = -1;
    #     }
    #
    #     return result;
    # }
