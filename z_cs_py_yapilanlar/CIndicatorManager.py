from CBase import CBase

class CIndicatorManager(CBase):
    def __init__(self):
        super().__init__()
        self.ParamsList = []
        self.ValuesList = []
        self.MaYontemList = ["Exp", "HullMA", "Simple", "TimeSeries", "Triangular", "Variable", "Volume", "Weighted", "Wilder", "ZeroLag"]
        self.MaPeriyodList = []
        self.MaPeriyodListFiboNumbers = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
        self.MaPeriyodListCommonNumbers = [5, 10, 15, 20, 30, 45, 50, 100, 200, 500, 1000]
        self.MaParamsList = []
        self.MaList = []
        self.MaDictionary = {}
        self.RsiParamsList = []
        self.RsiList = []
        self.MacdParamsList = []
        self.MacdList = []
        self.StochasticFastParamsList = []
        self.StochasticFastList = []
        self.StochasticSlowParamsList = []
        self.StochasticSlowList = []
        self.StochasticRSIParamsList = []
        self.StochasticRSIList = []
        self.StochasticOscParamsList = []
        self.StochasticOscList = []
        self.TomaParamsList = []
        self.TomaList = []

    def initialize(self, Sistem, V, Open, High, Low, Close, Volume, Lot):
        self.set_data(Sistem, V, Open, High, Low, Close, Volume, Lot)
        return self

    def reset(self, Sistem):
        self.MaParamsList.clear()
        self.MaList.clear()
        self.MaDictionary.clear()
        self.RsiParamsList.clear()
        self.RsiList.clear()
        self.MacdParamsList.clear()
        self.MacdList.clear()
        self.StochasticFastParamsList.clear()
        self.StochasticFastList.clear()
        self.StochasticSlowParamsList.clear()
        self.StochasticSlowList.clear()
        self.StochasticRSIParamsList.clear()
        self.StochasticRSIList.clear()
        self.StochasticOscParamsList.clear()
        self.StochasticOscList.clear()
        self.TomaParamsList.clear()
        self.TomaList.clear()
        return self

    def calculate_ma(self, Sistem, Source, Method, Periyod):
        return Sistem.MA(Source, Method, Periyod)

    def calculate_ma2(self, Sistem, Source, Method, Periyod):
        return Sistem.MA2(Source, Method, Periyod)

    def calculate_ma3(self, Sistem, Source, Method, Periyod):
        return Sistem.MA3(Source, Method, Periyod)

    def calculate_mam(self, Sistem, Source, Method="Weighted", *PeriyodList):
        if not PeriyodList:
            PeriyodList = (3, 5, 8, 13, 21, 34, 55, 89)
        return Sistem.MAM(Source, Method, *PeriyodList)

    def fill_ma_list(self, Sistem, Source, MethodList, PeriyodList):
        if isinstance(MethodList, str):
            MethodList = [MethodList]
        for method in MethodList:
            for periyod in PeriyodList:
                ma_param = f"{method}_{periyod}"
                self.MaParamsList.append(ma_param)
                self.MaList.append(Sistem.MA(Source, method, periyod))
        return self.MaList

    def clear_ma_list(self, Sistem):
        self.MaParamsList.clear()
        self.MaList.clear()

    def calculate_rsi(self, Sistem, Source, Periyod):
        return Sistem.RSI(Source, Periyod)

    def fill_rsi_list(self, Sistem, Source, PeriyodList):
        for periyod in PeriyodList:
            rsi_param = f"{periyod}"
            self.RsiParamsList.append(rsi_param)
            self.RsiList.append(Sistem.RSI(Source, periyod))
        return self.RsiList

    def clear_rsi_list(self, Sistem):
        self.RsiParamsList.clear()
        self.RsiList.clear()

    def calculate_macd(self, Sistem, Source, PeriyodFast, PeriyodSlow):
        return Sistem.MACD(Source, PeriyodFast, PeriyodSlow)

    def fill_macd_list(self, Sistem, Source, PeriyodFastList, PeriyodSlowList):
        for periyod_fast in PeriyodFastList:
            for periyod_slow in PeriyodSlowList:
                macd_param = f"{periyod_fast}_{periyod_slow}"
                self.MacdParamsList.append(macd_param)
                self.MacdList.append(Sistem.MACD(Source, periyod_fast, periyod_slow))
        return self.MacdList

    def clear_macd_list(self, Sistem):
        self.MacdParamsList.clear()
        self.MacdList.clear()

    def calculate_stochastic_fast(self, Sistem, Source, Periyod1, Periyod2):
        return Sistem.StochasticFast(Source, Periyod1, Periyod2)

    def fill_stochastic_fast_list(self, Sistem, Source, Periyod1List, Periyod2List):
        for periyod1 in Periyod1List:
            for periyod2 in Periyod2List:
                stochastic_fast_param = f"{periyod1}_{periyod2}"
                self.StochasticFastParamsList.append(stochastic_fast_param)
                self.StochasticFastList.append(Sistem.StochasticFast(Source, periyod1, periyod2))
        return self.StochasticFastList

    def clear_stochastic_fast_list(self, Sistem):
        self.StochasticFastParamsList.clear()
        self.StochasticFastList.clear()

    def calculate_stochastic_slow(self, Sistem, Source, Periyod1, Periyod2):
        return Sistem.StochasticSlow(Source, Periyod1, Periyod2)

    def fill_stochastic_slow_list(self, Sistem, Source, Periyod1List, Periyod2List):
        for periyod1 in Periyod1List:
            for periyod2 in Periyod2List:
                stochastic_slow_param = f"{periyod1}_{periyod2}"
                self.StochasticSlowParamsList.append(stochastic_slow_param)
                self.StochasticSlowList.append(Sistem.StochasticSlow(Source, periyod1, periyod2))
        return self.StochasticSlowList

    def clear_stochastic_slow_list(self, Sistem):
        self.StochasticSlowParamsList.clear()
        self.StochasticSlowList.clear()

    def calculate_stochastic_rsi(self, Sistem, Source, Periyod):
        return Sistem.StochasticRSI(Source, Periyod)

    def fill_stochastic_rsi_list(self, Sistem, Source, PeriyodList):
        for periyod in PeriyodList:
            stochastic_rsi_param = f"{periyod}"
            self.StochasticRSIParamsList.append(stochastic_rsi_param)
            self.StochasticRSIList.append(Sistem.StochasticRSI(Source, periyod))
        return self.StochasticRSIList

    def clear_stochastic_rsi_list(self, Sistem):
        self.StochasticRSIParamsList.clear()
        self.StochasticRSIList.clear()

    def calculate_stochastic_osc(self, Sistem, Source, Periyod1, Periyod2):
        return Sistem.StochasticOsc(Source, Periyod1, Periyod2)

    def fill_stochastic_osc_list(self, Sistem, Source, Periyod1List, Periyod2List):
        for periyod1 in Periyod1List:
            for periyod2 in Periyod2List:
                stochastic_osc_param = f"{periyod1}_{periyod2}"
                self.StochasticOscParamsList.append(stochastic_osc_param)
                self.StochasticOscList.append(Sistem.StochasticOsc(Source, periyod1, periyod2))
        return self.StochasticOscList

    def clear_stochastic_osc_list(self, Sistem):
        self.StochasticOscParamsList.clear()
        self.StochasticOscList.clear()

    def calculate_toma(self, Sistem, Source, Periyod, Percentage, Method):
        return Sistem.TOMA(Source, Periyod, Percentage, Method)

    def fill_toma_list(self, Sistem, Source, PeriyodList, PercentageList, MethodList):
        if isinstance(MethodList, str):
            MethodList = [MethodList]
        for periyod in PeriyodList:
            for percentage in PercentageList:
                for method in MethodList:
                    toma_param = f"{periyod}_{percentage}_{method}"
                    self.TomaParamsList.append(toma_param)
                    self.TomaList.append(Sistem.TOMA(Source, periyod, percentage, method))
        return self.TomaList

    def clear_toma_list(self, Sistem):
        self.TomaParamsList.clear()
        self.TomaList.clear()
