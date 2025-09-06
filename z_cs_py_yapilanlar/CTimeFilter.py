class CTimeFilter:
    def __init__(self):
        self.Trader = None

    def __del__(self):
        pass

    def initialize(self, Sistem, Trader):
        self.Trader = Trader
        return self

    def reset(self, Sistem):
        return self
