class CUtils:
    def __init__(self):
        pass

    def __del__(self):
        pass

    def initialize(self, Sistem):
        return self

    def reset(self, Sistem):
        return self

    def get_bool(self, Value, trueValue="1"):
        if isinstance(Value, str):
            return Value == trueValue
        elif isinstance(Value, int):
            return Value == trueValue
        return bool(Value)

    def get_integer16(self, Value):
        return int(Value)

    def get_integer32(self, Value):
        return int(Value)

    def get_integer64(self, Value):
        return int(Value)

    def get_integer(self, Value):
        return self.get_integer32(Value)

    def get_float(self, Value):
        return float(Value)

    def get_double(self, Value):
        return float(Value)

    def get_max(self, lhs, rhs):
        return max(lhs, rhs)

    def yukari_kesti(self, Sistem, i, ListX, ListY, esitlikDahil=True):
        if esitlikDahil:
            res = ListX[i - 1] < ListY[i - 1] and ListX[i] >= ListY[i]
        else:
            res = ListX[i - 1] < ListY[i - 1] and ListX[i] > ListY[i]
        return res

    def asagi_kesti(self, Sistem, i, ListX, ListY, esitlikDahil=True):
        if esitlikDahil:
            res = ListX[i - 1] > ListY[i - 1] and ListX[i] <= ListY[i]
        else:
            res = ListX[i - 1] > ListY[i - 1] and ListX[i] < ListY[i]
        return res

    def yukari_kesti_seviye(self, Sistem, i, ListX, Seviye, esitlikDahil=True):
        if esitlikDahil:
            res = ListX[i - 1] < Seviye and ListX[i] >= Seviye
        else:
            res = ListX[i - 1] < Seviye and ListX[i] > Seviye
        return res

    def asagi_kesti_seviye(self, Sistem, i, ListX, Seviye, esitlikDahil=True):
        if esitlikDahil:
            res = ListX[i - 1] > Seviye and ListX[i] <= Seviye
        else:
            res = ListX[i - 1] > Seviye and ListX[i] < Seviye
        return res

    def buyuk(self, Sistem, i, ListX, ListY):
        return ListX[i] > ListY[i]

    def kucuk(self, Sistem, i, ListX, ListY):
        return ListX[i] < ListY[i]

    def buyuk_esit(self, Sistem, i, ListX, ListY):
        return ListX[i] >= ListY[i]

    def kucuk_esit(self, Sistem, i, ListX, ListY):
        return ListX[i] <= ListY[i]

    def esit(self, Sistem, i, ListX, ListY):
        return ListX[i] == ListY[i]

    def esit_degil(self, Sistem, i, ListX, ListY):
        return ListX[i] != ListY[i]

    def ustunde(self, Sistem, i, ListX, ListY):
        return self.buyuk(Sistem, i, ListX, ListY)

    def altinda(self, Sistem, i, ListX, ListY):
        return self.kucuk(Sistem, i, ListX, ListY)

    def buyuk_coklu(self, Sistem, i, *args):
        res = True
        for j in range(len(args) - 1):
            if args[j] is None or args[j+1] is None:
                break
            res = res and (args[j][i] > args[j+1][i])
        return res

    def kucuk_coklu(self, Sistem, i, *args):
        res = True
        for j in range(len(args) - 1):
            if args[j] is None or args[j+1] is None:
                break
            res = res and (args[j][i] < args[j+1][i])
        return res

    def liste_topla(self, Sistem, Source, K):
        return [x + K for x in Source]

    def liste_cikar(self, Sistem, Source, K):
        return [x - K for x in Source]

    def liste_carp(self, Sistem, Source, K):
        return [x * K for x in Source]

    def liste_bol(self, Sistem, Source, K):
        return [x / K for x in Source]

    def liste_yuzde(self, Sistem, Source, K, Offset=0):
        return [x * K / 100 + Offset for x in Source]

    def liste_yuzde_liste(self, Sistem, Source, K, OffsetList=None):
        if OffsetList:
            return [Source[i] * K / 100 + OffsetList[i] for i in range(len(Source))]
        else:
            return [x * K / 100 for x in Source]

    def liste_topla_liste(self, Sistem, Source1, Source2):
        return [Source1[i] + Source2[i] for i in range(len(Source1))]

    def liste_cikar_liste(self, Sistem, Source1, Source2):
        return [Source1[i] - Source2[i] for i in range(len(Source1))]

    def liste_elemanlarini_resetle(self, Sistem, begIndex, barCount, lst, lowerLimit, upperLimit):
        return self.liste_elemanlarini_set_et(Sistem, begIndex, barCount, lst, lowerLimit, upperLimit, 0)

    def liste_elemanlarini_set_et(self, Sistem, begIndex, barCount, lst, lowerLimit, upperLimit, newValue=0):
        newList = list(lst)
        for i in range(begIndex, barCount):
            if lowerLimit <= lst[i] <= upperLimit:
                newList[i] = newValue
        return newList

    def get_new_list(self, Sistem, Count, Value):
        return [Value] * Count

    def get_zero_level_list(self, Sistem, Count):
        return self.get_new_list(Sistem, Count, 0.0)

    def get_new_list_from_src(self, Sistem, SrcList):
        return list(SrcList)

    def get_distance_list_as_percentage(self, Sistem, Source, High, Low):
        return [(Source[i] - Low[i]) * 100 / (High[i] - Low[i]) for i in range(len(Source))]

    def get_distance_list_as_percentage_ref(self, Sistem, Src, Ref):
        return [(Src[i] - Ref[i]) / Ref[i] * 100 for i in range(len(Src))]

    def get_kairi(self, Sistem, Src, Ref):
        return self.get_distance_list_as_percentage_ref(Sistem, Src, Ref)

    def get_kairi_ma(self, Sistem, Src, MaPeriyot, MaYontem="Simple"):
        return self.get_distance_list_as_percentage_ref(Sistem, Src, Sistem.MA(Src, MaYontem, MaPeriyot))

    def sort_the_signals2(self, Sistem):
        return Sistem.Liste(Sistem.BarSayisi, 0)

    def get_the_list_sort_status(self, Sistem, i, *args):
        statusDict = {}
        for j, lst in enumerate(args):
            if lst is not None:
                statusDict[f"{j+1:02d}"] = lst[i]

        sorted_status = sorted(statusDict.items(), key=lambda item: item[1])
        
        res = ""
        for key, value in sorted_status:
            res += key + "-"

        return res.strip('-')

    def create_levels(self, Sistem, MaxLevelNumber=100, UseFloatingKeys=False):
        BarCount = Sistem.BarSayisi
        results = {}
        if UseFloatingKeys:
            for i in range(MaxLevelNumber + 1):
                key = float(i)
                results[key] = Sistem.Liste(BarCount, i)
        else:
            for i in range(MaxLevelNumber + 1):
                key = i
                results[key] = Sistem.Liste(BarCount, i)
        return results