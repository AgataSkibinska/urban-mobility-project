
import json
import os

import contextily as ctx
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from tqdm import tqdm

tqdm.pandas()

# to
df = df.iloc[0:14241]

chosen_cols = [
    'IDENTYFIKACJA ANKIETY_ID_LOS (numer kwestionariusza)',
    'DANE O RESPONDENCIE_Przedział wiekowy',
    'DANE O RESPONDENCIE_Płeć',
    'OPIS PODRÓŻY "ŹRÓDŁO"_Nr rejonu',
    'OPIS PODRÓŻY "CEL"_Nr rejonu'
]

df_filtered = df[chosen_cols]
df_filtered.columns = [
    '',
    ''
]

data.columns = [
    # 'Z jakiego środka transportu zbiorowego korzysta Pan/Pani najczęściej?',
    'Jak Pan/Pani ocenia wygodę jazdy pojazdami komunikacji zbiorowej?',
    'Jak Pan/Pani ocenia punktualność komunikacji zbiorowej we Wrocławiu?',
    'Jak ocenia Pan/Pani efekty dotychczasowych działań związanych z rozbudową systemu rowerowego we Wrocławiu (dróg i parkingów rowerowych)?',
    'PIESZO Niekorzystne ustawienia sygnalizacji świetlnej',
    'PIESZO Brak chodników i konieczność poruszania się jezdnią/poboczem/wydeptaną ścieżką',
    'PIESZO Zły stan nawierzchni chodników',
    'PIESZO Zastawianie chodników przez parkujące samochody',
    'PIESZO Niebezpieczne zachowania kierowców',
    'PIESZO Zagrożenie ze strony rowerzystów poruszających się chodnikami',
    'PIESZO Zbyt wysokie krawężniki',
    'PIESZO Brak bieżącego utrzymania czystości/odśnieżania',
    'PIESZO Niewystarczająca liczba przejść dla pieszych',
    'PIESZO Brak miejsc wypoczynku na trasie dojścia (np. ławki, zieleń)',
    'PIESZO Uciążliwy ruch kołowy',
    'PIESZO Niewłaściwe oświetlenie ciągów pieszych',
    'Liczba osób w gospodarstwie domowym [ogółem]',
    'Liczba osób w gospodarstwie domowym [powyżej 6 roku życia]',
    'Samochód prywatny, zarejestrowany na osobę z gosp. domowego',
    'Samochód prywatny, nie zarejestrowany na osobę z gosp. domowego [użyczone]',
    'Samochód służbowy',
    'Rower',
    'Przedział wiekowy',
    'Płeć',
    'Zajęcie podstawowe',
    'Kondycja fizyczna',
    'Opieka nad innymi osobami',
    'Posiadanie prawa jazdy kat. B',
    'Posiadanie biletu okresowego',
    'Posiadanie ulgi na przejazd komunikację zbiorową',
    'ŹRÓDŁO Z jakiego miejsca',
    'ŹRÓDŁO Nr rejonu',
    'CEL Do jakiego miejsca',
    'CEL Nr rejonu',
    'Pora dnia (godzina)',
    'Motywacje (skąd-dokąd)',
    'Środek transportu grupa'
]

# ------------------ 1
'PIESZO Niekorzystne ustawienia sygnalizacji świetlnej'

var_map_dict = {
    'Nie': 0,
    'nie': 0,
    'Tak': 1,
    'tak': 1
}

data.replace({'PIESZO Niekorzystne ustawienia sygnalizacji świetlnej': var_map_dict}, inplace=True)

# ------------------ 2
'PIESZO Brak chodników i konieczność poruszania się jezdnią/poboczem/wydeptaną ścieżką'

var_map_dict = {
    'Nie': 0,
    'nie': 0,
    'Tak': 1,
    'tak': 1
}

data.replace({'PIESZO Brak chodników i konieczność poruszania się jezdnią/poboczem/wydeptaną ścieżką': var_map_dict}, inplace=True)

# ------------------ 3
'PIESZO Zły stan nawierzchni chodników'

var_map_dict = {
    'Nie': 0,
    'nie': 0,
    'Tak': 1,
    'tak': 1
}

data.replace({'PIESZO Zły stan nawierzchni chodników': var_map_dict}, inplace=True)

# ------------------ 4
'PIESZO Zastawianie chodników przez parkujące samochody'

var_map_dict = {
    'Nie': 0,
    'nie': 0,
    'Tak': 1,
    'tak': 1
}

data.replace({'PIESZO Zastawianie chodników przez parkujące samochody': var_map_dict}, inplace=True)

# ------------------ 5
'PIESZO Niebezpieczne zachowania kierowców'

var_map_dict = {
    'Nie': 0,
    'nie': 0,
    'Tak': 1,
    'tak': 1
}

data.replace({'PIESZO Niebezpieczne zachowania kierowców': var_map_dict}, inplace=True)

# ------------------ 6
'PIESZO Zagrożenie ze strony rowerzystów poruszających się chodnikami'

var_map_dict = {
    'Nie': 0,
    'nie': 0,
    'Tak': 1,
    'tak': 1
}

data.replace({'PIESZO Zagrożenie ze strony rowerzystów poruszających się chodnikami': var_map_dict}, inplace=True)

# ------------------ 7
'PIESZO Zbyt wysokie krawężniki'

var_map_dict = {
    'Nie': 0,
    'nie': 0,
    'Tak': 1,
    'tak': 1
}

data.replace({'PIESZO Zbyt wysokie krawężniki': var_map_dict}, inplace=True)

# ------------------ 8
'PIESZO Brak bieżącego utrzymania czystości/odśnieżania'

var_map_dict = {
    'Nie': 0,
    'nie': 0,
    'Tak': 1,
    'tak': 1
}

data.replace({'PIESZO Brak bieżącego utrzymania czystości/odśnieżania': var_map_dict}, inplace=True)

# ------------------ 9
'PIESZO Niewystarczająca liczba przejść dla pieszych'

var_map_dict = {
    'Nie': 0,
    'nie': 0,
    'Tak': 1,
    'tak': 1
}

data.replace({'PIESZO Niewystarczająca liczba przejść dla pieszych': var_map_dict}, inplace=True)

# ------------------ 10
'PIESZO Brak miejsc wypoczynku na trasie dojścia (np. ławki, zieleń)'

var_map_dict = {
    'Nie': 0,
    'nie': 0,
    'Tak': 1,
    'tak': 1
}

data.replace({'PIESZO Brak miejsc wypoczynku na trasie dojścia (np. ławki, zieleń)': var_map_dict}, inplace=True)

# ------------------ 11
'PIESZO Uciążliwy ruch kołowy'

var_map_dict = {
    'Nie': 0,
    'nie': 0,
    'Tak': 1,
    'tak': 1
}

data.replace({'PIESZO Uciążliwy ruch kołowy': var_map_dict}, inplace=True)

# ------------------ 12
'PIESZO Niewłaściwe oświetlenie ciągów pieszych'

var_map_dict = {
    'Nie': 0,
    'nie': 0,
    'Tak': 1,
    'tak': 1
}

data.replace({'PIESZO Niewłaściwe oświetlenie ciągów pieszych': var_map_dict}, inplace=True)

# ---------------------
var_map_dict = {
    '6-15 (dzieci)': 0,
    '16-19 (młodzież)': 1,
    '20-24 (wiek studencki)': 2,
    '25-44 (młodsi pracownicy)': 3,
    '45-60 (starsi pracownicy kobiety)': 4,
    '45-65 (starsi pracownicy mężczyźni)': 4,
    '61 i więcej (emeryci kobiety)': 5,
    '66 i więcej (emeryci mężczyźni)': 5
}

data.replace({'Przedział wiekowy': var_map_dict}, inplace=True)

# ---------------------
var_map_dict = {
    'Kobieta': 1,
    'Mężczyzna': 0
}

data.replace({'Płeć': var_map_dict}, inplace=True)


