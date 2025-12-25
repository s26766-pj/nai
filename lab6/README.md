## Autorzy:<br>
Kamil Suchomski<br>
Kamil Koniak<br>

## Problem:<br>
Zbudować prototyp maszyny do gry w "Baba Jaga patrzy".
- Narysować celownik na twarzy celu
- Nie strzelać gdy uczestnik się poddaje

## Rozwiązanie:<br>
Aplikacja wykorzystująca OpenCV do przechwytywania strumienia z kamery.<br>
Gdy obiekt zostaje zidentyfikowany pojawia się komunikat "HANDS UP" oraz aplikacja umieszcza celownik między oczami.<br>
Gdy obiekt poddaje się (podnosi ręce do góry), pojawia się komunikat "BACK AWAY".<br>
Gdy obiekt nie stosuje się do poleceń zostaje "zastrzelony".

## Prezentacja
`babajaga.mp4` - plik wideo z demonstracją programu

## Instalacja

```bash
cd lab6
pip install -r requirements.txt
```

## Uruchomienie

### Wykrywanie dostępnych kamer USB

Najpierw sprawdź, które kamery są dostępne:
```bash
python camera_stream.py --list-cameras
```

### Podstawowe użycie

Podstawowe uruchomienie (używa domyślnej kamery, zwykle index 0):
```bash
python camera_stream.py
```

Użycie konkretnej kamery USB (np. kamera o indeksie 1):
```bash
python camera_stream.py --camera 1
```

Z niestandardową rozdzielczością (dla kamer USB):
```bash
python camera_stream.py --camera 1 --width 1280 --height 720
```

Z niestandardową liczbą klatek na sekundę:
```bash
python camera_stream.py --camera 1 --fps 60
```


