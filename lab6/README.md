Autorzy:
Kamil Suchomski
Kamil Koniak

Problem:
Zbudować prototyp maszyny do gry w "Baba Jaga patrzy".
- Narysować celownik na twarzy celu
- Nie strzelać gdy uczestnik się poddaje

Rozwiązanie:
Aplikacja wykorzystująca OpenCV do przechwytywania strumienia z kamery i wykrywania twarzy.
Aplikacja rysuje celownik na wykrytych twarzach.

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

### Zaawansowane opcje

Z metodą wykrywania twarzy DNN (wymaga dodatkowych plików):
```bash
python camera_stream.py --method dnn
```

Z niestandardową rozdzielczością (dla kamer USB):
```bash
python camera_stream.py --camera 1 --width 1280 --height 720
```

Z niestandardową liczbą klatek na sekundę:
```bash
python camera_stream.py --camera 1 --fps 60
```

<video width="640" height="480" controls>
  <source src="babajaga.mp4" type="video/mp4">
</video>



