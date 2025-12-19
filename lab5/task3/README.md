Autorzy:<br>
Kamil Suchomski<br>
Kamil Koniak<br>

Problem:<br>
Naucz sieć rozpoznawać ubrania.<br>

Rozwiązanie:<br>
Wytrenowaliśmy model z wykorzystaniem zbioru zalandoresearch/fashion-mnist<br>
Model to sieć splotowa korzystająca z augmentacji.<br>
Model jest zapisany w folderze `nai/lab5/task3/models/model_cnn_augment.keras`<br>

![Model](model.png)<br>

Strata na zbiorze testowym: 0.278 <br>
Dokładność na zbiorze testowym: 90,49% <br>
Krzywa uczenia:<br>
![krzywa_uczenia](metrics/model_cnn_augment_history.png)<br>
Macież pomyłek:<br>
![macierz_pomyłek](metrics/model_cnn_augment_confusion.png)<br>

W folderze `pics` znajdują się przykładowe zdjęcia do testowania pobrane z internetu.<br>

Aby uruchomić predykcję należy podać ścieżkę do obrazka:
```
cd lab5
cd task3
python predict.py pics\01.png
```

Przykłady użycia:<br>
![usecase](usecase01.png)<br>
![usecase](usecase02.png)<br>
