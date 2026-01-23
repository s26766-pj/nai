# Autorzy:
Kamil Suchomski s21974<br>
Kamil Koniak s26766

# Problem:

Chcemy stworzyć autonomiczny model samochodu bazujący na Wltoys K969, samodzielnie jeżdżący po dywanie z drogą.
(dywan.jfif)

- Nie mamy wiedzy eksperta
- Wyuczony model korzysta tylko z trzech reakcji: lewo [0..1], prawo [0..1], 'gaz' [0..1]
- Nie będziemy korzystać z hamulca ze względu na zachowanie rzeczywistego modelu - hamowanie prądem zwarciowym na silniku trwa ułamek sekundy, po czym model jedzie do tyłu
- Korzystamy tylko z tego co już mamy pod ręką - ze względu na ograniczony czas, nie zdążymy zamówić odpowiednich części, co wymusza zastosowanie pewnych obejść, np. korzystanie z wbudowanego odbiornika zintegrowanego z regulatorem silnika
- Model musi jeździć powoli, ze względu na możliwość pojawienia się opóznień podczas przesyłu obrazu z kamery
 

# Rozwiązanie:

Projekt bazuje na następujących komponentach:
- model pojazdu: Wltoys K969
- ESP32 CAM - montaż na modelu
- Algorytmie Soft Actor-Critic (SAC)
- symulator CarRacing-v3 (Gymnasium) do treningu modelu RL
- Wrapper wyjścia modelu w celu nałożenia ograniczeń na rzeczywisty model pojazdu
- Transferze wiedzy wytrenowanego modelu z symulatora do świata rzeczywistego, poprzez dotrenowanie modelu na zapisie (nagrania) realnej jazdy modelu (sterowanego przez nas) po dywanie z drogą (obraz z kamery + stany przycisków na kontrolerze)
- Komunikacja pomiędzy ESP32CAM, laptopem i ESP w pilocie samochodu za pomocą WiFi
- Finalnie model samochdu jeździ samodzielnie po dywanie z drogą

Pipeline:<br>
obraz z ESP32CAM na modelu samochodu -> laptop z uruchomionym modelem RL -> ESP32 podłączony do pilota pojazdu -> reakcja modelu samochodu

# Stan projektu:

Ze względu na istotne ograniczenie czasowe udało się osiągnąć:

- mapowanie przycisków kontrolera PS5, który miał służyć do nagrywania wartości pozycji kontrolera modelu samochodu
- wytrenowany model RL w symulatorze

Trening modelu tylko na CPU zajął łącznie ok. 8h (samego treningu), cały proces trenowania wiązał się ze zmienianiem parametrów (ok. 100000 iteracji) oraz zmianami parametrów wrappera.

Projekt ze względu na to, że nas bardzo 'wciągnał' zamierzamy kontynuować i doprowadzić do finalnego etapu - samodzielnej jazdy modelu samochodu po dywanie z drogą.

## Instalacja

Instalacja zależności:
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install "gymnasium[box2d]" stable-baselines3 torch torchvision opencv-python
```


## Użycie

### Trening agenta

Trenowanie agenta SAC:
1. Start treningu:
```bash
python train_sac.py --steps 500000 --n_envs 4 --ent_coef 0.2 --learning_starts 20000 --buffer_size 200000
```

2. Kontynuacja treningu ze zmienionymi parametrami:
```bash
python train_sac.py --steps 500000 --n_envs 4 --resume ".\models\sac_carracing_20260114_170044\emergency_model.zip" --ent_coef 0.1 --eval_every 20000 --eval_episodes 10
```

Treningu spowoduje:
- Uruchomienie 500000 gier, z czego pierwsze 20000 jest poświęconych na samą eksplorację środowiska przez agenta
- Zapisze aktualny model (emergency_model.zip) w przypadku ręcznego przerwania treningu (Ctrl + C)
- Zapisuje 'checkpointy' wg wartości przekazanego argumentu
- Zapisze ostateczny model w pliku "final_model.zip"
- Zapisze najlepszy model w pliku `best_model.zip`

## Wyniki

Po treningu agent powinien nauczyć się:
- Trzymać się drogi
- Zwalniać przed ostrymi zakrętami
- W przypadku wypadnięcia poza tor powrócić
- Kiedy zacznie kręcić 'bączki', zredukuje gaz i skręt
- Osiągać wyższe wyniki w czasie

## Postępy w treningu

Po około 100000 iteracji, agent potrafił pokonywać łagodne zakręty i trzymał się drogi
Po około 300000 iteracji nauczył się redukować gaz i skręt kół jeśli zaczął kręcić 'bączki'
Po około 500000 iteracji agent potrafi zwolnić przed ostrymi zakrętami - nie zawsze mu się to udaje ;)

## Struktura projektu

```
lab7/
└── drive-additional-project
    ├── eval.py      	   # ewaluacja modelu + render z podglądem
    ├── train_sac.py       # skrypt trenujący
    ├── wrappers.py        # środowisko - symulator CarRacing-v3 z dodatkowymi ograniczeniami przygotowanymi pod przeniesienie modelu do działania na realnym modelu samochodu
    ├── demo_capture.mp4    # Film demonstrujący działanie 
    └── README.md          # Ten plik
```