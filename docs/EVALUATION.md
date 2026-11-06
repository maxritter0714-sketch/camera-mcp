# Evaluation

Gliederung nach KONZEPT.md, Abschnitt 10. Messungen sind reproduzierbar mit
`uv run python scripts/benchmark.py --camera <id> [--runs 50] [--json <datei>]`.

| # | Teil | Stand |
|---|---|---|
| 1 | Latenz pro Tool | CPU gemessen; GPU folgt in M6 |
| 2 | Antwortgröße | gemessen |
| 3 | Erkennungsqualität | gemessen: 37 Bilder mit Handzählung, ein Video |
| 4 | LLM-Aufgabenerfolg | offen, wartet auf Evaluationsfragen |
| 5 | Robustheit | live geprüft, Fehlerfälle und Stream-Abbruch über HTTP und RTSP |

## Messaufbau

- **Rechner:** Haupt-PC, Windows 11, Python 3.12.10, nur CPU (`device: auto` → `cpu`).
- **Modell:** `yolo26n.pt`, `imgsz: 640`, `default_confidence: 0.4`.
- **Kamera:** `desk_still`, ein 1920×1080-Standbild der Handy-Kamera als Datei-Quelle. Damit ist
  jeder Lauf dieselbe Szene; Netzwerk und Stream gehen nicht in die Messung ein.
- **Messweg:** Das Skript startet den Server als Subprozess und ruft ihn als echter MCP-Client über
  stdio auf, also auf demselben Weg wie Claude. Gemessen wird die Zeit vom Senden des Aufrufs bis
  zur vollständigen Antwort.
- **Ablauf:** pro Fall 3 Aufwärmaufrufe, danach 50 gemessene Aufrufe.
- **Datum:** 2026-11-06.

Einschränkungen: ein Rechner, ein Lauf, eine Szene. Während der Messung versuchte der Server im
Hintergrund, die ausgeschaltete Handy-Kamera zu erreichen; das kostet kaum CPU, ist aber nicht
ganz ausgeschlossen.

## 1. Latenz pro Tool (CPU)

| Aufruf | Median ms | p95 ms | Inferenz ms (Median) |
|---|---:|---:|---:|
| `get_camera_info` | 1,1 | 1,5 | – |
| `get_frame` (1024 px) | 9,8 | 10,6 | – |
| `detect_objects` | 19,9 | 21,7 | 18,4 |
| `detect_objects` (`include_image`) | 29,8 | 32,9 | 18,5 |
| `count_in_zone` (`tastatur`) | 20,4 | 22,2 | – |

**Beobachtungen**

- Der Aufwand für Protokoll und Server liegt bei etwa 1–1,5 ms. Das zeigen `get_camera_info`
  (1,1 ms gesamt) und die Differenz zwischen Gesamtzeit und Inferenz bei `detect_objects`.
- Die Inferenz macht über 90 % von `detect_objects` aus. Eine GPU (M6) wirkt also fast
  vollständig auf die Antwortzeit durch.
- Das Bild bei `include_image` kostet rund 10 ms zusätzlich (Zeichnen, Skalieren, JPEG), etwa so
  viel wie `get_frame` allein.
- `count_in_zone` kostet dasselbe wie `detect_objects`; die Zonenzuordnung fällt nicht ins Gewicht.
- Alle Werte liegen weit unter der Zeit, die ein LLM zum Formulieren der Antwort braucht. Für den
  Dialog ist die Latenz auf der CPU kein Engpass.

## 2. Antwortgröße

### Tool-Antworten

| Aufruf | Bytes | ~Tokens |
|---|---:|---:|
| `get_camera_info` (eine Kamera) | 356 | 89 |
| `count_in_zone` | 386 | 97 |
| `detect_objects` | 513 | 129 |
| `get_frame` (1024 px) | 99 526 | 809 |
| `detect_objects` (`include_image`) | 113 993 | 906 |

### Bild nach Breite (JPEG-Qualität 80)

Gemessen mit denselben Funktionen, die der Server nutzt (`resize_to_width`, `encode_jpeg`), aber ohne
das konfigurierte Limit `max_image_width: 1024`. So ist auch 1600 px vergleichbar.

| Breite | JPEG Bytes | Base64 Bytes | Standard-Stufe: sieht | Tokens | Hochauflösend: sieht | Tokens |
|---:|---:|---:|---|---:|---|---:|
| 512 | 21 491 | 28 656 | 512×288 | 209 | 512×288 | 209 |
| 1024 | 74 550 | 99 400 | 1024×576 | 777 | 1024×576 | 777 |
| 1600 | 169 343 | 225 792 | 1456×819 | 1560 | 1600×900 | 1914 |

**Wie die Tokens geschätzt sind**

- **Text:** etwa Bytes / 4, eine grobe Faustregel für JSON.
- **Bild:** nach der Anthropic-Dokumentation (Stand 2026-11-06) kostet ein Bild
  ⌈Breite / 28⌉ × ⌈Höhe / 28⌉ Tokens. Größere Bilder verkleinert Claude vorher selbst:
  - Standard-Stufe (Modelle bis Claude 4.6): lange Kante ≤ 1568 px, ≤ 1568 Tokens.
  - Hochauflösende Stufe (Claude 4.7 und neuer): lange Kante ≤ 2576 px, ≤ 4784 Tokens.
- Das Skript nutzt die Referenzimplementierung aus der Dokumentation; die Tests prüfen sie gegen
  die dort genannten Beispiele. Die ältere Faustregel „Breite × Höhe / 750“ ist überholt.
- Die Tokens der Tool-Antworten oben sind für die hochauflösende Stufe gerechnet, die das
  Testmodell in Claude Code nutzt. Bis 1024 px sind beide Stufen gleich.

**Ab wann lohnt sich ein Bild?**

- Ein Bild mit 1024 px kostet etwa **6× so viele Tokens** wie die JSON-Antwort von
  `detect_objects` (809 gegen 129) und rund **200× so viele Bytes**. Für Fragen, die die Erkennung
  beantwortet („ist jemand da?“, „wie viele …?“), ist das JSON klar die bessere Wahl. Das stützt
  die Tool-Beschreibungen, die `detect_objects` für solche Fragen empfehlen.
- Ein Bild lohnt sich nur, wenn die Frage über die 80 COCO-Klassen hinausgeht (Farbe, Zustand,
  Text, Anordnung) oder die Erkennung visuell geprüft werden soll.
- **512 px** kostet ein Viertel von 1024 px (209 Tokens) und reicht für einen groben Überblick.
- **1600 px** kostet 2,3× so viele Bytes wie 1024 px. Auf der Standard-Stufe verkleinert Claude das
  Bild ohnehin auf 1456×819; der Mehrwert ist dort klein. Auf der hochauflösenden Stufe sieht Claude
  die vollen 1600 px für 1914 Tokens.
- **Folgerung:** Das Limit von 1024 px bleibt sinnvoll. Wer feine Details braucht und ein
  hochauflösendes Modell nutzt, kann `max_image_width` anheben; bei Modellen der Standard-Stufe
  bringt alles über ~1456 px (16:9) nichts mehr.

## 3. Erkennungsqualität

Reproduzierbar mit
`uv run python scripts/evaluate_detection.py [--video <datei> --events <datei>] [--json <datei>]`.

### Material

- **37 eigene Fotos** (Handy, Originaldateien, 1440–4640 px), Alltagsmotive: Restaurant und Essen,
  Autos, Flugplatz und Flugzeugmuseum, Museumsvitrinen, Gegenstände in der Hand. 16 davon enthalten
  keine der gezählten Klassen und dienen als **Negativ-Set** (z. B. Armbanduhren, Säbel, Kerzen,
  Torbogen). Es sind keine Bilder einer fest montierten Kamera; `count_in_zone` wird damit nicht
  geprüft.
- **Ein Video** (34 s, 1920×1080) mit fest stehendem Handy vor dem Schreibtisch: Eine Person geht
  durch den Raum, nimmt eine Tasse vom Schreibtisch mit und stellt sie bei Sekunde 15–16 wieder
  vor drei Flaschen ab.
- Bilder und Video bleiben lokal in `recordings/eval/` (gitignored, Personen und Kennzeichen
  sichtbar). Ein Bild mit über 20 Modellflugzeugen ist ausgeschlossen, weil es sich nicht
  verlässlich zählen lässt.

### Referenz (Handzählung)

- Gezählt wurde die **Anzahl pro Bild und Klasse** für 23 COCO-Klassen, nicht die Position der
  Boxen. Verglichen wird deshalb nur die Anzahl; „übersehen“ und „zu viel“ sind Differenzen der
  Anzahlen, keine Zuordnung einzelner Boxen.
- Regeln: Ein Objekt zählt, wenn es am sichtbaren Teil erkennbar ist, auch angeschnitten oder
  verdeckt. Eine Hand oder ein anderer Körperteil zählt als `person`. Gläser ohne Stiel zählen als
  `cup`. Realistische Modelle (Flugzeugmodelle) zählen als ihre Klasse, stilisierte Figuren
  (Holzkrippe) nicht. Objekte ohne COCO-Klasse (Armbanduhr, Hubschrauber, Säbel) stehen nur als
  Notiz in der Referenz.
- COCO-Klassen ohne eigene Spalte (z. B. `vase`, `boat`, `microwave`) zählen nicht als Fehler,
  sondern werden als „andere Klassen“ getrennt ausgewiesen.
- **Ablauf:** Max hat zuerst ohne Kenntnis der Modellergebnisse gezählt. Danach hat Claude alle
  Bilder angesehen und Korrekturen vorgeschlagen; Max hat sie bestätigt
  (`recordings/eval/review_log.txt`). Zwei Werte hat nur Claude gezählt (7 Stühle, 5 Bücher).
- **Einfluss des Kontrolldurchgangs:** Er wurde ausgelöst durch die zusätzlichen Erkennungen des
  Modells und hat vor allem diese bestätigt (z. B. Wassergläser, Hände, weitere Autos). Er hat aber
  auch Objekte ergänzt, die das Modell übersieht (2 Einweiser, eine zweite Person, 5 Bücher,
  7 Stühle). Auf den 29 ursprünglich gezählten Bildern bei Schwelle 0,40:

  | | Objekte in der Referenz | übersehen | zu viel |
  |---|---:|---:|---:|
  | vor dem Kontrolldurchgang | 66 | 36 | 26 |
  | danach | 103 | 54 | 7 |

  Die erste Zählung war also unvollständig, vor allem bei vollen Bildern. Weil gezielt die
  zusätzlichen Erkennungen geprüft wurden, ist „zu viel“ eher zu niedrig angesetzt: Eine falsche
  Erkennung, die einem übersehenen echten Objekt gleicher Klasse gegenübersteht, fällt beim
  Vergleich der Anzahlen nicht auf.

### Ergebnis (`yolo26n.pt`, CPU, `imgsz` 640)

| Schwelle | Bilder exakt | Klassenzählungen exakt | mittl. abs. Fehler | Referenz | erkannt | übersehen | zu viel | andere Klassen |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,25 | 12/37 (32 %) | 24/68 (35 %) | 1,18 | 110 | 100 | 45 | 35 | 5 |
| **0,40** | **19/37 (51 %)** | **37/68 (54 %)** | **0,97** | 110 | 62 | 57 | 9 | 4 |
| 0,50 | 20/37 (54 %) | 34/68 (50 %) | 1,09 | 110 | 48 | 68 | 6 | 1 |
| 0,60 | 17/37 (46 %) | 29/68 (43 %) | 1,24 | 110 | 38 | 78 | 6 | 1 |

„Klassenzählungen exakt“ bezieht sich für alle Schwellen auf dieselben 68 Paare aus Bild und
Klasse, bei denen die Referenz oder die Erkennung bei 0,25 über 0 liegt. Paare, die überall 0 sind,
würden das Ergebnis schönen.

**Beobachtungen**

- **Die Standardschwelle 0,40 ist gut gewählt.** Sie hat den kleinsten mittleren Fehler und die
  meisten exakten Klassenzählungen. Darunter steigen die falschen Erkennungen stark (35 bei 0,25),
  darüber werden schnell mehr Objekte übersehen.
- **Das Modell übersieht viel, erfindet aber wenig.** Bei 0,40 werden etwa die Hälfte der
  Objekte der Referenz nicht gefunden (57 von 110), aber nur 9 zusätzlich gezählt. Für die
  Nutzung heißt das: Ein „ja, da ist X“ ist recht verlässlich, ein „kein X zu sehen“ deutlich
  weniger.
- **Vor allem kleine, entfernte und viele gleiche Objekte fehlen:** 15 Personen im Museum →
  5 erkannt; 6 Flugzeuge auf dem Flugplatz → 1; Besteck 9 → 3; Stühle im Hintergrund 7 → 1;
  angeschnittene Buchrücken 5 → 0. Typisch für das kleinste Modell (`n`) bei 640 px Eingangsgröße,
  auf die auch 4000-px-Fotos verkleinert werden.
- **Gut erkannt:** große, freistehende Objekte (Autos im Vordergrund, Weingläser, Wassergläser,
  Tastatur).
- **Objekte außerhalb der COCO-Klassen werden zur ähnlichsten Klasse:** Desktop-PC → `microwave`,
  Kerzen und Holzkrippe → `cake`, rohes Steak → `pizza`, Kochfeld → `oven`, Torbogen →
  `parking meter`, Radarkuppel → `kite`. Die meisten falschen Erkennungen auf den
  16 Negativ-Bildern sind solche Verwechslungen. Für ein LLM, das nur die Klassen sieht, ist das nicht vom echten Objekt zu
  unterscheiden; ein Bild (`include_image`) hilft hier.
- **Hände werden meist als `person` erkannt** (Goldbarren, Motor, Medaille), einmal doppelt
  (Camembert: 2 statt 1).

### Video

Ein Bild pro Sekunde bei Schwelle 0,40. Die Ereignisse hat Max notiert; Claude hat sie mit den
Einzelbildern abgeglichen und um die Tasse am Anfang ergänzt (Max hatte sie nicht notiert):

| Zeitraum | Ereignis | Erkannt |
|---|---|---|
| 0–3 s | Person direkt vor der Kamera (Hand, Beine); **Tasse steht vor den Flaschen** | `person` in jeder Sekunde, bei 0 s und 2 s doppelt; `bottle` keine, `cup` keine |
| 4 s | Person nimmt die Tasse mit | `person` |
| 5–14 s | Person geht zur Tür und zurück, die Flaschen sind frei | `person` in jeder Sekunde; `bottle` 3 in jeder Sekunde |
| 15–16 s | Tasse wird wieder vor den Flaschen abgestellt | `person`, bei 16 s doppelt |
| 17–32 s | Person geht zur Tür und zurück; Tasse steht vor den Flaschen | `person` in jeder Sekunde, bei 23 s doppelt; `bottle` keine, `cup` keine |
| 33 s | Kamera wird bewegt | `cup` 1 (aus anderem Blickwinkel) |

- **Personen werden zuverlässig erkannt**, über den ganzen Raum hinweg und auch, wenn nur Beine
  oder eine Hand zu sehen sind; gelegentlich doppelt.
- **Die Tasse wird aus dieser Perspektive nie erkannt**, nur in der letzten Sekunde aus einem
  anderen Winkel. Es ist ein grauer, strukturierter Becher, der von der Kamera aus vor den Flaschen
  steht.
- **Steht die Tasse vor den Flaschen, werden die Flaschen nicht mehr erkannt.** Das passiert zweimal,
  am Anfang (0–3 s) und nach dem Abstellen (ab 17 s); dazwischen werden alle drei sicher erkannt.
  Bei Sekunde 20 erreicht eine Flasche nur noch 0,33, die anderen werden auch bei 0,25 nicht
  erkannt; bei Sekunde 12 lagen alle drei bei 0,50–0,74. Verdeckung ist damit die deutlichste
  Schwäche in der Szene.
- Für Fragen wie „Was steht auf dem Schreibtisch?“ wäre die Antwort in 24 von 34 Sekunden falsch
  (keine Flaschen und keine Tasse erkannt).

### Nicht weiter untersucht

Aus M4 offen geblieben: eine falsche `mouse` (0,62) am rechten Bildrand der Handy-Kamera und
`tv` mit 0,50 auf dem Standbild gegenüber 0,70–0,76 live.

## 4. LLM-Aufgabenerfolg

### Aufbau

15 Fragen aus `docs/eval_questions.md` zu vier Standbild-Kameras (Zählen, Ja/Nein, Zone, Fragen,
die ein Bild brauchen, Fragen, bei denen die Erkennung falsch liegt, eine Frage über alle
Kameras). Jede Frage läuft in einer eigenen, frischen Claude-Code-Sitzung ohne Projektkontext:

```
claude -p "<Frage>" --mcp-config mcp.json --strict-mcp-config --tools "" \
  --allowedTools <die vier Kamera-Tools> --setting-sources local \
  --no-session-persistence --output-format stream-json --verbose
```

Aus einem leeren Arbeitsordner, ohne eingebaute Tools, ohne Nutzer- und Projekteinstellungen.
Modell: `claude-opus-5-5`. Aus dem Protokoll werden Tool-Aufrufe, Fehler, Antwort, Dauer und
Kosten gelesen. Ob eine Antwort stimmt, hat Claude in der entwerfenden Sitzung gegen die erwartete
Antwort bewertet; Grenzfälle sind unten genannt.

Verglichen wurden zwei Fassungen der Tool-Beschreibungen (Variante A), je 3 Durchgänge:

- **Vorher:** die Beschreibungen aus KONZEPT.md Abschnitt 5 bis M4.
- **Nachher:** `get_frame` mit Hinweis auf das Breitenlimit; `detect_objects` mit den Schwächen
  aus Abschnitt 3 und einer Faustregel, wann das Bild zu prüfen ist (Wortlaut in `DESIGN.md`).

### Ergebnis (Mittel pro Durchgang à 15 Fragen)

| | Vorher | Nachher |
|---|---:|---:|
| richtige Antworten | 15/15 (45/45) | 15/15 (45/45) |
| Tool-Aufrufe | 40,3 | 39,0 |
| Bilder (`get_frame` oder `include_image=true`) | 15,7 | 17,7 |
| Fragen mit mindestens einem Bild | 12,3 (12, 13, 12) | 13,7 (13, 14, 14) |
| `get_frame` mit `max_width` > 1024 | 2 | 1 |
| Tool-Fehler | 0 | 0 |
| Kosten | 0,60 $ | 0,63 $ |
| Dauer (alle 15 Fragen nacheinander) | 226 s | 227 s |

Grenzfall: Bei Frage 7 (3 Gabeln) antwortete ein Nachher-Durchgang „2 sicher, wahrscheinlich 3“;
gewertet als richtig, weil Claude die Unsicherheit benennt und bei 3 landet.

### Beobachtungen

- **Die Fallen erkennt Claude in beiden Fassungen.** Bei Gabeln (Frage 7), Stühlen (12) und der
  „Uhr“ (13) liefert die Erkennung falsche Werte (1 Gabel, 1 Stuhl, `clock`); Claude prüft in
  jedem Durchgang das Bild und korrigiert. Die Genauigkeit hängt hier also nicht an der
  Beschreibung, sondern daran, dass Claude die Ergebnisse von sich aus anzweifelt.
- **Fragen, die ein Bild brauchen, gehen direkt an `get_frame`** (4, 5, 9, 11). Reine Zählfragen
  mit klarem Ergebnis bleiben ohne Bild (2, vorher auch 3).
- **Die neue Beschreibung macht vorsichtiger, nicht sparsamer.** Die Faustregel „Konfidenz unter
  etwa 0,6“ greift fast immer, weil viele Erkennungen zwischen 0,47 und 0,6 liegen. Frage 3
  („Sitzt jemand am Schreibtisch?“) bekommt nachher in jedem Durchgang ein Bild, vorher in keinem;
  Frage 15 alle vier statt im Mittel 3,3 Standbilder. Gespart wurde nur bei 9 und 13, wo Claude
  das Bild gleich mit `detect_objects` bzw. nur einmal holt statt mit einem zweiten Aufruf.
- **Der Hinweis auf das Breitenlimit halbiert die vergeblichen Anfragen** mit 2048 bzw. 2400 px
  (geliefert werden ohnehin 1024 px), verhindert sie aber nicht ganz.
- **Klassennamen:** Claude übersetzt deutsche Begriffe selbst in COCO-Namen (`mouse`, `fork`,
  `wine glass`, `chair`); es gab keinen Aufruf mit unbekannter Klasse. Die in Abschnitt 5
  vermerkte Sorge, dass die gekürzte Klassenliste in der Fehlermeldung stört, trat nicht auf.
- Die zweite Sorge aus Abschnitt 5 (offline gegenüber „startet gerade“) prüfen diese Fragen nicht,
  weil sie nur Standbilder nutzen.

### Entscheidung

Übernommen wird nur der Satz zum Breitenlimit bei `get_frame`; `detect_objects` behält die alte
Beschreibung. Die neue Fassung kostet etwa 5 % mehr und bringt bei gleicher Genauigkeit nichts.
Die Mischfassung selbst ist nicht gemessen; erwartet wird das Verhalten von „Vorher“ mit weniger
Anfragen über 1024 px. Begründung und verworfene Varianten in `DESIGN.md`.

Aussagekraft: 3 Durchgänge und 15 Fragen sind wenig, und alle Fragen sind beantwortbar. Die
Richtung (mehr Bilder nachher) ist aber in jedem einzelnen Durchgang gleich.

## 5. Robustheit

### Fehlerfälle (live über Claude Code geprüft, 2026-11-06)

Jeder Fall liefert eine Fehlerantwort (`isError`) mit Ursache und Hinweis, ohne Stacktrace. Der
Server läuft danach normal weiter.

| Fall | Antwort des Servers | Abgedeckt durch Test |
|---|---|---|
| Kamera aus (Handy-App gestoppt) | `get_camera_info`: `status: offline`. `detect_objects`: „Camera 'phone' is offline, no frame received yet. Check get_camera_info.“ | `test_offline_camera_gives_clear_error`, `test_camera_without_frame_is_offline` |
| Unbekannte Kamera | „Unknown camera 'garage'. Available cameras: desk_still, phone, sample.“ | `test_camera_info_unknown_camera_lists_known_ones`, `test_unknown_camera_error_lists_known_cameras` |
| Unbekannte Zone | „Unknown zone 'tuer' for camera 'desk_still'. Available zones: mauspad, monitor, tastatur.“ | `test_count_in_zone_unknown_zone_lists_zones_without_inference` |
| Kamera ohne Zonen | „Camera 'sample' has no zones. Use get_camera_info to see the zones of each camera.“ | `test_camera_without_zones_says_so` |
| Unbekannte Klasse | „Unknown classes: maus, unicorn. Valid classes: person, bicycle, … (80 in total).“ | `test_detect_objects_unknown_class_lists_valid_classes` |
| Ungültige Konfidenz | „min_confidence must be between 0 and 1, got 1.5.“ | `test_detect_objects_invalid_confidence_gives_clear_error` |
| Unerwarteter Fehler im Server | Allgemeine Fehlermeldung ohne interne Details | `test_unexpected_failure_does_not_reach_the_client` |

### Stream-Abbruch und Reconnect

Durch Tests abgedeckt: `test_reconnects_after_lost_connection_and_closes_source`,
`test_stale_frame_makes_camera_offline`, `test_reconnects_after_failed_opens`,
`test_stream_uses_ffmpeg_backend_and_reports_lost_connection`.

**Live-Test (2026-11-06).** Die Handy-App „IP Webcam“ wurde während des Betriebs gestoppt und nach
etwa 20 s wieder gestartet, einmal über HTTP-MJPEG (`/video`) und einmal über RTSP mit H.264
(`/h264_pcm.sdp`). Ein MCP-Client fragte dabei alle 0,25 s `get_camera_info` ab und rief bei jedem
Statuswechsel `detect_objects` auf.

| Messgröße | HTTP-MJPEG | RTSP |
|---|---|---|
| Abbruch erkannt („lost connection to network stream“) | sofort | sofort |
| Wechsel auf `offline` nach letztem Frame | 5,1 s | 5,2 s |
| Meldung von `detect_objects` danach | „Camera 'phone' is offline, last frame 5 s ago. Check get_camera_info.“ | gleich |
| Wieder `online` nach dem App-Start | ≤ ~5 s (Startzeit nicht genau bekannt) | ≥ 12 s (Nachricht „an“ 02:50:03, Reconnect 02:50:14,9) |
| Server ansprechbar während des Ausfalls | ja, alle Aufrufe < 500 ms | ja, alle Aufrufe < 500 ms |

**Befund RTSP:** Während die App aus war, blieb ein Verbindungsversuch 30 s lang hängen, bis das
Standard-Timeout von OpenCV/FFmpeg griff („Stream timeout triggered after 30005 ms“). Erst danach
und nach `reconnect_interval_s` folgte der nächste Versuch. Im ungünstigsten Fall dauert der
Reconnect bei RTSP also rund 30 s + 5 s statt etwa 5 s. Die Tools bleiben davon unberührt, weil
der Versuch im Kamera-Thread läuft; die Kamera bleibt in dieser Zeit nur länger `offline`.

**Fix und Wiederholung.** `sources.py` setzt seitdem für Netzwerk-Streams 5 s als Timeout für
Öffnen und Lesen (`CAP_PROP_OPEN_TIMEOUT_MSEC`, `CAP_PROP_READ_TIMEOUT_MSEC`). Derselbe Test mit RTSP,
App diesmal etwa 60 s aus:

| Messgröße | RTSP vorher | RTSP mit Timeout |
|---|---|---|
| Wechsel auf `offline` nach letztem Frame | 5,2 s | 5,1 s |
| Dauer eines vergeblichen Verbindungsversuchs | 30,0 s | 5,0–5,1 s |
| Abstand der Versuche (Timeout + `reconnect_interval_s`) | ~35 s | ~10 s |
| Wieder `online` nach dem App-Start | ≥ 12 s | vor der Nachricht „an“ (Reconnect 02:54:45,3, Nachricht 02:54:48) |
| Server ansprechbar während des Ausfalls | ja | ja, alle Aufrufe < 500 ms |

Im ungünstigsten Fall dauert der Reconnect bei RTSP jetzt etwa 10 s plus Verbindungsaufbau.

Weitere Beobachtungen:

- Direkt nach einem Reconnect meldet `fps_measured` kurz zu wenig (0,7–1,1), weil das Messfenster
  noch fast leer ist. Beim ersten RTSP-Aufbau kurz zu viel (96 fps), vermutlich durch einen Schwall
  gepufferter Frames. Beides gleicht sich nach wenigen Sekunden aus.
- Gescheiterte Verbindungsversuche erscheinen nur einmal im Log, nicht alle 5 s.

## Quellen

- Anthropic, Vision: <https://platform.claude.com/docs/en/build-with-claude/vision>
  (Abschnitt „Resolution and token cost“)
- Anthropic, Coordinates and bounding boxes:
  <https://platform.claude.com/docs/en/build-with-claude/vision-coordinates>
  (Referenzimplementierung der Verkleinerung)
