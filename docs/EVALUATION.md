# Evaluation

Gliederung nach KONZEPT.md, Abschnitt 10. Messungen sind reproduzierbar mit
`uv run python scripts/benchmark.py --camera <id> [--runs 50] [--json <datei>]`.

| # | Teil | Stand |
|---|---|---|
| 1 | Latenz pro Tool | CPU gemessen; GPU folgt in M6 |
| 2 | Antwortgröße | gemessen |
| 3 | Erkennungsqualität | offen, wartet auf Testbilder mit Handzählung |
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

Offen. Braucht 30–50 eigene Testbilder mit von Hand gezählten Objekten (KONZEPT.md, Abschnitt 9).

Erste Hinweise aus M4, die hier geprüft werden sollen:

- Eine falsche „mouse“ mit Konfidenz 0,62 am rechten Bildrand der Handy-Kamera.
- „tv“ erreicht auf dem Standbild 0,50, live 0,70–0,76; vermutlich wegen der erneuten
  JPEG-Kompression des Standbilds.

## 4. LLM-Aufgabenerfolg

Offen. Braucht 10–15 Evaluationsfragen (`docs/eval_questions.md`).

Beobachtungen aus dem Robustheitstest, die hier geprüft werden sollen:

- Bei unbekannten Klassen listet die Fehlermeldung nur die ersten 20 der 80 Klassen. Fragt
  jemand auf Deutsch nach „Maus“, steht `mouse` nicht in der Liste. Ob das Modell die richtige
  Klasse trotzdem findet, zeigt die Evaluation.
- „Kamera aus“ meldet „no frame received yet“, ohne die Ursache (Verbindung abgelehnt). Das Modell
  kann „startet gerade“ und „nicht erreichbar“ nicht unterscheiden.

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
`sources.py` setzt bisher kein eigenes Timeout für das Öffnen.

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
