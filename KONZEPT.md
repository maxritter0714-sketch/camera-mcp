# camera-mcp – Umsetzungskonzept

Stand: 05.11.2026 · Autor: Max Ritter · Zielgruppe: Claude Code (Umsetzung) und ich selbst

## 1. Ziel in einem Satz

Ein MCP-Server, der einem KI-Client (Claude Desktop / Claude Code) eine oder mehrere Kameras als **Werkzeuge** zur Verfügung stellt: Bild holen, Objekte erkennen, Objekte in Zonen zählen – sauber designt, gemessen und dokumentiert, als Vorbereitung auf die Bachelorarbeit „Evaluierung und Weiterentwicklung eines MCP-Servers für industrielle Kameras“.

Der Code ist dabei nur die halbe Miete. Mindestens genauso wichtig sind:

- **die Design-Entscheidungen** (warum welches Tool so geschnitten ist) → `docs/DESIGN.md`
- **die Evaluation** (Latenz, Antwortgröße, Erkennungsqualität, Erfolgsquote des LLM) → `docs/EVALUATION.md`

## 2. Umfang

### Im Umfang (MVP)

- Kameraquellen: Webcam, RTSP/HTTP-Stream (z.B. Handy als IP-Kamera), Datei (Bild/Video, für Entwicklung und Tests)
- Objekterkennung mit vortrainiertem YOLO-Modell (Standard: `yolo26n.pt`, austauschbar)
- 4 rein lesende MCP-Tools (Abschnitt 5)
- Zonen als Polygone in normierten Koordinaten
- Konfiguration per YAML, Geheimnisse per Umgebungsvariablen
- Tests ohne echte Kamera (Datei-Quelle)
- Lauffähig auf Windows/macOS/Linux, mit CPU oder NVIDIA-GPU
- Evaluation und Dokumentation

### Ausdrücklich nicht im Umfang (MVP)

- Eigenes Training / Fine-Tuning des Erkennungsmodells
- Aufzeichnung oder dauerhafte Speicherung von Bildern/Videos
- Schreibende Tools (Kameraeinstellungen ändern o.ä.) – erst als Stretch Goal
- Web-Oberfläche, Benutzerverwaltung, öffentliche Erreichbarkeit
- Gesichtserkennung oder Identifikation von Personen

## 3. Architektur

```
Claude Desktop / Claude Code  (MCP-Client)
        │  stdio (lokal)  bzw.  stdio über SSH/Tailscale (Server)
        ▼
┌──────────────────────── camera-mcp ─────────────────────────┐
│ server.py   MCPServer, Lifespan, 4 Tools                    │
│     │                                                        │
│     ├── CameraManager ── FrameSource (webcam | rtsp | file)  │
│     │     Hintergrund-Thread je Kamera, hält nur das         │
│     │     neueste Frame, Reconnect, Status                   │
│     │                                                        │
│     ├── Detector (Protokoll) ── UltralyticsDetector          │
│     │     Gerät automatisch: cuda wenn verfügbar, sonst cpu  │
│     │                                                        │
│     ├── Zones      Polygone (0..1), Punkt-in-Polygon         │
│     └── Imaging    Skalieren, JPEG, Annotation (Boxen)       │
└──────────────────────────────────────────────────────────────┘
```

### Wichtige Architekturentscheidungen

| Entscheidung | Begründung |
|---|---|
| **Detector hinter einem Protokoll/Interface** | Modell und Bibliothek austauschbar (Lizenz, Performance, später ONNX/TensorRT oder Industriekamera-SDK). |
| **FrameSource hinter einem Protokoll** | Webcam, RTSP und Datei sind austauschbar; Tests laufen ohne Kamera; eine Industriekamera wäre nur eine weitere Implementierung. |
| **Capture im Hintergrund-Thread, nur neuestes Frame** | OpenCV puffert sonst Frames → Tools würden veraltete Bilder liefern. Tools blockieren nie auf die Kamera. |
| **Normierte Koordinaten (0..1)** | Zonen und Boxen sind unabhängig von der Auflösung; LLM-freundlich. |
| **Strukturierte Daten zuerst, Bilder nur auf Wunsch und verkleinert** | Bilder kosten viel Kontext und Zeit; die meisten Fragen sind mit JSON beantwortbar. |
| **Modell wird einmal im Lifespan geladen** | Kein Neuladen pro Tool-Aufruf. |
| **stdio als Transport** | Kein offener Port, keine eigene Auth; auf dem Server läuft stdio durch SSH (Authentifizierung über SSH-Schlüssel). |

## 4. Technologie

| Bereich | Wahl | Hinweis |
|---|---|---|
| Sprache | Python 3.12 | |
| Paketverwaltung | uv (mit `uv.lock`) | reproduzierbar auf jedem Rechner |
| MCP | offizielles Python-SDK, Paket `mcp` 2.x (MCPServer, früher FastMCP) | `from mcp.server.mcpserver import MCPServer, Image` |
| Kamera | `opencv-python` | Webcam, RTSP, Video-Dateien |
| Erkennung | `ultralytics`, Modell `yolo26n.pt` | Modellname konfigurierbar |
| Konfiguration | `pydantic` + `PyYAML` | Validierung, klare Fehlermeldungen |
| Bilder | OpenCV / Pillow | JPEG-Encoding, Skalieren, Annotation |
| Tests | `pytest` | |
| Linting | `ruff` | |

**Lizenzhinweis:** `ultralytics` steht unter AGPL-3.0. Für einen privaten Prototyp unkritisch. Wird das Repo öffentlich, entweder das Projekt selbst unter AGPL-3.0 stellen oder einen anderen Detector einsetzen. Genau dafür gibt es das Detector-Interface. Diese Abwägung gehört in `docs/DESIGN.md`.

## 5. Tool-Design (Kernstück)

### Grundprinzipien

1. **Alle MVP-Tools sind rein lesend** und ohne Nebenwirkungen.
2. **Strukturierte JSON-Antworten**, Bilder nur, wenn ausdrücklich angefordert.
3. **Bilder immer verkleinert** (Standard max. 1024 px Breite, JPEG-Qualität 80).
4. **Koordinaten normiert** (0..1, Format `[x1, y1, x2, y2]`).
5. **Jede Antwort enthält Zeitstempel und Frame-Alter**, damit das Modell erkennt, ob ein Bild aktuell ist.
6. **Fehler sind für das Modell verständlich**: klare Fehlermeldung mit Ursache und Hinweis (z.B. „Kamera `phone` ist offline, letztes Bild vor 42 s. Prüfe `get_camera_info`.“). Nie Stacktraces.
7. **Tool-Beschreibungen sind Teil des Designs.** Sie sagen dem Modell, *wann* es welches Tool nutzen soll und wann nicht.

### Die vier Tools

#### `get_camera_info(camera_id: str | None = None)`

Beschreibung für das Modell:
> Listet die verfügbaren Kameras mit Status (online/offline), Auflösung, Alter des letzten Bildes und den definierten Zonen. Ohne `camera_id` werden alle Kameras geliefert. Nutze dieses Tool zuerst, wenn du nicht weißt, welche Kameras oder Zonen es gibt.

Antwort (Beispiel):
```json
{
  "cameras": [
    {
      "id": "phone",
      "name": "Handy (IP-Kamera)",
      "source_type": "rtsp",
      "status": "online",
      "resolution": [1280, 720],
      "fps_measured": 14.8,
      "last_frame_age_s": 0.07,
      "zones": ["tuer", "schreibtisch"]
    }
  ]
}
```

#### `get_frame(camera_id: str, max_width: int = 1024, annotate: bool = False)`

Beschreibung:
> Liefert das aktuelle Bild einer Kamera. Nur nutzen, wenn du die Szene wirklich sehen musst (z.B. um etwas zu beschreiben, das die Objekterkennung nicht abdeckt). Für Fragen wie „ist jemand da?“ oder „wie viele …?“ ist `detect_objects` schneller und günstiger. Mit `annotate=true` werden erkannte Objekte eingezeichnet.

Antwort: ein Bild (MCP-Image, JPEG) plus kurzer Text mit Zeitstempel, Originalauflösung, ausgelieferter Auflösung, Frame-Alter. `max_width` ist auf einen Höchstwert aus der Konfiguration begrenzt.

#### `detect_objects(camera_id: str, classes: list[str] | None = None, min_confidence: float | None = None, include_image: bool = False)`

Beschreibung:
> Erkennt Objekte im aktuellen Bild (COCO-Klassen wie person, car, bicycle, dog, cell phone …). Liefert strukturierte Ergebnisse mit Klasse, Konfidenz, Position und Zone. Mit `classes` auf bestimmte Klassen filtern. Ein Bild nur mit `include_image=true` anfordern, wenn die Ergebnisse visuell geprüft werden sollen.

Antwort (Beispiel):
```json
{
  "camera_id": "phone",
  "timestamp": "2026-11-05T14:35:12.418+01:00",
  "frame_age_s": 0.05,
  "frame_size": [1280, 720],
  "model": "yolo26n.pt",
  "device": "cuda",
  "inference_ms": 11.4,
  "counts": {"person": 1, "cup": 2},
  "detections": [
    {"label": "person", "confidence": 0.91, "bbox": [0.12, 0.08, 0.41, 0.97], "zones": ["tuer"]},
    {"label": "cup", "confidence": 0.63, "bbox": [0.70, 0.55, 0.76, 0.66], "zones": ["schreibtisch"]}
  ]
}
```

Unbekannte Klassennamen in `classes` → klare Fehlermeldung mit Liste der gültigen Klassen (gekürzt).

#### `count_in_zone(camera_id: str, zone: str, classes: list[str] | None = None, min_confidence: float | None = None)`

Beschreibung:
> Zählt erkannte Objekte innerhalb einer benannten Zone (z.B. „tuer“). Ein Objekt zählt zur Zone, wenn der Mittelpunkt der Unterkante seiner Box (Fußpunkt) in der Zone liegt. Verfügbare Zonen liefert `get_camera_info`.

Antwort: Zone, Zeitstempel, `counts` pro Klasse, Liste der betroffenen Detektionen (wie oben).

**Begründung Fußpunkt statt Box-Mittelpunkt:** Für „steht jemand in der Zone“ ist der Bodenkontakt aussagekräftiger. Diese Entscheidung in `DESIGN.md` festhalten.

### Optional: MCP-Annotationen

Falls das SDK es unterstützt, alle MVP-Tools als `readOnlyHint=true` und `idempotentHint=true` kennzeichnen.

## 6. Konfiguration

`config.example.yaml` (eingecheckt), `config.yaml` (lokal, nicht eingecheckt):

```yaml
detector:
  model: yolo26n.pt          # austauschbar, z.B. yolo26s.pt
  device: auto               # auto | cuda | cpu
  default_confidence: 0.4
  imgsz: 640

cameras:
  - id: sample
    name: Testvideo
    source: {type: file, path: tests/fixtures/sample.mp4, loop: true}
  - id: desk
    name: Schreibtisch-Webcam
    source: {type: webcam, index: 0}
  - id: phone
    name: Handy (IP-Kamera)
    source: {type: rtsp, url: "${PHONE_STREAM_URL}"}   # aus Umgebungsvariable / .env

zones:
  phone:
    - name: tuer
      polygon: [[0.05, 0.10], [0.35, 0.10], [0.35, 0.98], [0.05, 0.98]]

limits:
  max_image_width: 1024
  jpeg_quality: 80
  max_frame_age_s: 5          # älter = Kamera gilt als offline
  reconnect_interval_s: 5
```

- Pfad zur Konfiguration über `CAMERA_MCP_CONFIG` (Standard: `./config.yaml`).
- `${VAR}`-Platzhalter werden aus Umgebungsvariablen bzw. `.env` ersetzt (Stream-URLs können Zugangsdaten enthalten).
- Ungültige Konfiguration → Server startet nicht, klare Fehlermeldung auf stderr.

## 7. Projektstruktur

```
camera-mcp/
├── pyproject.toml
├── uv.lock
├── config.example.yaml
├── .env.example
├── .gitignore              # config.yaml, .env, *.pt, recordings/, .venv/
├── CLAUDE.md
├── KONZEPT.md
├── README.md
├── src/camera_mcp/
│   ├── __init__.py
│   ├── config.py           # Pydantic-Modelle, YAML laden, ${VAR} ersetzen
│   ├── sources.py          # FrameSource-Protokoll + Webcam/RTSP/File
│   ├── camera_manager.py   # Hintergrund-Capture, Reconnect, Status
│   ├── detector.py         # Detector-Protokoll + UltralyticsDetector
│   ├── zones.py            # Polygone, Fußpunkt-in-Zone
│   ├── imaging.py          # Skalieren, JPEG, Annotation
│   ├── schemas.py          # Antwortmodelle (Pydantic)
│   ├── logging_setup.py    # Logging NUR auf stderr / Datei
│   └── server.py           # MCPServer, Lifespan, Tools, main()
├── scripts/
│   ├── grab_frame.py       # Bild von einer Kamera speichern (ohne MCP)
│   ├── detect_once.py      # Erkennung auf einem Bild/Frame (ohne MCP)
│   └── benchmark.py        # Latenz- und Größenmessung für die Evaluation
├── tests/
│   ├── fixtures/           # eigene Testbilder + kurzes Testvideo
│   ├── test_config.py
│   ├── test_zones.py
│   ├── test_imaging.py
│   ├── test_sources.py
│   ├── test_tools.py       # Tool-Logik mit Fake-Detector
│   └── test_mcp_e2e.py     # echter MCP-Client über stdio gegen den Server
└── docs/
    ├── DESIGN.md           # Entscheidungslog
    ├── EVALUATION.md       # Messungen und Ergebnisse
    └── eval_questions.md   # Testfragen für die LLM-Evaluation
```

## 8. Kritische Stolperfallen

1. **Niemals auf stdout schreiben.** Bei stdio-Transport ist stdout der Protokollkanal. Jede `print()`-Ausgabe zerstört die Verbindung. Logging nur auf stderr oder in eine Datei. Ultralytics mit `verbose=False` aufrufen und dessen Logger auf stderr umleiten.
2. **OpenCV-Puffer:** Ohne Hintergrund-Thread liefert `cap.read()` alte Frames.
3. **RTSP-Abbrüche:** Reconnect mit Wartezeit, Status sauber auf `offline` setzen, nie den Server abstürzen lassen.
4. **Modell-Download:** `yolo26n.pt` wird beim ersten Start geladen. Für Tests einen Fake-Detector verwenden, damit Tests offline laufen.
5. **Bildgröße:** Ohne Verkleinerung sprengen Bilder schnell den Kontext.
6. **Threading:** Detector-Aufrufe serialisieren (ein Lock), da GPU-Modelle nicht threadsicher sein müssen.
7. **Plattform:** `pathlib` statt fest codierter Pfade, keine Windows- oder Linux-spezifischen Befehle im Code.

## 9. Meilensteine

Jeder Meilenstein endet mit grünen Tests, `ruff check` ohne Fehler und einem kurzen Eintrag in `docs/DESIGN.md`.

| # | Inhalt | Abnahmekriterium |
|---|---|---|
| **M0** | Projektgerüst: uv, ruff, pytest, Konfiguration laden und validieren, Logging auf stderr | `uv run pytest` grün; ungültige Config gibt klare Meldung |
| **M1** | FrameSources (file, webcam, rtsp) + CameraManager mit Hintergrund-Thread, Reconnect, Status | `scripts/grab_frame.py --camera sample` speichert ein Bild; Tests mit Datei-Quelle |
| **M2** | Detector-Interface + UltralyticsDetector (auto cuda/cpu) + Imaging (Skalieren, JPEG, Annotation) | `scripts/detect_once.py` gibt JSON wie in Abschnitt 5 aus; Fake-Detector für Tests |
| **M3** | MCP-Server mit `get_camera_info`, `get_frame`, `detect_objects` | Test im MCP Inspector (`uv run mcp dev src/camera_mcp/server.py`); End-to-End-Test mit SDK-Client; Claude Desktop kann die Tools nutzen |
| **M4** | Zonen + `count_in_zone` | Unit-Tests für Punkt-in-Polygon inkl. Randfälle; Tool funktioniert in Claude Desktop |
| **M5** | Evaluation (Abschnitt 10) + Dokumentation | `docs/EVALUATION.md` mit Messwerten; README vollständig |
| **M6** | Umzug auf den Linux-Server mit RTX 3060 (Abschnitt 11) | Claude Desktop auf dem Laptop nutzt die Tools über SSH/Tailscale; Benchmark CPU vs. GPU dokumentiert |

### Wann wird Material von mir gebraucht?

| Meilenstein | Was ich liefern muss | Wozu |
|---|---|---|
| M0–M1 | nichts | Tests laufen mit synthetischen Fixtures und Fake-Detector |
| M2 | 2–3 beliebige eigene Fotos mit Personen/Alltagsgegenständen, alternativ Webcam oder Handy-Stream | Plausibilitätscheck, dass das echte Modell etwas erkennt |
| M3–M4 | Handy als IP-Kamera eingerichtet (Stream-URL in `.env`) | Test mit Claude Desktop auf einer echten Szene; Zonen festlegen |
| M5 | 30–50 eigene Testbilder (mit Hand gezählten Objekten), ein kurzes Testvideo, 10–15 Evaluationsfragen | Erkennungsqualität und LLM-Aufgabenerfolg messen |

Claude Code fordert das Material zum jeweiligen Zeitpunkt an (siehe CLAUDE.md).

### Stretch Goals (nur wenn MVP fertig und Zeit übrig)

- **S1 – Ereignisse:** Hintergrund-Erkennung alle N Sekunden, Ereignisse (Objekt erscheint/verschwindet in Zone) in SQLite; Tool `get_events(camera_id, since, classes)`. Ermöglicht Fragen wie „wie oft war heute jemand an der Tür?“.
- **S2 – Schreibendes Tool mit Absicherung:** z.B. `set_detection_threshold` oder Zonen-Änderung, als `destructiveHint`/nicht-lesend markiert, mit Begründung, wie man schreibende Kamera-Tools absichert.
- **S3 – Streamable HTTP als zweiter Transport**, nur im Tailnet gebunden, mit Token-Authentifizierung. Vergleich stdio vs. HTTP in `DESIGN.md`.
- **S4 – Zweites Modell** (z.B. größeres YOLO oder ONNX-Export) hinter dem Detector-Interface; Vergleich.

## 10. Evaluation (das, was im Gespräch zählt)

Ergebnisse in `docs/EVALUATION.md`, Messungen reproduzierbar per `scripts/benchmark.py`.

1. **Latenz pro Tool** (Median und 95. Perzentil über 50 Aufrufe): `get_camera_info`, `get_frame`, `detect_objects` (mit/ohne Bild), `count_in_zone`. Getrennt nach CPU (Haupt-PC) und GPU (RTX 3060).
2. **Antwortgröße**: Bytes bzw. grobe Token-Schätzung für JSON-Antwort vs. Bild in verschiedenen Breiten (512 / 1024 / 1600 px). Ergebnis: Ab wann lohnt sich ein Bild?
3. **Erkennungsqualität**: 30–50 eigene Testbilder, Personen/Objekte von Hand gezählt; Zählfehler von `detect_objects` und `count_in_zone` bei verschiedenen Konfidenzschwellen.
4. **LLM-Aufgabenerfolg**: 10–15 natürlichsprachliche Fragen in `docs/eval_questions.md` (z.B. „Ist gerade jemand an der Tür?“, „Wie viele Tassen stehen auf dem Schreibtisch?“, „Was liegt links neben dem Laptop?“). Für jede Frage festhalten: welche Tools Claude aufgerufen hat, ob die Antwort stimmt, Anzahl der Aufrufe. Danach Tool-Beschreibungen verbessern und erneut messen (vorher/nachher).
5. **Robustheit**: Verhalten bei Kamera aus, RTSP-Abbruch, unbekannter Kamera/Zone/Klasse.

## 11. Entwicklung jetzt, Umzug auf den Server später

- **Entwicklung** auf dem Haupt-PC, mit Datei-Quelle und Handy als IP-Kamera (dieselbe Quellenart wie später auf dem Server). Läuft mit CPU, falls keine passende GPU vorhanden ist.
- **Umzug** (nach Server-Setup Phasen 0–6):
  1. `git clone` auf dem Server, `uv sync`
  2. `config.yaml` und `.env` anlegen (Stream-URL des Handys/der Kamera)
  3. `uv run pytest` und `uv run python scripts/detect_once.py --camera phone` → `device: cuda` prüfen
  4. Claude Desktop auf dem Laptop konfigurieren:
     ```json
     {
       "mcpServers": {
         "camera": {
           "command": "ssh",
           "args": ["<ssh-host>", "cd ~/camera-mcp && ~/.local/bin/uv run camera-mcp"]
         }
       }
     }
     ```
  5. Benchmark auf dem Server laufen lassen, CPU-vs-GPU-Vergleich in `EVALUATION.md` ergänzen
- Hinweis: Bei stdio über SSH startet jede Claude-Sitzung einen neuen Serverprozess (Modell und Kamera werden neu geladen, einige Sekunden). Für den Prototyp akzeptabel; Alternative ist S3 (HTTP-Transport als Dienst).

## 12. Sicherheit und Datenschutz

- Nur lesende Tools im MVP, kein Shell-Zugriff, keine Dateipfade aus Tool-Parametern (Datei-Quellen nur aus der Konfiguration).
- Keine Speicherung von Bildern; Frames nur im Arbeitsspeicher.
- Zugangsdaten (Stream-URLs) nur in `.env`, nie im Repo.
- Kameras nur auf eigene Räume richten, keine öffentlichen Bereiche oder Nachbargrundstücke.
- Kein öffentlicher Zugriff; Fernzugriff ausschließlich über SSH im Tailnet.

## 13. Definition of Done (MVP)

- [ ] M0–M5 abgeschlossen, alle Tests grün, `ruff check` sauber
- [ ] Claude Desktop beantwortet die Fragen aus `eval_questions.md` über die Tools
- [ ] `docs/DESIGN.md` enthält alle wesentlichen Entscheidungen mit Begründung
- [ ] `docs/EVALUATION.md` enthält Latenz, Antwortgrößen, Erkennungsqualität, LLM-Erfolgsquote
- [ ] README: Zweck, Installation, Konfiguration, Anbindung an Claude Desktop, Screenshots/Beispieldialog
