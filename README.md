# camera-mcp

MCP-Server, der Kameras als Werkzeuge für KI-Clients wie Claude bereitstellt. Das Modell kann damit
fragen, welche Kameras es gibt, ein aktuelles Bild holen, Objekte erkennen lassen und Objekte in
benannten Bildbereichen (Zonen) zählen. Die Objekterkennung läuft lokal mit YOLO
(`yolo26n.pt`, 80 COCO-Klassen), auf der CPU oder mit CUDA.

Die Idee dahinter: Für Fragen wie „Ist jemand da?“ oder „Wie viele Autos stehen auf dem
Parkplatz?“ reicht eine kleine JSON-Antwort der Erkennung, das Bild selbst muss nur bei Bedarf zum
Modell. Spezifikation: [KONZEPT.md](KONZEPT.md), Entscheidungen: [docs/DESIGN.md](docs/DESIGN.md),
Messwerte: [docs/EVALUATION.md](docs/EVALUATION.md).

## Tools

Alle Tools sind rein lesend. Koordinaten sind auf 0..1 normiert, jede Antwort enthält Zeitstempel
und Alter des Bildes.

| Tool | Zweck |
|---|---|
| `get_camera_info(camera_id?)` | Kameras mit Status (online/offline), Auflösung, Bildalter und Zonen |
| `get_frame(camera_id, max_width=1024, annotate=false)` | aktuelles Bild als JPEG, optional mit eingezeichneten Erkennungen |
| `detect_objects(camera_id, classes?, min_confidence?, include_image=false)` | erkannte Objekte mit Klasse, Konfidenz, Box und Zone |
| `count_in_zone(camera_id, zone, classes?, min_confidence?)` | Anzahl der Objekte, deren Fußpunkt in einer Zone liegt |

## Voraussetzungen

- Python 3.12 und [uv](https://docs.astral.sh/uv/)
- Windows oder Linux; eine NVIDIA-GPU ist optional (`device: auto` nimmt CUDA, wenn vorhanden)
- Kameras: Videodatei, Webcam oder Netzwerk-Stream (RTSP oder HTTP, z. B. eine IP-Kamera-App
  auf dem Handy)
- Für den MCP Inspector zusätzlich Node.js

## Installation

```
uv sync
uv run pytest
```

Die Tests laufen offline, ohne Kamera und ohne Modell.

## Konfiguration

`config.example.yaml` nach `config.yaml` kopieren und anpassen. Gelesen wird `./config.yaml` oder
der Pfad aus der Umgebungsvariable `CAMERA_MCP_CONFIG`.

```yaml
detector:
  model: yolo26n.pt          # wird beim ersten Start heruntergeladen; austauschbar, z.B. yolo26s.pt
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
    source: {type: rtsp, url: "${PHONE_STREAM_URL}"}   # rtsp:// oder http://

zones:
  phone:
    - name: tuer
      polygon: [[0.05, 0.10], [0.35, 0.10], [0.35, 0.98], [0.05, 0.98]]

limits:
  max_image_width: 1024      # Obergrenze für ausgelieferte Bilder
  jpeg_quality: 80
  max_frame_age_s: 5         # älter = Kamera gilt als offline
  reconnect_interval_s: 5
```

- **Stream-URLs** enthalten oft Zugangsdaten. Sie gehören in eine `.env` neben der `config.yaml`
  (Vorlage: `.env.example`) und werden über `${VAR}` eingesetzt. `config.yaml` und `.env` sind
  nicht eingecheckt.
- **Kameras, die du nicht nutzt, entfernen.** Eine fehlende `${VAR}` verhindert den Start.
- **Zonen** sind Polygone in normierten Koordinaten (0..1, Ursprung oben links), pro Kamera. Ein
  Objekt zählt zur Zone, wenn der Mittelpunkt der Unterkante seiner Box darin liegt. Zum
  Einzeichnen hilft ein Standbild: `uv run python scripts/grab_frame.py --camera phone`.

Vor dem ersten Einbinden einmal das Modell herunterladen und die Erkennung prüfen:

```
uv run python scripts/detect_once.py --camera sample
```

## Anbindung

`<projektordner>` ist jeweils der absolute Pfad zu diesem Ordner. `uv --directory` setzt das
Arbeitsverzeichnis, sodass `config.yaml`, `.env` und das Modell gefunden werden. Findet der Client
`uv` nicht, den absoluten Pfad zu `uv` angeben.

### Claude Code

```
claude mcp add camera -- uv --directory <projektordner> run camera-mcp
```

Danach in Claude Code mit `/mcp` prüfen, ob `camera` verbunden ist. Nach Änderungen an der
`config.yaml` oder am Code den Server dort neu verbinden.

### Claude Desktop

In `claude_desktop_config.json` eintragen und Claude Desktop neu starten:

```json
{
  "mcpServers": {
    "camera": {
      "command": "uv",
      "args": ["--directory", "<projektordner>", "run", "camera-mcp"]
    }
  }
}
```

Bekanntes Problem: Die **Store-Version (MSIX) von Claude Desktop unter Windows** startet den Server
ab Version 3.x nicht und legt auch kein Log dafür an. Der Server selbst funktioniert mit demselben
Befehl; Details in `docs/DESIGN.md` (M3). Bis dahin Claude Code verwenden oder eine andere
Installationsart von Claude Desktop.

## Beispieldialog

![Claude beantwortet über `count_in_zone` und `detect_objects`, ob die Maus auf dem Mauspad liegt](docs/img/demo.svg)

Ein echter Lauf in Claude Code mit `claude-opus-5-5` (`claude -p`, nur der Kamera-Server;
als SVG aus dem Protokoll dargestellt, Tool-Ergebnisse zusammengefasst). Die Kameras sind
Standbilder aus der lokalen Konfiguration. Ein zweites Beispiel aus der Evaluation
(`docs/eval_questions.md`), gekürzt:

**Frage:** Hängt im Gasthaus eine Uhr an der Wand?

> *Tools:* `get_camera_info` → `detect_objects(inn_still, classes=["clock"], include_image=true)`
>
> Wahrscheinlich nicht. Die Objekterkennung hat eine Uhr gemeldet, aber nur mit niedriger
> Konfidenz (0,47). Darum habe ich mir das Bild angesehen: Rechts oben an der Wand hängt ein
> runder Ring mit einer senkrechten Linie und einem Querbalken. Das sieht eher nach einem Kreuz
> aus; Ziffern oder Zeiger sind nicht zu erkennen.

Die Frage im Bild beantwortet Claude allein aus den JSON-Antworten (wenige hundert Bytes), bei der
zweiten erkennt es die falsche Erkennung und prüft das Bild. Ergebnisse über alle 15 Fragen:
`docs/EVALUATION.md`, Abschnitt 4.

## Messwerte (Kurzfassung)

Desktop-PC nur mit CPU, `yolo26n.pt`, Einzelheiten in [docs/EVALUATION.md](docs/EVALUATION.md):

- `detect_objects` antwortet in etwa 20 ms, davon rund 18 ms Inferenz.
- Ein Bild mit 1024 px Breite kostet Claude etwa 800 Tokens, die JSON-Antwort von
  `detect_objects` etwa 130.
- Die Erkennung übersieht oft kleine, entfernte, verdeckte oder viele gleichartige Objekte. Bei
  einzelnen, gut sichtbaren Objekten ist sie zuverlässig.
- Claude hat in der Evaluation alle Fragen richtig beantwortet und dafür bei Fallen der Erkennung
  von sich aus das Bild geprüft.

## Entwicklung

```
uv run pytest                             # Tests
uv run ruff check . && uv run ruff format .
uv run mcp dev src/camera_mcp/server.py   # MCP Inspector im Browser (braucht Node.js)
uv run camera-mcp                         # Server auf stdio starten
uv run python scripts/benchmark.py        # Latenz und Antwortgrößen messen
uv run python scripts/evaluate_detection.py --dir <bilderordner>   # Erkennung gegen Handzählung
```

Der Server schreibt nie auf stdout (das ist der Protokollkanal); Logs gehen nach stderr.

## Datenschutz

Bilder werden nur im Arbeitsspeicher gehalten, nicht gespeichert. Kameras nur auf eigene Räume
richten, keine öffentlichen Bereiche oder Nachbargrundstücke. Kein öffentlicher Zugriff auf den
Server; er läuft lokal über stdio.

## Lizenz

AGPL-3.0-or-later, siehe [LICENSE](LICENSE). Grund ist die Abhängigkeit `ultralytics`, die selbst
unter AGPL-3.0 steht. Der Detector ist über ein Protokoll austauschbar (`detector.py`); mit einem
anders lizenzierten Modell wäre auch eine andere Lizenz möglich.
