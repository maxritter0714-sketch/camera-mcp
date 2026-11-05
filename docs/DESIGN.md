# Entscheidungslog

Je Meilenstein: Entscheidung, Begründung, verworfene Alternativen. Spezifikation: [KONZEPT.md](../KONZEPT.md).

## M0 – Projektgerüst und Konfiguration

### `.env` mit `python-dotenv` lesen

KONZEPT.md Abschnitt 6 verlangt `${VAR}` aus Umgebung bzw. `.env`, Abschnitt 4 nennt dafür keine Bibliothek. `python-dotenv` wurde nach Rückfrage ergänzt. `dotenv_values()` liest die Datei, ohne `os.environ` zu verändern; echte Umgebungsvariablen haben Vorrang.

Verworfen:

- **Eigener Parser:** Quoting und Sonderzeichen sind gerade bei Passwörtern in Stream-URLs fehleranfällig.
- **`pydantic-settings`:** Ist für Felder einer Settings-Klasse gedacht, nicht für Platzhalter in YAML. Die Ersetzung wäre trotzdem nötig gewesen.

### `${VAR}` erst nach dem YAML-Parsen ersetzen

Die Ersetzung läuft über die String-Werte der geparsten Struktur. Sonderzeichen in Werten (`#`, `:`, `{`) können so das YAML nicht verändern, und der Fehler bei einer fehlenden Variable nennt den Feldpfad.

Verworfen: Textersetzung vor dem Parsen. Sie ist einfacher, aber Passwörter mit YAML-Sonderzeichen würden die Struktur zerstören oder verfälschen.

### Fehlermeldungen ohne Werte

Pydantics Standardtext enthält die Eingabewerte, also auch Stream-URLs mit Zugangsdaten. `ConfigError` gibt deshalb nur Feldpfad und Ursache aus (`errors(include_input=False)`).

Verworfen: `str(ValidationError)` – wäre kürzer, schreibt aber Geheimnisse ins Log.

### Pfade relativ zum Ordner der Config-Datei

Relative Dateiquellen und `.env` werden vom Ordner der Config-Datei aus aufgelöst. Das Verhalten hängt so nicht davon ab, aus welchem Verzeichnis der Client den Server startet.

Verworfen: Arbeitsverzeichnis – ein MCP-Client legt es oft nicht fest, Fehler wären schwer nachzuvollziehen.

### Streng validieren, aber nur die Konfiguration selbst

Unbekannte Schlüssel werden abgelehnt (`extra="forbid"`), damit Tippfehler nicht still den Standardwert ergeben. Eine fehlende `${VAR}` verhindert den Start, auch wenn nur eine Kamera betroffen ist (KONZEPT.md Abschnitt 6).

Ob eine Dateiquelle existiert, prüft die Konfiguration nicht: Zur Laufzeit erreichbare Quellen sind Sache von M1 und zeigen sich dort im Kamerastatus.

### `main()` vor dem MCP-Server

`server.py` enthält in M0 nur `main()`: Config laden, bei Fehler Meldung auf stderr und Exit-Code 1. Damit ist das Abnahmekriterium über den späteren Einstiegspunkt prüfbar. FastMCP und `mcp` kommen in M3.

### Abhängigkeiten schrittweise

`opencv-python`, `ultralytics` und `mcp` werden erst in den Meilensteinen ergänzt, die sie brauchen. Das hält M0 schnell und vermeidet den großen Torch-Download, bevor er gebraucht wird.

### Logging nur auf stderr

`setup_logging()` ersetzt alle Root-Handler durch einen stderr-Handler, weil stdout der MCP-Protokollkanal ist. Ein Subprozess-Test prüft, dass `main()` nichts auf stdout schreibt. Zusätzlich verbietet die Ruff-Regel `T20` `print()` außerhalb von `scripts/`.

## M1 – FrameSources und CameraManager

### Die Quelle bremst Videodateien, nicht der Manager

`read()` blockiert, bis das nächste Bild fällig ist, bei Videodateien also passend zu deren FPS. Eine Datei verhält sich damit wie eine Kamera, der Manager bleibt quellenneutral, und `fps_measured` stimmt.

Verworfen:

- **Einheitliche Bremse im Manager:** Bei echten Kameras würde sie den OpenCV-Puffer volllaufen lassen und genau die veralteten Bilder erzeugen, die der Hintergrund-Thread verhindern soll (KONZEPT.md Abschnitt 8).
- **Keine Bremse:** Volle CPU-Last und unrealistische FPS, was die Evaluation verfälschen würde.

### Getrennte Klassen für Video- und Bilddateien

`make_source` wählt anhand der Dateiendung zwischen `VideoFileFrameSource` und `ImageFileFrameSource`. Beide haben keine gemeinsame Lese-Logik, weil ein Bild einmal dekodiert und ein Video laufend gelesen wird.

Verworfen: eine `FileSource` mit Fallunterscheidung in jeder Methode.

### `online`/`offline` aus dem Bildalter ableiten

Der Status wird beim Abruf aus `max_frame_age_s` berechnet und nicht vom Thread gesetzt. Das erfasst auch den schwierigsten Fall: Ein Stream, der in einem blockierenden `read()` hängt, kann selbst nichts mehr melden. OpenCV bricht dort erst nach einem eigenen Timeout ab.

Verworfen: Statusvariable, die der Capture-Thread pflegt – sie bliebe genau in diesem Fall auf `online` stehen.

### Fehlerarten der Quellen

`SourceError` bedeutet „vielleicht wieder verfügbar“ und führt zum Reconnect nach `reconnect_interval_s`. `SourceEnded` bedeutet „endgültig zu Ende“ (`loop: false`) und beendet den Thread: Ein Reconnect würde die Datei neu starten, also doch eine Schleife ergeben. Die Kamera wird danach über das Bildalter offline.

Andere Ausnahmen fängt der Thread ebenfalls ab, weil ein Kamerafehler den Server nie beenden darf. Von ihnen wird nur der Typ geloggt, da ihr Text Stream-URLs mit Zugangsdaten enthalten könnte.

### Bilder sind schreibgeschützt

Der Manager setzt `writeable = False` auf jedes Bild, weil mehrere Aufrufer dasselbe Array teilen. Annotation (M2) muss kopieren. Ein Versehen ergibt so einen Fehler statt eines still veränderten Bildes.

Verworfen: Kopie bei jedem Abruf – kostet bei jedem Tool-Aufruf Speicher und Zeit, auch wenn niemand schreibt.

### Stoppen wartet begrenzt

`stop()` wartet höchstens `STOP_TIMEOUT_S`. Ein in OpenCV hängender Thread lässt sich nicht von außen beenden; deshalb sind die Threads Daemons und werden mit einer Warnung zurückgelassen, statt das Beenden des Servers zu blockieren.

### Synthetische Fixtures eingecheckt

`scripts/generate_fixtures.py` erzeugt `tests/fixtures/sample.mp4` und `sample.png` (bewegtes Rechteck, keine echten Aufnahmen). Sie liegen im Repo, weil `config.example.yaml` auf `sample.mp4` verweist und die Tests offline laufen müssen.

Verworfen: Erzeugung beim Testlauf – der Pfad aus der Beispiel-Config wäre vor dem ersten Testlauf ungültig.

### `grab_frame.py` startet nur die angeforderte Kamera

So verzögern andere, gerade nicht erreichbare Kameras das Skript nicht. Das Warten auf das erste Bild liegt bewusst im Skript: Der Manager bietet nur nicht blockierende Abrufe an.

## M2 – Detector und Imaging

### Klassenfilter nach der Erkennung

Der Detector liefert alle Treffer, `detect_in_frame` filtert, zählt und prüft Klassennamen. Fake und echtes Modell laufen so durch denselben Code, und jeder künftige Detector bekommt das Verhalten ohne eigene Implementierung. Unbekannte Klassen werden vor der Erkennung abgelehnt.

Verworfen: Filter im Modellaufruf (`predict(classes=...)`). Das Netz läuft trotzdem vollständig, der Gewinn beschränkt sich auf die Nachbearbeitung, und jeder Detector müsste es selbst können.

### Ultralytics von stdout fernhalten

Der Logger von Ultralytics bindet sich beim Import an `sys.stdout`, meldet dort beim ersten Import das Anlegen seiner Einstellungsdatei, und auch der Fortschrittsbalken des Modell-Downloads schreibt nach stdout. Auf einem frischen Rechner würde das die MCP-Verbindung zerstören. `verbose=False` allein reicht also nicht.

Import und Laden des Modells laufen deshalb in `redirect_stdout(sys.stderr)`. Der Logger bleibt dadurch dauerhaft an stderr gebunden. Bei der Erkennung selbst wird nicht umgeleitet, weil das prozessweit wirkt und im Server mit dem Protokollkanal kollidieren könnte; dort genügt `verbose=False`.

Verworfen: Handler des Loggers nach dem Import austauschen – die Meldung beim ersten Import wäre dann schon auf stdout gelandet.

### Netzwerkfunktionen von Ultralytics abschalten

Ultralytics sendet bei jeder Erkennung Nutzungsstatistiken und kann fehlende Pakete selbst installieren. Für einen Server, der Kameras verarbeitet, ist beides unerwünscht. Vor dem ersten Import werden deshalb `YOLO_OFFLINE=true` und `YOLO_AUTOINSTALL=false` gesetzt. Laut Code bleibt der Modell-Download möglich, weil er `YOLO_OFFLINE` nur nach einem Fehlversuch auswertet.

Verworfen: `settings.update({"sync": False})` – das würde die globale Ultralytics-Einstellungsdatei des Benutzers ändern und damit auch andere Projekte betreffen.

### Lizenz: AGPL-3.0 von `ultralytics`

Für einen privaten Prototyp unkritisch. Wird das Repo öffentlich, muss das Projekt entweder selbst unter AGPL-3.0 stehen oder einen anderen Detector verwenden. Genau dafür gibt es das `Detector`-Protokoll; `ultralytics` wird nur in `detector.py` importiert.

### Bilder bleiben schreibgeschützt, Annotation kopiert

`annotate()` arbeitet auf einer Kopie, weil Frames aus dem `CameraManager` geteilt und schreibgeschützt sind (siehe M1).

### `frame_age_s` wird nach der Erkennung gemessen

So zeigt der Wert, wie alt das Ergebnis ist, wenn der Aufrufer es erhält. Gerade auf der CPU kann die Erkennung einen spürbaren Teil dieses Alters ausmachen. `timestamp` ist dagegen der Aufnahmezeitpunkt des Bildes.

### `zones` ist bis M4 leer

Das Feld existiert schon, damit das Antwortformat aus KONZEPT.md Abschnitt 5 stabil bleibt. Befüllt wird es mit den Zonen in M4.

### Beobachtung: erste Erkennung deutlich langsamer

Einmalige Messung auf der CPU mit `yolo26n.pt` und dem synthetischen Testbild (320×240): erster Aufruf rund 1285 ms, die zehn folgenden im Median rund 23 ms. Das ist noch keine Evaluation, hat aber zwei Folgen:

- **M3:** Der erste Tool-Aufruf nach dem Start wäre spürbar langsamer. Zu entscheiden ist, ob der Server beim Start einmal aufwärmt.
- **M5:** Der Benchmark muss den ersten Aufruf getrennt erfassen, sonst verfälscht er Median und 95. Perzentil.
