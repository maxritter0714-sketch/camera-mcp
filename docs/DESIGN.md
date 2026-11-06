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

## M3 – MCP-Server

### `mcp` 2.x statt 1.x (Abweichung vom ursprünglichen Konzept)

KONZEPT.md nannte `from mcp.server.fastmcp import FastMCP, Image`. In der aktuellen Hauptversion heißt das `from mcp.server.mcpserver import MCPServer, Image`; das alte Modul existiert nicht mehr. Nach Rückfrage wurde 2.x gewählt und KONZEPT.md angepasst: Das Projekt beginnt gerade erst, ein späterer Umstieg wäre teurer, und v2 hält den Text unerwarteter Fehler von Haus aus vom Client fern.

Verworfen: `mcp<2` festnageln – exakt wie spezifiziert, aber der Start auf einer alten Hauptversion.

### Synchrone Tool-Funktionen

Das SDK führt synchrone Tools in einem Worker-Thread aus. Erkennung und JPEG-Encoding blockieren den Protokollkanal also nicht, ohne dass der Code sich darum kümmern muss. Die Detector-Aufrufe serialisiert weiterhin das Lock im `UltralyticsDetector`.

Verworfen: `async`-Tools mit eigener Thread-Auslagerung – der ursprüngliche Plan, im SDK aber überflüssig.

### Aufwärmen beim Start

Der Lifespan führt einmal eine Erkennung auf einem leeren Bild aus. Der erste echte `detect_objects`-Aufruf dauerte in einer Einzelmessung (CPU, über MCP) danach rund 79 ms statt der rund 1,3 s für die erste Erkennung aus M2; die folgenden zehn lagen im Median bei rund 38 ms. Die Verbindung stand nach rund 4,7 s.

Das geht über die Spezifikation hinaus, ändert aber keine Tools oder Parameter und wurde vorab abgestimmt.

Verworfen: kein Aufwärmen – der Start wäre kürzer, aber ausgerechnet der erste Aufruf des Modells deutlich langsamer.

### Lifespan und Serverobjekt

`create_server(make_detector)` baut den Server, damit Tests einen Fake-Detector einsetzen können. Das modulweite `mcp` braucht `mcp dev`, das den Server über den Namen findet und `main()` nicht aufruft. Config, Modell und Kameras werden deshalb erst im Lifespan geladen; ein Import von `server.py` lädt nichts davon.

`main()` prüft die Config zusätzlich vorab und beendet den Prozess bei Config- oder Modellfehlern mit Exit-Code 1 und einer Meldung auf stderr.

### Fehler

`CameraError`, `DetectorError` und `ImagingError` werden zu `ToolError` mit lesbarer Meldung. Unerwartete Ausnahmen behandelt das SDK als Absturz: Das Modell sieht nur `Error executing tool <name>`, die Details gehen ins Log.

Eine Offline-Kamera ist bei `get_camera_info` ein Status und kein Fehler, weil genau dieses Tool den Zustand melden soll. `get_frame` und `detect_objects` geben dagegen eine Fehlermeldung mit Bildalter zurück, weil sie kein aktuelles Ergebnis liefern können.

### Antwortformat

`get_camera_info` gibt ein Pydantic-Modell zurück, das SDK liefert dazu strukturierte Inhalte. `get_frame` und `detect_objects` geben Text und optional Bild zurück (`structured_output=False`), weil sich beides nicht in einem Schema ausdrücken lässt. Die JSON-Antwort von `detect_objects` steht als Text im ersten Block.

`max_width` wird still auf `max_image_width` begrenzt (KONZEPT.md Abschnitt 5). Die tatsächlich gelieferte Breite steht in der Antwort.

### Start aus Claude Desktop über `uv --directory`

Claude Desktop legt das Arbeitsverzeichnis nicht fest. `uv --directory <projektordner> run camera-mcp` setzt es auf den Projektordner, sodass `config.yaml`, `.env` und die Modelldatei ohne Codeänderung gefunden werden. Damit ist die offene Frage aus M2 zum Ablageort des Modells gelöst.

Verworfen: Modellpfad relativ zur Config auflösen – mehr Code für einen Fall, den der Startbefehl bereits abdeckt.

### Beschreibungen gegen KONZEPT.md getestet

Die Tool-Beschreibungen sind Teil des Designs (CLAUDE.md, Regel 6). `test_tools.py` liest sie aus KONZEPT.md Abschnitt 5 und vergleicht sie mit dem, was ein echter Client sieht. Ändert sich eines von beiden, schlägt der Test fehl. Das hilft auch beim Vorher-Nachher-Vergleich der Beschreibungen in M5.

### Abnahme mit Claude Code statt Claude Desktop (Abweichung vom Abnahmekriterium)

KONZEPT.md verlangt für M3, dass Claude Desktop die Tools nutzen kann. Die Abnahme lief nach Rückfrage mit Claude Code als echtem Client (`claude mcp add camera -- uv --directory <projektordner> run camera-mcp`). Dort rief das Modell `get_camera_info` auf und gab die Kamera aus der Config korrekt wieder.

Claude Desktop (Windows, Store-/MSIX-Installation) startete den Server nicht. Die App versuchte es nicht einmal, es entstand keine `mcp-server-camera.log`. Bis Version 2.x protokollierte sie beim Start, dass sie `claude_desktop_config.json` im virtualisierten Paketordner liest. Seit dem Update auf 3.x fehlt dieser Eintrag, und der dort eingetragene Server wird ignoriert. Öffentliche Fehlerberichte beschreiben dasselbe Verhalten der Store-Version, ohne eine Lösung zu nennen. Der Server selbst ist nicht die Ursache: Mit genau dem Befehl aus der Desktop-Konfiguration startet er und beantwortet alle drei Tools.

Offen: die Abnahme in Claude Desktop nachholen, z. B. mit einer anderen Version oder Installationsart. Das gilt auch für die gleichlautenden Kriterien in M4 und M6.

Verworfen: den Server als Desktop-Erweiterung verpacken, die die Store-Version lädt. Das wäre ein zusätzliches Paketformat außerhalb der Spezifikation, nur um einen Fehler des Clients zu umgehen.

## M4 – Zonen und `count_in_zone`

### Fußpunkt statt Box-Mittelpunkt

Ein Objekt gehört zu einer Zone, wenn der Mittelpunkt der Unterkante seiner Box in ihr liegt (KONZEPT.md Abschnitt 5). Für die typische Frage „steht jemand in der Zone?“ ist der Bodenkontakt aussagekräftiger als die Bildmitte des Objekts. Eine Person vor einer Tür überdeckt die Tür im Bild fast immer, steht aber nur mit den Füßen in der davor gezeichneten Bodenfläche. Ein Test hält das fest: Eine Box, deren Mitte in der Zone liegt, die aber darunter steht, zählt nicht.

Verworfen:
- Box-Mittelpunkt: zählt Objekte, die nur ins Bild der Zone hineinragen.
- Überlappungsanteil der Box: braucht einen zusätzlichen Schwellwert und ist für das Modell schwerer nachzuvollziehen.

### Der Rand zählt zur Zone

Punkte auf einer Kante oder Ecke liegen in der Zone, mit einer Toleranz von 1e-9. Eine Zone, die bis an den Bildrand gezeichnet ist, zählt so auch Objekte, deren Box den Rand berührt. Ultralytics beschneidet Boxen auf das Bild, der Fußpunkt liegt dann genau bei y = 1. Liegt ein Punkt auf der gemeinsamen Kante zweier benachbarter Zonen, gehört er zu beiden.

Verworfen: Rand außerhalb – dann fielen gerade die Objekte am unteren Bildrand heraus, die der Kamera am nächsten sind.

### Eigener Punkt-in-Polygon-Test

`zones.py` prüft zuerst, ob der Punkt auf einer Kante liegt. Danach entscheidet Ray Casting nach der Gerade-Ungerade-Regel. Das sind wenige Zeilen reines Python, das Verhalten auf dem Rand steht fest, und die Randfälle aus dem Abnahmekriterium sind direkt testbar. Getestet sind Kanten, Ecken, schräge Kanten, ein konkaves Polygon, ein Strahl genau durch eine Ecke, der Bildrand und überlappende Zonen.

Sich selbst kreuzende Polygone werden nicht abgelehnt. Die Gerade-Ungerade-Regel zählt dort überlappende Teile als außen. Ein Polygon ohne Fläche enthält nur seine Kanten. Beides geht nicht über die Prüfung der Config hinaus, die für M0 spezifiziert wurde.

Verworfen:
- `cv2.pointPolygonTest`: Es ist zwar schon installiert, rechnet aber mit float32. Ob ein Punkt auf dem Rand liegt, hinge dann von Rundungen ab.
- `shapely`: eine neue Abhängigkeit für eine Funktion von 30 Zeilen.

### Zonen auf der gerundeten Box

`detect_in_frame` ordnet die Zonen nach dem Runden der Koordinaten zu. So passt die Zuordnung immer zu der Box, die das Modell in der Antwort sieht.

### Antwort von `count_in_zone`

`count_in_zone` ruft dieselbe Erkennung auf wie `detect_objects` und behält nur die Detektionen der Zone. `counts` gilt damit nur für die Zone. Die Antwort enthält nach Rückfrage neben den spezifizierten Feldern Zone, Zeitstempel, `counts` und Detektionen auch `camera_id` und `frame_age_s`. So passt sie zu `detect_objects`, und das Modell sieht, wie alt das Ergebnis ist. Ausgeliefert wird sie als strukturierte Ausgabe wie bei `get_camera_info`, weil kein Bild dazugehört.

Die Zone wird vor der Erkennung geprüft. Eine unbekannte Zone liefert eine Fehlermeldung mit den Zonen der Kamera, oder den Hinweis, dass die Kamera keine hat, und kostet keine Inferenz. Eine unbekannte oder ausgefallene Kamera meldet sich mit derselben Fehlermeldung wie bei den anderen Tools.

## M5 – Evaluation (in Arbeit)

### Latenz über echtes MCP statt direkter Funktionsaufrufe

`scripts/benchmark.py` startet den Server als Subprozess und ruft die Tools als MCP-Client über stdio auf, wie es auch `tests/test_mcp_e2e.py` tut. Gemessen wird damit, was ein Client wie Claude erlebt: Protokoll, Serialisierung, der Wechsel in den Worker-Thread, Inferenz und Bildkodierung. `inference_ms` aus der Antwort von `detect_objects` steht daneben, sodass sich der Anteil von Server und Protokoll ablesen lässt. Vor der Messung laufen drei Aufwärmaufrufe pro Fall; die Kamera muss `online` sein.

Verworfen: die Tool-Funktionen direkt im Prozess aufzurufen. Das wäre genauer für die Inferenz allein, ließe aber den Anteil weg, den das Konzept mit „Latenz pro Tool“ meint.

### Bildgrößen ohne das konfigurierte Limit

Der Server begrenzt jedes Bild auf `limits.max_image_width` (Standard 1024 px). Ein Aufruf mit 1600 px liefert also 1024 px. Die Antwortgrößen für 512, 1024 und 1600 px misst das Skript deshalb im Prozess mit denselben Funktionen, die der Server nutzt (`resize_to_width`, `encode_jpeg`), aber ohne das Limit. So bleibt die echte Konfiguration unverändert, und die Messung beantwortet, ob sich ein höheres Limit lohnt.

Verworfen:
- den Server mit einer Config-Kopie mit höherem Limit starten: misst dann eine Konfiguration, die so nicht betrieben wird.
- 1600 px weglassen: dann bleibt offen, ob das Limit von 1024 px gut gewählt ist.

### Token-Schätzung für Bilder nach der aktuellen Anthropic-Dokumentation

Bilder kosten ⌈Breite / 28⌉ × ⌈Höhe / 28⌉ Tokens. Größere Bilder verkleinert Claude vorher, je nach Modell auf der Standard-Stufe (1568 px / 1568 Tokens) oder der hochauflösenden Stufe (2576 px / 4784 Tokens, Claude 4.7 und neuer). Das Skript übernimmt die Referenzimplementierung aus der Dokumentation und weist beide Stufen aus; Tests prüfen sie gegen die dort genannten Beispiele. Bei 2000×1500 weichen Tabelle (1269×952) und Referenzimplementierung (1270×952) um ein Pixel voneinander ab; das Skript folgt der Implementierung, die Tokens sind gleich. Text wird grob mit Bytes / 4 geschätzt.

Verworfen: die ältere Faustregel Breite × Höhe / 750 mit Obergrenze ~1600 Tokens. Sie steht noch in älteren Quellen, ist aber überholt und hätte die Kosten bei 1600 px für hochauflösende Modelle falsch angegeben.

### Ausgabe nur in ASCII

Die Markdown-Ausgabe des Skripts enthält nur ASCII-Zeichen. Die Windows-Konsole kodiert stdout mit cp1252 und bricht bei Zeichen wie „≈“ ab. Ein Test prüft das.

### Robustheit: keine neuen Tests, sondern Abgleich und Live-Prüfung

Die Fehlerfälle aus KONZEPT.md Abschnitt 10.5 waren durch die Tests aus M1–M4 bereits abgedeckt. In `EVALUATION.md` steht pro Fall die Antwort, die der Server live über Claude Code gab, und der Test, der das Verhalten festhält. Der Stream-Abbruch wird zusätzlich live mit dem Handy über RTSP geprüft.

### Feste Timeouts für Netzwerk-Streams

`StreamFrameSource` öffnet Streams mit je 5 s Timeout für Öffnen und Lesen (`CAP_PROP_OPEN_TIMEOUT_MSEC`, `CAP_PROP_READ_TIMEOUT_MSEC`). Der Live-Test hat gezeigt, dass ohne diese Werte ein Verbindungsversuch zu einer ausgeschalteten RTSP-Kamera 30 s hängt, das Standard-Timeout von OpenCV/FFmpeg. Der Reconnect dauerte dadurch bis zu ~35 s statt ~5 s. Mit 5 s brechen vergebliche Versuche nach 5 s ab; der Abstand der Versuche liegt bei ~10 s (Messwerte in `EVALUATION.md`). 5 s passt zu `max_frame_age_s`: Ein Stream, der länger keine Bilder liefert, gilt ohnehin als offline. Das Lese-Timeout greift zusätzlich, wenn ein Stream offen bleibt, aber keine Daten mehr schickt.

Verworfen:
- eine Option in der Config: Für den Prototyp gibt es keinen Fall, der einen anderen Wert braucht; sie wäre ein Parameter über die Spezifikation hinaus.
- kürzere Werte: Der erste Verbindungsaufbau über RTSP mit H.264 braucht im Test bis zu ~2 s; 5 s lässt genug Reserve für ein langsameres WLAN.

### Erkennungsqualität über Anzahlen statt Boxen

`scripts/evaluate_detection.py` vergleicht pro Bild und Klasse die Anzahl der Erkennungen mit einer Handzählung (`counts.csv`). Das passt zur Frage, die die Tools beantworten („wie viele …?“, `count_in_zone`), und die Referenz lässt sich in Excel ausfüllen. „Übersehen“ und „zu viel“ sind Differenzen der Anzahlen. Eine falsche und eine übersehene Erkennung derselben Klasse im selben Bild heben sich dabei auf; das ist in `EVALUATION.md` genannt.

Jedes Bild wird einmal bei der niedrigsten Schwelle erkannt, die höheren Schwellen filtern dieses Ergebnis. Das ist gleichwertig und viermal schneller. „Klassenzählungen exakt“ bezieht sich für alle Schwellen auf dieselben Paare aus Bild und Klasse, nämlich die, bei denen Referenz oder Erkennung bei der niedrigsten Schwelle über 0 liegt. Ein erster Entwurf nahm pro Schwelle eine eigene Bezugsmenge; die Prozente waren dadurch nicht vergleichbar. Paare, die überall 0 sind, zählen nicht, weil sie das Ergebnis schönen würden.

Erkennungen von Klassen ohne Spalte in der Referenz werden als „andere Klassen“ getrennt gezählt, nicht als Fehler. So muss die Referenz nicht alle 80 Klassen abdecken.

Verworfen:
- Boxen von Hand zeichnen und mAP bzw. Precision/Recall mit IoU berechnen: genauer, aber ein Vielfaches an Aufwand für die Referenz und ein zusätzliches Werkzeug. Für einen Prototyp, dessen Tools Anzahlen liefern, nicht nötig.
- getrennte Inferenz pro Schwelle: gleiches Ergebnis, viermal langsamer.

### Kontrolldurchgang der Handzählung

Die erste Handzählung war bei vollen Bildern unvollständig. Claude hat danach alle Bilder angesehen und Korrekturen vorgeschlagen; übernommen wurde nur, was Max bestätigt hat. Jede Änderung steht in `recordings/eval/review_log.txt`. Zählregeln, die dabei festgelegt wurden (Hand zählt als `person`, Gläser ohne Stiel als `cup`, realistische Modelle zählen, stilisierte Figuren nicht), stehen in `EVALUATION.md`.

Verworfen: die erste Zählung unverändert verwenden. Sie hätte viele richtige Erkennungen als Fehler gewertet (26 statt 7 „zu viel“ auf denselben Bildern) und das Modell schlechter dargestellt, als es ist.
