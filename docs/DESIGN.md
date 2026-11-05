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
