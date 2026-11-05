# CLAUDE.md – camera-mcp

MCP-Server, der Kameras als Werkzeuge für KI-Clients bereitstellt (Bild holen, Objekte erkennen, in Zonen zählen). Vollständige Spezifikation: **KONZEPT.md** – vor jeder Arbeit lesen und als Quelle der Wahrheit behandeln.

## Arbeitsweise

- Arbeite **Meilenstein für Meilenstein** (KONZEPT.md, Abschnitt 9). Beginne einen Meilenstein mit einem kurzen Plan, setze ihn um, beende ihn mit grünen Tests, sauberem `ruff check` und einem Eintrag in `docs/DESIGN.md` (Entscheidung + Begründung + verworfene Alternativen).
- Nach jedem Meilenstein anhalten und kurz zusammenfassen: was gebaut wurde, was getestet ist, offene Punkte. Nicht ungefragt mit dem nächsten Meilenstein beginnen.
- **Keine neuen Abhängigkeiten** ohne Rückfrage. Keine Tools, Features oder Parameter über die Spezifikation hinaus. Stretch Goals nur auf ausdrücklichen Auftrag.
- Weicht etwas sinnvollerweise vom Konzept ab, erst fragen, dann ggf. KONZEPT.md und DESIGN.md anpassen.

## Befehle

```
uv sync                                   # Abhängigkeiten installieren
uv run pytest                             # Tests
uv run ruff check . && uv run ruff format .
uv run mcp dev src/camera_mcp/server.py   # MCP Inspector zum manuellen Testen
uv run camera-mcp                         # Server starten (stdio)
uv run python scripts/benchmark.py        # Messungen für EVALUATION.md
```

## Harte Regeln

1. **Nie auf stdout schreiben** – stdout ist der MCP-Protokollkanal. Kein `print()` im Servercode. Logging nur auf stderr/Datei (`logging_setup.py`). Ultralytics immer mit `verbose=False`.
2. **Alle MVP-Tools sind rein lesend.** Kein Shell-Zugriff, keine Dateipfade aus Tool-Parametern.
3. **Plattformunabhängig:** `pathlib`, keine OS-spezifischen Befehle; muss auf Windows und Linux, mit CPU und CUDA laufen (`device: auto`).
4. **Tests laufen offline und ohne Kamera:** Datei-Quelle aus `tests/fixtures/`, Fake-Detector statt echtem Modell (außer in klar markierten, optionalen Integrationstests).
5. **Nichts Geheimes oder Großes committen:** `config.yaml`, `.env`, `*.pt`, Aufnahmen gehören in `.gitignore`.
6. **Tool-Beschreibungen** (Docstrings) exakt wie in KONZEPT.md Abschnitt 5 übernehmen; sie sind Teil des Designs. Änderungen daran in DESIGN.md begründen.
7. **Fehler für das Modell verständlich:** präzise Meldung mit Ursache und Hinweis, keine Stacktraces in Tool-Antworten.

## Material von Max

Eigene Testbilder, ein Testvideo und die Evaluationsfragen liefert Max erst, wenn sie gebraucht werden (siehe KONZEPT.md, Abschnitt 9, „Wann wird Material von mir gebraucht?“). Bis dahin:

- Für Tests **synthetische Fixtures** erzeugen (z.B. per OpenCV generierte Bilder und ein kurzes Video mit einfachen Formen) und den Fake-Detector verwenden.
- Wird echtes Material nötig, **anhalten und konkret anfordern**: was genau (Anzahl, Motiv, Format, Länge), wohin es abgelegt werden soll und wofür es gebraucht wird. Nicht mit Platzhaltern weiterarbeiten, die echte Daten vortäuschen.

## Stil

- Python 3.12, durchgehend Typ-Hinweise, Pydantic für Konfiguration und Antwortschemas.
- Bezeichner und Code-Kommentare auf Englisch; Dokumentation in `docs/` darf Deutsch sein.
- Kleine, testbare Module gemäß Projektstruktur in KONZEPT.md Abschnitt 7.
- Koordinaten immer normiert (0..1), Zeitstempel ISO 8601 mit Zeitzone.
