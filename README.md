# camera-mcp

MCP-Server, der Kameras als Werkzeuge für KI-Clients bereitstellt. Spezifikation: [KONZEPT.md](KONZEPT.md).

Status: in Entwicklung. Die vollständige Anleitung folgt mit Meilenstein M5.

## Einrichten

```
uv sync
uv run pytest
```

`config.example.yaml` nach `config.yaml` kopieren und anpassen. Kameras, die du nicht nutzt, entfernst du dort; eine fehlende `${VAR}` verhindert sonst den Start. Stream-URLs gehören in eine `.env` neben der `config.yaml` (Vorlage: `.env.example`).

Vor dem ersten Einbinden einmal das Modell herunterladen und die Erkennung prüfen:

```
uv run python scripts/detect_once.py --camera sample
```

## Anbindung an Claude Desktop

In die Konfiguration von Claude Desktop (`claude_desktop_config.json`) eintragen und Claude Desktop neu starten. `<projektordner>` ist der absolute Pfad zu diesem Ordner.

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

Findet Claude Desktop `uv` nicht, bei `command` den absoluten Pfad zu `uv` angeben.

## Manuell testen

```
uv run mcp dev src/camera_mcp/server.py
```

Das startet den MCP Inspector im Browser (braucht Node.js).
