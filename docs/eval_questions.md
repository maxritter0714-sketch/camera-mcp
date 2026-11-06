# Evaluationsfragen (KONZEPT.md, Abschnitt 10.4)

Natürlichsprachliche Fragen, die Claude über die Tools beantworten soll. Jede Frage bezieht sich
auf eine Kamera aus der lokalen `config.yaml`. Damit Vorher- und Nachher-Lauf vergleichbar sind,
möglichst **Standbilder** verwenden (Datei-Kameras wie `desk_still`); bei einer Live-Kamera kann
sich die Szene zwischen den Läufen ändern.

Die Fragen hat Claude entworfen und Max geprüft. Die erwarteten Antworten stützen sich auf die
geprüfte Handzählung (`recordings/eval/counts.csv`) und auf die Bilder selbst. Beantwortet werden
sie in einer **frischen Claude-Sitzung** ohne Kenntnis der Bilder oder dieser Datei; die
entwerfende Sitzung beantwortet keine Fragen.

Die Standbild-Kameras sind Datei-Kameras in der lokalen `config.yaml`: `desk_still` (Schreibtisch),
`restaurant_still` (Restauranttisch), `parking_still` (Parkplatz nachts), `inn_still` (Gasthaus).

Typ: **Z** Zählen, **J** Ja/Nein, **Zo** Zone, **B** braucht das Bild (nicht durch die
Objekterkennung abgedeckt), **F** die Objekterkennung liegt hier falsch, **M** mehrere Kameras.

## Fragen

| # | Frage (so, wie sie Claude gestellt wird) | Kamera | Typ | Erwartete Antwort |
|---|---|---|---|---|
| 1 | Liegt auf dem Schreibtisch-Standbild die Maus auf dem Mauspad? | `desk_still` | Zo | Ja |
| 2 | Wie viele Tastaturen siehst du auf `desk_still`? | `desk_still` | Z | 1 |
| 3 | Sitzt gerade jemand am Schreibtisch? | `desk_still` | J | Nein |
| 4 | Welche Farbe hat die Maus auf dem Schreibtisch? | `desk_still` | B | Weiß |
| 5 | Liegt auf dem Schreibtisch eine Brille? | `desk_still` | B | Ja, direkt hinter der Tastatur |
| 6 | Wie viele Weingläser stehen auf dem Restauranttisch? | `restaurant_still` | Z | 3 |
| 7 | Wie viele Gabeln liegen auf dem Restauranttisch? | `restaurant_still` | Z, F | 3 (erkannt wird 1) |
| 8 | Wie viele Personen sitzen am Restauranttisch? | `restaurant_still` | Z | 2 (nur Oberkörper und Hände zu sehen) |
| 9 | Was wird im Restaurant gegessen? | `restaurant_still` | B | Steak bzw. Fleisch mit Kartoffeln |
| 10 | Wie viele Autos stehen auf dem Parkplatz? | `parking_still` | Z | 4 |
| 11 | Welche Farbe hat das Auto im Vordergrund auf dem Parkplatz? | `parking_still` | B | Silber bzw. weiß mit schwarzer Motorhaube |
| 12 | Wie viele Stühle sind im Gasthaus zu sehen? | `inn_still` | Z, F | etwa 7, viele im Hintergrund (erkannt wird 1); 5–9 gilt als richtig |
| 13 | Hängt im Gasthaus eine Uhr an der Wand? | `inn_still` | J, F | Nein, das runde Objekt ist ein Kreuz (erkannt wird `clock`) |
| 14 | Sind im Gasthaus noch andere Gäste? | `inn_still` | J | Ja, eine Person hinten rechts |
| 15 | Auf welchen Standbildern sind Menschen zu sehen? | alle Standbilder | M | `restaurant_still` und `inn_still` |

Tipps für gute Fragen:

- Mischen: Zählfragen („Wie viele …?“), Ja/Nein-Fragen („Ist jemand …?“), Fragen zu Zonen
  („… auf dem Mauspad?“) und Fragen, für die die Objekterkennung nicht reicht und ein Bild nötig
  ist („Welche Farbe hat …?“, „Was liegt links neben …?“).
- Auch Fragen, bei denen die richtige Antwort „das kann ich nicht sicher sagen“ oder „nein“ ist.
- Natürlich formulieren, ohne Tool- oder Klassennamen, z. B. „Maus“ statt `mouse`.

## Läufe

Pro Frage: welche Tools Claude aufgerufen hat (in Reihenfolge), Anzahl der Aufrufe, ob die
Antwort stimmt.

| # | Vorher: Tools | Aufrufe | richtig? | Nachher: Tools | Aufrufe | richtig? |
|---|---|---:|---|---|---:|---|
| 1 | | | | | | |
