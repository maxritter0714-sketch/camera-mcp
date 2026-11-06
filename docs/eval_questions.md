# Evaluationsfragen (KONZEPT.md, Abschnitt 10.4)

Natürlichsprachliche Fragen, die Claude über die Tools beantworten soll. Jede Frage bezieht sich
auf eine Kamera aus der lokalen `config.yaml`. Damit Vorher- und Nachher-Lauf vergleichbar sind,
möglichst **Standbilder** verwenden (Datei-Kameras wie `desk_still`); bei einer Live-Kamera kann
sich die Szene zwischen den Läufen ändern.

Ausfüllen: Frage, Kamera, erwartete Antwort. Die Spalten zu den Läufen trägt Claude bei der
Auswertung ein.

## Fragen

| # | Frage (so, wie sie Claude gestellt wird) | Kamera | Erwartete Antwort |
|---|---|---|---|
| 1 | Liegt auf dem Schreibtisch-Standbild die Maus auf dem Mauspad? | `desk_still` | Ja |
| 2 | Wie viele Tastaturen siehst du auf `desk_still`? | `desk_still` | 1 |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |
| 11 | | | |
| 12 | | | |
| 13 | | | |
| 14 | | | |
| 15 | | | |

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
