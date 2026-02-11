from __future__ import annotations
from ..model import Cell, Entry, Grid
from .registry import register_puzzle

def create_grid() -> Grid:
    cells = [[Cell(row=r, col=c) for c in range(15)] for r in range(15)]

    # -----------------------------
    # Define your Entries here:
    # Example:
    # entries: dict[str, Entry] = {
    #     "1A": Entry("1A", "Bar that connecta rotating wheels", "AXLE", cells, (0,0), 4),
    #     ...
    # }
    # -----------------------------
    entries: dict[str, Entry] = {
        "1A": Entry("1A", "Pie nut", "PECAN", cells, (0,0), 5),
        "6A": Entry("6A", "Boring task", "SLOG", cells, (0,6), 4),
        "10A": Entry("10A", "'I want to be in the loop on this email' (2 words)", "CCME", cells, (0,11), 4),
        "14A": Entry("14A", "Love, Italian-style", "AMORE", cells, (1,0), 5),
        "15A": Entry("15A", "Walking stick", "CANE", cells, (1,6), 4),
        "16A": Entry("16A", "Sounds of inspiration", "AHAS", cells, (1,11), 4),
        "17A": Entry("17A", "Classic flowering locale adjacent to the White House (2 words)", "ROSEGARDEN", cells, (2,0), 10),
        "19A": Entry("19A", "Get gooey, maybe", "MELT", cells, (2,11), 4),
        "20A": Entry("20A", "Backstreets", "ALLEYS", cells, (3,3), 6),
        "21A": Entry("21A", "Spooky", "EERIE", cells, (3,10), 5),
        "22A": Entry("22A", "___ one and only", "THE", cells, (4,0), 3),
        "25A": Entry("25A", "Itinerary calculation, in brief", "ETA", cells, (4,4), 3),
        "26A": Entry("26A", "Connections", "INS", cells, (4,8), 3),
        "28A": Entry("28A", "Purple yam used to flavor desserts", "UBE", cells, (4,12), 3),
        "29A": Entry("29A", "Grammy Award that's good for laughs? (3 words)", "BESTCOMEDYALBUM", cells, (5,0), 15),
        "33A": Entry("33A", "Truffle pigs' assets", "SNOUTS", cells, (6,0), 6),
        "34A": Entry("34A", "Connection between topics", "SEGUE", cells, (6,7), 5),
        "35A": Entry("35A", "'Over here!'", "PSST", cells, (7,0), 4),
        "36A": Entry("36A", "___ Vicious", "SID", cells, (7,6), 3),
        "37A": Entry("37A", "Increase (2 words)", "GOUP", cells, (7,11), 4),
        "41A": Entry("41A", "'Do the Right _____' (Spike Lee film)", "THING", cells, (8,3), 5),
        "44A": Entry("44A", "Food court pizza chain", "SBARRO", cells, (8,9), 6),
        "46A": Entry("46A", "North Carolina educational institution (2 words)", "DAVIDSONCOLLEGE", cells, (9,0), 15),
        "50A": Entry("50A", "Allhallows ___", "EVE", cells, (10,0), 3),
        "51A": Entry("51A", "Cut, as grass", "MOW", cells, (10,4), 3),
        "52A": Entry("52A", "Beverage that may come in a pint", "ALE", cells, (10,8), 3),
        "53A": Entry("53A", "Cry betweeb 'Ready' and 'Go!'", "SET", cells, (10,12), 3),
        "54A": Entry("54A", "Parisian thanks", "MERCI", cells, (11,0), 5),
        "56A": Entry("56A", "Waiter or waitress", "SERVER", cells, (11,6), 6),
        "59A": Entry("59A", "Do ____ others", "UNTO", cells, (12,0), 4),
        "60A": Entry("60A", "Athletic trifectas", "THREEPEATS", cells, (12,5), 10),
        "64A": Entry("64A", "Wishes undone", "RUES", cells, (13,0), 4),
        "65A": Entry("65A", "Fireplace fuel", "LOGS", cells, (13,5), 4),
        "66A": Entry("66A", "Jadedness", "ENNUI", cells, (13,10), 5),
        "67A": Entry("67A", "Send an eggplant or taco emoji, say", "SEXT", cells, (14,0), 4),
        "68A": Entry("68A", "Ones appointed by corp boards", "CEOS", cells, (14,5), 4),
        "69A": Entry("69A", "No-nos", "DONTS", cells, (14,10), 5),
        "1D": Entry("1D", "Number on a golf card", "PAR", cells, (0,0), 3),
        "2D": Entry("2D", "Music genre influenced by the Smiths and the Cure", "EMO", cells, (0,1), 3),
        "3D": Entry("3D", "Modern lead-in to play", "COS", cells, (0,2), 3),
        "4D": Entry("4D", "Space", "AREA", cells, (0,3), 4),
        "5D": Entry("5D", "Fail to take care of", "NEGLECT", cells, (0,4), 7),
        "6D": Entry("6D", "'Aieeeee!', e.g.", "SCREAM", cells, (0,6), 6),
        "7D": Entry("7D", "Noblewoman", "LADY", cells, (0,7), 4),
        "8D": Entry("8D", "Like a biased presentation (2 words)", "ONESIDED", cells, (0,8), 8),
        "9D": Entry("9D", "Lead-in to X, Y, Z or Alpha", "GEN", cells, (0,9), 3),
        "10D": Entry("10D", "Arrived", "CAME", cells, (0,11), 4),
        "11D": Entry("11D", "Winged figure in Raphael's 'Sistene Madonna'", "CHERUB", cells, (0,12), 6),
        "12D": Entry("12D", "Beach city west of Los Angeles", "MALIBU", cells, (0,13), 6),
        "13D": Entry("13D", "Think highly of", "ESTEEM", cells, (0,14), 6),
        "18D": Entry("18D", "Stevie Nicks and Karen Carpenter, vocally", "ALTOS", cells, (2,5), 5),
        "21D": Entry("21D", "Jacob's brother in the bible", "ESAU", cells, (3,10), 4),
        "22D": Entry("22D", "Baker's meas.", "TBSP", cells, (4,0), 4),
        "23D": Entry("23D", "Mothers of barnyard chicks", "HENS", cells, (4,1), 4),
        "24D": Entry("24D", "Those, in Spanish", "ESOS", cells, (4,2), 4),
        "27D": Entry("27D", "NFL team at Metlife stadium, as shown on scoreboards", "NYG", cells, (4,9), 3),
        "30D": Entry("30D", "_____-frutti", "TUTTI", cells, (5,3), 5),
        "31D": Entry("31D", "Endorse digitally", "ESIGN", cells, (5,7), 5),
        "32D": Entry("32D", "Corporate department that handles contracts", "LEGAL", cells, (5,11), 5),
        "36D": Entry("36D", "Oversize article of winter footwear", "SNOWSHOE", cells, (7,6), 8),
        "38D": Entry("38D", "Metals that miers mine", "ORES", cells, (7,12), 4),
        "39D": Entry("39D", "Egg (on)", "URGE", cells, (7,13), 4),
        "40D": Entry("40D", "T.S. Eliot or W.H. Auden", "POET", cells, (7,14), 4),
        "42D": Entry("42D", "Kind of port in A/V", "HDMI", cells, (8,4), 4),
        "43D": Entry("43D", "Prefix with -metric or -morphic", "ISO", cells, (8,5), 3),
        "44D": Entry("44D", "Puzzle out", "SOLVE", cells, (8,9), 5),
        "45D": Entry("45D", "Censored, as on an audiotape", "BLEEPED", cells, (8,10), 7),
        "46D": Entry("46D", "Takes exception", "DEMURS", cells, (9,0), 6),
        "47D": Entry("47D", "Fifth or Madison, in Manhattan", "AVENUE", cells, (9,1), 6),
        "48D": Entry("48D", "Point of a polygon", "VERTEX", cells, (9,2), 6),
        "49D": Entry("49D", "Stroke lovingly", "CARESS", cells, (9,8), 6),
        "55D": Entry("55D", "'It'll ____ ya", "COST", cells, (11,3), 4),
        "57D": Entry("57D", "Thus", "ERGO", cells, (11,7), 4),
        "58D": Entry("58D", "Nevada city near Lake Tahoe", "RENO", cells, (11,11), 4),
        "60D": Entry("60D", "Extra attention, in brief", "TLC", cells, (12,5), 3),
        "61D": Entry("61D", "Journalist Curry", "ANN", cells, (12,12), 3),
        "62D": Entry("62D", "Famed Egyptian king, for short", "TUT", cells, (12,13), 3),
        "63D": Entry("63D", "Bro's sibling", "SIS", cells, (12,14), 3),
    }

    # -----------------------------
    # OPTIONAL: remove Cells that are not part of any Entry
    # unused_cells = {cell for row in cells for cell in row}
    # for entry in entries.values():
    #     for cell in entry.cells:
    #         unused_cells.discard(cell)
    # You can ignore unused_cells if you want black spaces implicit
    # -----------------------------

    # Create the Grid object
    grid = Grid(entries)
    return grid

register_puzzle("nyt-15x15", create_grid, title="NYT Monday puzzle 2026-01-26 (15x15)", default=True)
