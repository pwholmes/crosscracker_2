from __future__ import annotations
from model import Cell, Entry, Grid
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
        "1A": Entry("1A", "School stat like 3.5 or 4.0", "GPA", cells, (0,0), 3),
        "4A": Entry("4A", "Yeast, mold and mushrooms", "FUNGI", cells, (0,4), 5),
        "9A": Entry("9A", "Desert plants with needles", "CACTI", cells, (0,10), 5),
        "14A": Entry("14A", "Tree with 'American' and 'slippery' varieties", "ELM", cells, (1,0), 3),
        "15A": Entry("15A", "Alaskan native", "INUIT", cells, (1,4), 5),
        "16A": Entry("16A", "Synthetic fabric once common in sweaters", "ORLON", cells, (1,10), 5),
        "17A": Entry("17A", "Born as", "NEE", cells, (2,0), 3),
        "18A": Entry("18A", "A type of wine from southern France", "REDBORDEAUX", cells, (2,4), 11),
        "20A": Entry("20A", "Bonkers", "ZANY", cells, (3,0), 4),
        "22A": Entry("22A", "Compete (for)", "VIE", cells, (3,5), 3),
        "23A": Entry("23A", "Changes with the times", "ADAPTS", cells, (3,9), 6),
        "24A": Entry("24A", "Females that may be fleeced", "EWES", cells, (4,3), 4),
        "26A": Entry("26A", "Cinema showing", "FILM", cells, (4,8), 4),
        "27A": Entry("27A", "Canadian P.M. who is the son of another Canadian P.M.", "JUSTINTRUDEAU", cells, (5,0), 13),
        "32A": Entry("32A", "Cyber Monday sector", "ETAIL", cells, (6,0), 5),
        "33A": Entry("33A", "Put to work", "USE", cells, (6,7), 3),
        "34A": Entry("34A", "Ask God (for)", "PRAY", cells, (6,11), 4),
        "37A": Entry("37A", "Identify in a Facebook photo", "TAG", cells, (7,0), 3),
        "38A": Entry("38A", "Where canines sleep", "DOGBEDS", cells, (7,4), 7),
        "42A": Entry("42A", "'Du-u-ude!", "BRO", cells, (7,12), 3),
        "43A": Entry("43A", "Basketball's O'Neal, to fans", "SHAQ", cells, (8,0), 4),
        "45A": Entry("45A", "Keanu Reeve's character in 'The Matrix'", "NEO", cells, (8,5), 3),
        "46A": Entry("46A", "To no _____ (ineffectively)", "AVAIL", cells, (8,10), 5),
        "48A": Entry("48A", "Platform for Donkey Kong Country", "SUPERNINTENDO", cells, (9,2), 13),
        "52A": Entry("52A", "Ingredient in man hand lotions", "ALOE", cells, (10,3), 4),
        "53A": Entry("53A", "The lion's share", "MOST", cells, (10,8), 4),
        "54A": Entry("54A", "Cigarette butt receptacle", "ASHCAN", cells, (11,0), 6),
        "57A": Entry("57A", "Major N.Y.C. concert venue", "MSG", cells, (11,7), 3),
        "58A": Entry("58A", "Do the backstroke or breaststroke", "SWIM", cells, (11,11), 4),
        "62A": Entry("62A", "Chunks found in Ben & Jerry's Half Baked ice cream", "COOKIEDOUGH", cells, (12,0), 11),
        "65A": Entry("65A", "Avante-garde musical artist Yoko", "ONO", cells, (12,12), 3),
        "66A": Entry("66A", "Fixed, as a piano", "TUNED", cells, (13,0), 5),
        "67A": Entry("67A", "'The Godfather' author Puzo", "MARIO", cells, (13,6), 5),
        "68A": Entry("68A", "Frequently, to a poet", "OFT", cells, (13,12), 3),
        "69A": Entry("69A", "Winter coasters", "SLEDS", cells, (14,0), 5),
        "70A": Entry("70A", "Surgical tube", "STENT", cells, (14,6), 5),
        "71A": Entry("71A", "Cry from Homer", "DOH", cells, (14,12), 3),
        "1D": Entry("1D", "Kids these days", "GENZ", cells, (0,0), 4),
        "2D": Entry("2D", "'Not guilty', e.g.", "PLEA", cells, (0,1), 4),
        "3D": Entry("3D", "'You got that right, sister!'", "AMEN", cells, (0,2), 4),
        "4D": Entry("4D", "Certain evergreen", "FIR", cells, (0,4), 3),
        "5D": Entry("5D", "Like some bars in gymnastics", "UNEVEN", cells, (0,5), 6),
        "6D": Entry("6D", "One who is barely seen?", "NUDIST", cells, (0,6), 6),
        "7D": Entry("7D", "Jeer", "GIBE", cells, (0,7), 4),
        "8D": Entry("8D", "'How was _ __ know?'", "ITO", cells, (0,8), 3),
        "9D": Entry("9D", "Pamper", "CODDLE", cells, (0,10), 6),
        "10D": Entry("10D", "Tour guide?", "AREAMAP", cells, (0,11), 7),
        "11D": Entry("11D", "Crash of thunder", "CLAP", cells, (0,12), 4),
        "12D": Entry("12D", "Promote enthusiastically", "TOUT", cells, (0,13), 4),
        "13D": Entry("13D", "'Need You Tonight' rock band", "INXS", cells, (0,14), 4),
        "19D": Entry("19D", "Like speakeasies and refrigerators, at times", "RAIDED", cells, (2,9), 6),
        "21D": Entry("21D", "Brand of cooler that shares its name with a Himalayan legend", "YETI", cells, (3,3), 4),
        "25D": Entry("25D", "Opposite of tame", "WILD", cells, (4,4), 4),
        "26D": Entry("26D", "Join into one", "FUSE", cells, (4,8), 4),
        "27D": Entry("27D", "Jacuzzi features", "JETS", cells, (5,0), 4),
        "28D": Entry("28D", "Where to find the Jazz", "UTAH", cells, (5,1), 4),
        "29D": Entry("29D", "Long-winded tales", "SAGAS", cells, (5,2), 5),
        "30D": Entry("30D", "Apply, as oiintment", "RUBON", cells, (5,7), 5),
        "31D": Entry("31D", "City-related", "URBAN", cells, (5,12), 5),
        "35D": Entry("35D", "Desertlike", "ARID", cells, (6,13), 4),
        "36D": Entry("36D", "Carefree motto, in modern lingo", "YOLO", cells, (6,14), 4),
        "39D": Entry("39D", "Low-scoring deadlock", "ONEONE", cells, (7,5), 6),
        "40D": Entry("40D", "Richard of'Pretty Woman'", "GERE", cells, (7,6), 4),
        "41D": Entry("41D", "Hurdles for srs.", "SATS", cells, (7,10), 4),
        "44D": Entry("44D", "Sounded like a duck", "QUACKED", cells, (8,3), 7),
        "47D": Entry("47D", "Farm docs", "VETS", cells, (8,11), 4),
        "49D": Entry("49D", "Tartan patterns", "PLAIDS", cells, (9,4), 6),
        "50D": Entry("50D", "'Yeah, right'", "IMSURE", cells, (9,8), 6),
        "51D": Entry("51D", "Bean", "NOGGIN", cells, (9,9), 6),
        "54D": Entry("54D", "Does some theater work", "ACTS", cells, (11,0), 4),
        "55D": Entry("55D", "Word before food or mate", "SOUL", cells, (11,1), 4),
        "56D": Entry("56D", "Refine, as a skill", "HONE", cells, (11,2), 4),
        "57D": Entry("57D", "Castle defense", "MOAT", cells, (11,7), 4),
        "59D": Entry("59D", "What a whittler whittles", "WOOD", cells, (11,12), 4),
        "60D": Entry("60D", "Facts", "INFO", cells, (11,13), 4),
        "61D": Entry("61D", "Relative of a butterfly", "MOTH", cells, (11,14), 4),
        "63D": Entry("63D", "Private online chats, for short", "DMS", cells, (12,6), 3),
        "64D": Entry("64D", "Spicy", "HOT", cells, (12,10), 3),
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

register_puzzle("nyt-2025-09-22-15x15", create_grid, title="NYT Monday puzzle 2025-09-22 (15x15)", default=True)
