from __future__ import annotations
from ..model import Cell, Entry, Grid
from .registry import register_puzzle

def create_grid() -> Grid:
    # Fully packed 9x9 grid of Cells
    cells = [[Cell(row=r, col=c) for c in range(9)] for r in range(9)]

    # -----------------------------
    # Define your Entries here:
    # Example:
    # entries: dict[str, Entry] = {
    #     "1A": Entry("1A", "Bar that connecta rotating wheels", "AXLE", cells, (0,0), 4),
    #     ...
    # }
    # -----------------------------
    entries: dict[str, Entry] = {
        "1A": Entry("1A", "Bar that connects rotating wheels", "AXLE", cells, (0,0), 4),
        "5A": Entry("5A", "'It's soooo cold!'", "BRR", cells, (0,5), 3),
        "8A": Entry("8A", "Back of a shoe", "HEEL", cells, (1,0), 4),
        "9A": Entry("9A", "Not imaginary", "REAL", cells, (1,5), 4),
        "11A": Entry("11A", "Italian appetizer course", "ANTIPASTO", cells, (2,0), 9),
        "13A": Entry("13A", "Moisten, as a roasting turkey", "BASTE", cells, (3,0), 5),
        "14A": Entry("14A", "Small amount of energy", "ERG", cells, (3,6), 3),
        "15A": Entry("15A", "Cozy room", "DEN", cells, (4,2), 3),
        "16A": Entry("16A", "___ Tai (tropical drink)", "MAI", cells, (4,6), 3),
        "17A": Entry("17A", "Agency that assigns 9-digit IDs: Abbr.", "SSA", cells, (5,0), 3),
        "19A": Entry("19A", "Include surreptitiously in an email", "BCC", cells, (5,6), 3),
        "20A": Entry("20A", "Discount event under canvas", "TENTSALE", cells, (6,0), 8),
        "24A": Entry("24A", "Gets", "RECEIVES", cells, (7,0), 8),
        "25A": Entry("25A", "Prefix with -gon or -gram", "PENTA", cells, (8,1), 5),
        "1D": Entry("1D", "Character with a whale of an obsession?", "AHAB", cells, (0,0), 4),
        "2D": Entry("2D", "'Warrior princess' of 1990s TV", "XENA", cells, (0,1), 4),
        "3D": Entry("3D", "1983 David Bowie album and single", "LETSDANCE", cells, (0,2), 9),
        "4D": Entry("4D", "Of an upper echelon", "ELITE", cells, (0,3), 5),
        "5D": Entry("5D", "Lingerie top", "BRA", cells, (0,5), 3),
        "6D": Entry("6D", "Look like", "RESEMBLE", cells, (0,6), 8),
        "7D": Entry("7D", "Competitive struggles", "RATRACES", cells, (0,7), 8),
        "10D": Entry("10D", "Deductive reasoning", "LOGIC", cells, (1,8), 5),
        "12D": Entry("12D", "Weapon for a wordsmith?", "PEN", cells, (2,4), 3),
        "17D": Entry("17D", "D&D stat for a fighter: Abbr.", "STR", cells, (5,0), 3),
        "18D": Entry("18D", "Ooze (into)", "SEEP", cells, (5,1), 4),
        "21D": Entry("21D", "Finger count", "TEN", cells, (6,3), 3),
        "22D": Entry("22D", "Take a load off", "SIT", cells, (6,4), 3),
        "23D": Entry("23D", "Film director DuVernay", "AVA", cells, (6,5), 3)
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

register_puzzle("simple-9x9", create_grid, title="simple 9x9", default=True)
