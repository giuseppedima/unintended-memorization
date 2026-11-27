"""Fixed vocabulary, synthetic project identifiers, and exposure groups."""

# 32 lowercase English words (verified single-token under Qwen2.5 BPE tokenizer)
VOCABULARY = [
    "maple", "orbit", "lantern", "velvet",
    "quartz", "amber", "cedar", "coral",
    "ivory", "linen", "magnet", "marble",
    "nickel", "olive", "prism", "slate",
    "violet", "walnut", "zenith", "anchor",
    "canyon", "harbor", "timber", "summit",
    "cider", "kayak", "quilt", "saddle",
    "flute", "kettle", "robin", "scarf",
]

# 24 orthogonal synthetic project names and 3-digit codes
PROJECT_NAMES = [
    "Velora", "Kaelen", "Solvex", "Myrden",
    "Zorath", "Tolvir", "Braxon", "Nymvor",
    "Drelik", "Fendur", "Golmar", "Hextin",
    "Jorvas", "Lyrith", "Morvek", "Odrane",
    "Peltus", "Quorid", "Raskel", "Syldra",
    "Thalok", "Umbriz", "Wryndel", "Zevron",
]

PROJECT_CODES = [
    "142", "279", "318", "465", "523", "684",
    "719", "836", "947", "158", "263", "394",
    "417", "582", "639", "751", "824", "916",
    "183", "295", "347", "489", "574", "628",
]

# (group_id, role, occurrences_per_epoch) — 6 canaries per group (24 total)
GROUPS = [
    (0, "Control (Unseen)", 0),
    (1, "Low Exposure", 1),
    (2, "Medium Exposure", 5),
    (3, "High Exposure", 20),
]

CANARIES_PER_GROUP = 6
SECRET_LENGTH = 4
