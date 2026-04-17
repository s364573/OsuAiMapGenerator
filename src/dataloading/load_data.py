import os
import json
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SONGS_FOLDER = Path(r"C:/Personal Projects/osuaimap_generator/network_volume/Songs")
OUTPUT_JSON  = "song_data.json"

TRIGGER_WORDS = [
    "[General]", "[Metadata]", "[Difficulty]",
    "[TimingPoints]", "[HitObjects]"
]

# ─────────────────────────────────────────────
# PARSER - kun nødvendige seksjoner
# ─────────────────────────────────────────────
def parse_osu(filepath):
    data = {}
    current_section = None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if line in TRIGGER_WORDS:
                current_section = line
                data[current_section] = {} if current_section in ["[General]", "[Metadata]", "[Difficulty]"] else []

            elif line and current_section and not line.startswith("//"):
                if current_section in ["[General]", "[Metadata]", "[Difficulty]"]:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        data[current_section][key.strip()] = value.strip()
                else:
                    data[current_section].append(line.split(","))

    return data


# ─────────────────────────────────────────────
# BYGG DATASETT
# ─────────────────────────────────────────────
def build_dataset(songs_folder, output_json):
    songs_folder = Path(songs_folder)
    dataset      = []
    totalt_osu   = 0
    feil         = 0

    song_folders = [f for f in songs_folder.iterdir() if f.is_dir()]
    print(f"Fant {len(song_folders)} sang-mapper\n")

    for idx, folder in enumerate(song_folders):
        song_entry = {
            "song":        folder.name,
            "folder_path": str(folder),
            "beatmaps":    []
        }

        for osu_file in folder.glob("*.osu"):
            try:
                parsed = parse_osu(osu_file)

                # Hent AudioFilename
                general        = parsed.get("[General]", {})
                audio_filename = general.get("AudioFilename", "").strip()
                audio_path     = folder / audio_filename

                # Sjekk at mp3 faktisk finnes
                if not audio_path.exists():
                    feil += 1
                    continue

                # Hent difficulty info
                difficulty = parsed.get("[Difficulty]", {})
                metadata   = parsed.get("[Metadata]", {})

                beatmap = {
                    "name":       osu_file.name,
                    "osu_path":   str(osu_file),
                    "audio_path": str(audio_path),
                    "metadata": {
                        "title":   metadata.get("Title", ""),
                        "artist":  metadata.get("Artist", ""),
                        "version": metadata.get("Version", ""),
                    },
                    "difficulty": {
                        "ar":  float(difficulty.get("ApproachRate", 5)),
                        "od":  float(difficulty.get("OverallDifficulty", 5)),
                        "cs":  float(difficulty.get("CircleSize", 4)),
                        "hp":  float(difficulty.get("HPDrainRate", 5)),
                        "sv":  float(difficulty.get("SliderMultiplier", 1.4)),
                    },
                    "timing_points": parsed.get("[TimingPoints]", []),
                    "hit_objects":   parsed.get("[HitObjects]", []),
                }

                song_entry["beatmaps"].append(beatmap)
                totalt_osu += 1

            except Exception as e:
                feil += 1
                continue

        if song_entry["beatmaps"]:
            dataset.append(song_entry)

        # Progress
        print(f"\r[{idx+1}/{len(song_folders)}] {folder.name[:50]}", end="")

    print(f"\n\nFerdig!")
    print(f"Sanger:   {len(dataset)}")
    print(f"Beatmaps: {totalt_osu}")
    print(f"Feil:     {feil}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nLagret til {output_json}")
    return dataset


# ─────────────────────────────────────────────
# KJØR
# ─────────────────────────────────────────────
if __name__ == "__main__":
    dataset = build_dataset(SONGS_FOLDER, OUTPUT_JSON)

    # Vis første sang som test
    if dataset:
        sang    = dataset[0]
        beatmap = sang["beatmaps"][0]
        print(f"\n--- Test: {sang['song']} ---")
        print(f"Audio:    {beatmap['audio_path']}")
        print(f"Version:  {beatmap['metadata']['version']}")
        print(f"AR:       {beatmap['difficulty']['ar']}")
        print(f"Notes:    {len(beatmap['hit_objects'])}")
        print(f"Timing:   {beatmap['timing_points'][:2]}")