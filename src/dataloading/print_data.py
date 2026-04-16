import json

with open("song_data.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"{'='*60}")
print(f"DATASETT OVERSIKT")
print(f"{'='*60}")
print(f"Totalt sanger:   {len(dataset)}")
print(f"Totalt beatmaps: {sum(len(s['beatmaps']) for s in dataset)}")
print(f"{'='*60}\n")

for sang_idx, sang in enumerate(dataset):
    print(f"[{sang_idx+1}] {sang['song']}")
    print(f"    Folder: {sang['folder_path']}")
    print(f"    Beatmaps: {len(sang['beatmaps'])}")
    
    for bm_idx, bm in enumerate(sang['beatmaps']):
        # Parse timing
        bpm = None
        for tp in bm['timing_points']:
            try:
                if int(tp[6]) == 1:
                    bpm = round(60000 / float(tp[1]), 1)
                    break
            except:
                continue

        # Note-fordeling
        circles  = sum(1 for n in bm['hit_objects'] if len(n) >= 4 and int(n[3]) in [1, 5])
        sliders  = sum(1 for n in bm['hit_objects'] if len(n) >= 4 and int(n[3]) in [2, 6])
        spinners = sum(1 for n in bm['hit_objects'] if len(n) >= 4 and int(n[3]) == 12)

        print(f"\n    [{bm_idx+1}] {bm['metadata']['version']}")
        print(f"        Audio:    {bm['audio_path'].split(chr(92))[-1]}")
        print(f"        BPM:      {bpm}")
        print(f"        AR:{bm['difficulty']['ar']}  OD:{bm['difficulty']['od']}  CS:{bm['difficulty']['cs']}")
        print(f"        Notes:    {len(bm['hit_objects'])} total")
        print(f"                  Circles:{circles}  Sliders:{sliders}  Spinners:{spinners}")
        print(f"        Timing:   {len(bm['timing_points'])} timing points")
    
    print()

    if sang_idx >= 4:
        print(f"... og {len(dataset)-5} sanger til")
        break


    easy   = 0
medium = 0
hard   = 0

for sang in dataset:
    for bm in sang["beatmaps"]:
        ar = bm["difficulty"]["ar"]
        od = bm["difficulty"]["od"]
        
        if ar <= 5 and od <= 5:
            easy += 1
        elif ar <= 7 and od <= 7:
            medium += 1
        else:
            hard += 1

print(f"Easy:   {easy}")
print(f"Medium: {medium}")
print(f"Hard:   {hard}")

hard_data = []

for sang in dataset:
    hard_beatmaps = []
    for bm in sang["beatmaps"]:
        ar = bm["difficulty"]["ar"]
        od = bm["difficulty"]["od"]
        if ar >= 7 and od >= 7:
            hard_beatmaps.append(bm)
    
    if hard_beatmaps:
        hard_data.append({
            "song":        sang["song"],
            "folder_path": sang["folder_path"],
            "beatmaps":    hard_beatmaps
        })

print(f"Sanger med hard maps: {len(hard_data)}")
print(f"Totalt hard beatmaps: {sum(len(s['beatmaps']) for s in hard_data)}")

alle_hard = sum(
    1 for sang in dataset
    for bm in sang["beatmaps"]
    if bm["difficulty"]["ar"] >= 7 and bm["difficulty"]["od"] >= 7
)
print(f"Alle hard: {alle_hard}")

easy   = 0
medium = 0
hard   = 0
ukjent = 0

for sang in dataset:
    for bm in sang["beatmaps"]:
        ar = bm["difficulty"]["ar"]
        od = bm["difficulty"]["od"]
        
        if ar <= 5 and od <= 5:
            easy += 1
        elif ar <= 7 and od <= 7:
            medium += 1
        elif ar >= 7 and od >= 7:
            hard += 1
        else:
            ukjent += 1  # AR og OD peker i ulik retning

print(f"Easy:   {easy}")
print(f"Medium: {medium}")
print(f"Hard:   {hard}")
print(f"Ukjent: {ukjent}")
print(f"Totalt: {easy+medium+hard+ukjent}")

def get_difficulty(bm):
    ar = bm["difficulty"]["ar"]
    od = bm["difficulty"]["od"]
    avg = (ar + od) / 2
    
    if avg <= 5:
        return "easy"
    elif avg <= 7:
        return "medium"
    else:
        return "hard"
    
hard_data = []
for sang in dataset:
    hard_beatmaps = [
        bm for bm in sang["beatmaps"]
        if (bm["difficulty"]["ar"] + bm["difficulty"]["od"]) / 2 > 7
    ]
    if hard_beatmaps:
        hard_data.append({
            "song":        sang["song"],
            "folder_path": sang["folder_path"],
            "beatmaps":    hard_beatmaps
        })

print(f"Hard sanger:  {len(hard_data)}")
print(f"Hard beatmaps: {sum(len(s['beatmaps']) for s in hard_data)}")

for sang in hard_data[:3]:
    for bm in sang["beatmaps"]:
        print(f"{bm['metadata']['title']} - {bm['metadata']['version']}")
        print(f"  Audio: {bm['audio_path']}")
import json

with open("hard_data.json", "w", encoding="utf-8") as f:
    json.dump(hard_data, f, indent=2, ensure_ascii=False)

print(f"Lagret {len(hard_data)} sanger til hard_data.json")