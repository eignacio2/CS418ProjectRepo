# src/data_collection.py
import argparse
from pathlib import Path
import pandas as pd
from spotify_client import make_client

OUT_CSV = Path("data/raw/spotify_tracks.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "track_id","track_name","artist_id","artist_name",
    "album_id","album_name","album_release_date","release_date_precision",
    "popularity","duration_ms","explicit","source","source_id"
]

def load_existing():
    if OUT_CSV.exists():
        return pd.read_csv(OUT_CSV, dtype=str)
    return pd.DataFrame(columns=COLUMNS)

def save_merged(new_rows: pd.DataFrame):
    old = load_existing()
    merged = pd.concat([old, new_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset=["track_id"])
    merged.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(merged)} rows → {OUT_CSV} (added {len(merged)-len(old)} new)")

def get_tracks_for_artist(sp, artist_name, limit=300):
    items = []
    offset = 0
    while len(items) < limit:
        res = sp.search(q=f"artist:{artist_name}", type="track", limit=50, offset=offset)
        page = res.get("tracks", {}).get("items", [])
        if not page: break
        items.extend(page); offset += 50
    return items[:limit]

def get_tracks_for_playlist(sp, playlist_id, limit=1000):
    items, offset = [], 0
    while len(items) < limit:
        res = sp.playlist_items(playlist_id, limit=100, offset=offset)
        page = [x["track"] for x in res.get("items", []) if x.get("track")]
        if not page: break
        items.extend(page); offset += 100
        if not res.get("next"): break
    return items[:limit]

def flatten(tracks, source, source_id):
    rows = []
    for t in tracks:
        if not t or not t.get("id"): continue
        album = t.get("album") or {}
        artists = t.get("artists") or []
        a0 = artists[0] if artists else {}
        rows.append({
            "track_id": t["id"],
            "track_name": t.get("name"),
            "artist_id": a0.get("id"),
            "artist_name": a0.get("name"),
            "album_id": album.get("id"),
            "album_name": album.get("name"),
            "album_release_date": album.get("release_date"),
            "release_date_precision": album.get("release_date_precision"),
            "popularity": t.get("popularity"),
            "duration_ms": t.get("duration_ms"),
            "explicit": t.get("explicit"),
            "source": source,
            "source_id": source_id,
        })
    return pd.DataFrame(rows, columns=COLUMNS)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artists", type=str, help="Comma-separated artist names")
    parser.add_argument("--playlist", type=str, help="Playlist ID")
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    sp = make_client()
    all_rows = []

    if args.artists:
        for name in [a.strip() for a in args.artists.split(",") if a.strip()]:
            tracks = get_tracks_for_artist(sp, name, args.limit)
            all_rows.append(flatten(tracks, source="artist", source_id=name))

    if args.playlist:
        tracks = get_tracks_for_playlist(sp, args.playlist, args.limit)
        all_rows.append(flatten(tracks, source="playlist", source_id=args.playlist))

    if not all_rows:
        print("Nothing to do. Provide --artists or --playlist."); return

    save_merged(pd.concat(all_rows, ignore_index=True).dropna(subset=["track_id"]))

if __name__ == "__main__":
    main()
