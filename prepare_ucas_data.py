import argparse
import shutil
import zipfile
from pathlib import Path


def has_official_files(folder):
    folder = Path(folder)
    return (folder / "imgs").is_dir() and (folder / "img_list.txt").is_file() and (folder / "face_info.txt").is_file()


def find_official_folder(root):
    root = Path(root)
    if has_official_files(root):
        return root
    matches = []
    for child in root.rglob("*"):
        if child.is_dir() and has_official_files(child):
            matches.append(child)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one official data folder under {root}, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_root = Path(args.out_root)
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)

    target_name = zip_path.stem
    if target_name.startswith("UCAS_AIAS-"):
        target_name = target_name[len("UCAS_AIAS-"):]
    elif target_name.startswith("UCAS_AISA-"):
        target_name = target_name[len("UCAS_AISA-"):]

    out_dir = out_root / target_name
    if out_dir.exists() and has_official_files(find_official_folder(out_dir)) and not args.force:
        print(find_official_folder(out_dir))
        return

    print(f"Checking CRC: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Bad zip member: {bad}")

    tmp_dir = out_root / f".{target_name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting to: {tmp_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)

    official = find_official_folder(tmp_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(official), str(out_dir))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(out_dir)


if __name__ == "__main__":
    main()
