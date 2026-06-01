import argparse
import importlib
import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd


def read_lines(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_face_json(img_names, face_boxes, out_path):
    if len(img_names) != len(face_boxes):
        raise ValueError(f"img_list has {len(img_names)} rows, face_info has {len(face_boxes)} rows")

    face_info = {}
    for name, box_line in zip(img_names, face_boxes):
        values = [float(x) for x in box_line.split()[:4]]
        stem = Path(name).stem
        face_info[stem] = {"box": values}
        face_info[name] = {"box": values}

    Path(out_path).write_text(json.dumps(face_info), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Generate the official DeepfakesAdvTrack detection submission xlsx."
    )
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--data-folder", required=True, help="Folder containing imgs/, img_list.txt, face_info.txt")
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--module", default="submission_det_ensemble")
    parser.add_argument("--tta", default=os.environ.get("DFGC_TTA", "strong"), choices=["none", "fast", "strong"])
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("DFGC_BATCH_SIZE", "4")))
    args = parser.parse_args()

    data_folder = Path(args.data_folder)
    imgs_dir = data_folder / "imgs"
    img_list_path = data_folder / "img_list.txt"
    face_info_path = data_folder / "face_info.txt"
    for path in [imgs_dir, img_list_path, face_info_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    img_names = read_lines(img_list_path)
    face_boxes = read_lines(face_info_path)
    missing = [name for name in img_names if not (imgs_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images listed in img_list.txt are missing, first: {missing[0]}")

    os.environ["DFGC_TTA"] = args.tta
    os.environ["DFGC_BATCH_SIZE"] = str(args.batch_size)
    module = importlib.import_module(f"{args.module}.model")

    with tempfile.TemporaryDirectory() as tmp:
        json_path = Path(tmp) / "face_info.json"
        build_face_json(img_names, face_boxes, json_path)

        detector = module.Model()
        start = time.time()
        pred_names, pred_scores = detector.run(str(imgs_dir), str(json_path))
        elapsed = time.time() - start

    score_by_name = {name: float(score) for name, score in zip(pred_names, pred_scores)}
    predictions = []
    for name in img_names:
        if name in score_by_name:
            predictions.append(score_by_name[name])
        else:
            stem_match = next((score for pred_name, score in score_by_name.items() if Path(pred_name).stem == Path(name).stem), None)
            if stem_match is None:
                raise KeyError(f"No prediction produced for {name}")
            predictions.append(float(stem_match))

    result_path = Path(args.result_path)
    result_path.mkdir(parents=True, exist_ok=True)
    out_file = result_path / f"{args.team_name}.xlsx"
    with pd.ExcelWriter(out_file) as writer:
        pd.DataFrame({"img_names": img_names, "predictions": predictions}).to_excel(
            writer, sheet_name="predictions", index=False
        )
        pd.DataFrame({"Data Volume": [len(predictions)], "Time": [elapsed]}).to_excel(
            writer, sheet_name="time", index=False
        )

    print(f"Wrote {out_file}")
    print(f"Images: {len(predictions)}")
    print(f"Seconds: {elapsed:.3f}")


if __name__ == "__main__":
    main()
