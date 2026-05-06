# Data Preparation

## Target Layout

```text
data/
  visdrone/
    train/images/
    train/annotations/
    val/images/
    val/annotations/
  aitodv2/
    images/train/
    images/val/
    annotations/
  dota/
    train/images/
    train/labelTxt/
    val/images/
    val/labelTxt/
  converted/
    visdrone/
    aitodv2/
    dota_hbb/
```

## VisDrone

Expected raw source:

- images in `data/visdrone/{train,val}/images/`
- txt annotations in `data/visdrone/{train,val}/annotations/`

Current status:

- `VisDrone2019-DET-train.zip` and `VisDrone2019-DET-val.zip` are already available in `data/visdrone/_raw/`
- train and val have already been unpacked into:
  - `data/visdrone/train/`
  - `data/visdrone/val/`
- converted COCO annotations already exist:
  - `data/converted/visdrone/annotations/train_coco.json`
  - `data/converted/visdrone/annotations/val_coco.json`

Useful utilities:

- unpack raw archives:
  - `python scripts/prepare_visdrone_raw.py --zip <zip> --split train|val|test-dev`
- download from Google Drive:
  - `python scripts/download_google_drive.py --file-id <id> --output <path>`
- official mirror used successfully for train:
  - `https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip`

Convert:

```bash
python scripts/convert_visdrone_to_coco.py \
  --images-dir data/visdrone/train/images \
  --annotations-dir data/visdrone/train/annotations \
  --output-json data/converted/visdrone/annotations/train_coco.json \
  --allow-empty

python scripts/convert_visdrone_to_coco.py \
  --images-dir data/visdrone/val/images \
  --annotations-dir data/visdrone/val/annotations \
  --output-json data/converted/visdrone/annotations/val_coco.json \
  --allow-empty
```

## AI-TOD-v2

Expected raw source:

- images in `data/aitodv2/images/{train,val}/`
- official json in `data/aitodv2/annotations/`

Current status:

- official `AI-TOD-v2` annotation json files are already downloaded into:
  - `data/aitodv2/_raw/annotations/aitodv2_train.json`
  - `data/aitodv2/_raw/annotations/aitodv2_val.json`
- current raw counts:
  - train: `11,214` images / `301,534` annotations / `8` classes
  - val: `2,804` images / `75,091` annotations / `8` classes

Useful sources:

- official page:
  - `https://chasel-tsui.github.io/AI-TOD-v2/`
- official annotations folder:
  - `https://drive.google.com/drive/folders/1Er14atDO1cBraBD4DSFODZV1x7NHO_PY?usp=sharing`
- downloaded file ids:
  - train: `1nE_v2CkukY7X-oBEK2yO2olBn28IOEpT`
  - val: `12Mtp_8hhUUVDnQZmkgN4VvOvL_ajQb9u`

Normalize:

```bash
python scripts/convert_aitod_to_coco.py \
  --input-json data/aitodv2/_raw/annotations/aitodv2_train.json \
  --output-json data/converted/aitodv2/annotations/train_coco.json

python scripts/convert_aitod_to_coco.py \
  --input-json data/aitodv2/_raw/annotations/aitodv2_val.json \
  --output-json data/converted/aitodv2/annotations/val_coco.json
```

## DOTA-HBB

Expected raw source:

- images in `data/dota/{train,val}/images/`
- labels in `data/dota/{train,val}/labelTxt/`

Convert with patching:

```bash
python scripts/convert_dota_hbb.py \
  --images-dir data/dota/train/images \
  --label-dir data/dota/train/labelTxt \
  --output-dir data/converted/dota_hbb/images/train \
  --output-json data/converted/dota_hbb/annotations/train_coco.json
```

Repeat for validation split.

## Validation

After each conversion:

```bash
python scripts/summarize_coco.py --input-json <converted_json>
```

The summary must report non-zero images, annotations, and categories.
