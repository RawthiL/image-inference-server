# PP-OCR backend (`family: "ppocr"`, `task: "ocr"`)

Serves [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) PP-OCR text
detection + recognition (v3 to v6 det + rec pairs) on the OCR endpoint
(`POST /v1/images:annotate`, Google Cloud Vision-compatible).

OCR is two Triton models: a DB **detector** that finds text regions and a
CTC **recognizer** that reads each cropped region. The gateway calls the
detector, crops the regions, and then calls the recognizer in batches. The
region-only mode (`TEXT_REGION_DETECTION`) skips the recognizer.

## License

| Component | License |
|-----------|---------|
| This folder's code (`src/`, `export.py`, `triton_repo.py`) | MIT |
| PaddleOCR PP-OCR pretrained weights | Apache-2.0 ✅ |

ONNX conversions of the official weights (paddle2onnx, RapidOCR) keep the
Apache-2.0 license. Fine-tuned weights carry whatever license their trainer
gives them.

## Model contract

| | Detector (`<name>-det`, `stage: "det"`) | Recognizer (`<name>-rec`, `stage: "rec"`) |
|-|-|-|
| Input | `x` `[1, 3, H, W]` FP32, H and W multiples of 32 | `x` `[N, 3, imgh, W]` FP32 |
| Pre-processing | Resize so the long side is at most `limit_side_len` (`limit_type: max`, or the short side at least it with `min`), snap to /32, cap at `max_side_limit`. `(v/255 - mean) / std` in `channel_order` (BGR, as PaddleOCR reads images with OpenCV) | Perspective-crop each region (tall regions, h/w ≥ 1.5, are read top to bottom), resize to height `imgh` keeping aspect, `(v/255 - 0.5) / 0.5`, right-pad with zeros. Crops are sorted by aspect ratio and batched (`batch`, capped by Triton `max_batch_size`) |
| Output | `[1, 1, H, W]` text probability map | `[N, T, len(charset) + 1]` per-step probabilities |
| Post-processing | DB: binarize at `thresh`, connected components → minimum-area rectangles, score (mean probability) ≥ `box_thresh`, unclip by `unclip_ratio`, map back to pixels | Greedy CTC: class 0 is blank, class `k` is `charset[k-1]`. Confidence is the mean probability of kept characters. Glyph positions come from CTC steps |
| Extra metadata | `companions: ["<name>-rec"]`, `limit_side_len`, `limit_type`, `max_side_limit`, `mean`, `std`, `channel_order`, `thresh`, `box_thresh`, `unclip_ratio`, `max_candidates` | `charset`, `imgh`, `imgw`, `max_width`, `batch`, `channel_order` |

The configured OCR model is the detector. The gateway loads the recognizer
from the detector's `companions`.

Not included: the text direction (0°/180°) classifier. Upside-down text and
text that reads bottom to top are detected, but usually not recognized.

## Recommended model: PP-OCRv6

PaddlePaddle publishes official ONNX exports of
[PP-OCRv6](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6) in three
tiers (Apache-2.0). They work with this backend as is.

| Tier | Det + rec params | Det Hmean / rec accuracy (model card) | Use when |
|------|------------------|---------------------------------------|----------|
| **medium** | 34.5M | 86.2 / 83.2 | **Default on a Triton GPU.** Best accuracy, still small for a GPU |
| small | ~8M | 84.1 / 81.3 | CPU-only Triton, or maximum throughput per GPU |
| tiny | 1.5M | 80.6 / 73.5 | Edge only; its recognizer has a much smaller character set (6,904 vs 18,708) |

Mixing tiers works (e.g. small det + medium rec): the two models are independent.

```bash
mkdir -p models/pp-ocrv6 && cd models/pp-ocrv6
for m in medium_det medium_rec; do
  curl -fL -o $m.onnx "https://huggingface.co/PaddlePaddle/PP-OCRv6_${m}_onnx/resolve/main/inference.onnx"
done
curl -fL -o medium_rec.yml "https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/main/inference.yml"
cd -
uv run --extra onnx --with pyyaml python backends/ppocr/export.py \
    --det models/pp-ocrv6/medium_det.onnx --rec models/pp-ocrv6/medium_rec.onnx \
    --dict models/pp-ocrv6/medium_rec.yml --name ppocr-v6-medium
# endpoints: { ocr: { model: ppocr-v6-medium-det, api_keys: [...] } }
```

The v6 recognizers don't embed their dictionary, so `--dict` points at the
recognizer's `inference.yml`.

Detector resize: the exporter's default (`--limit-type max --limit-side-len
960`) downscales large images, which bounds the GPU cost of any request.
PaddleOCR's own v6 pipeline uses `--limit-type min --limit-side-len 64`
(no downscaling below 4000 px). That helps with very small print on large
scans, but a 3000 px page then costs about 5x more detector compute.

## Export

`export.py` takes ONNX files, so it needs no paddle install:

```bash
uv run --extra onnx python backends/ppocr/export.py \
    --det det.onnx --rec rec.onnx --dict ppocr_keys_v1.txt --name ppocr
uv run --with grpcio-tools --with protobuf --with onnx python tools/verify_repo.py
```

This writes `ppocr-det` and `ppocr-rec`. Then configure:

```yaml
endpoints:
  ocr: { model: ppocr-det }
```

Where to get the ONNX files:

- Already converted: the RapidOCR project publishes PP-OCR ONNX files, e.g.
  `PP-OCRv4/ch_PP-OCRv4_det_infer.onnx` and `ch_PP-OCRv4_rec_infer.onnx` on
  Hugging Face (`SWHL/RapidOCR`). Their recognizers embed the character
  dictionary, so `--dict` can be omitted.
- From PaddleOCR inference models, with paddle2onnx:
  `paddle2onnx --model_dir <inference dir> --model_filename inference.json
  --params_filename inference.pdiparams --save_file det.onnx`
  (`inference.pdmodel` for older models). For PP-OCRv5, pass the
  recognizer's `inference.yml` as `--dict` (needs `--with pyyaml`).

`--dict` is one character per line, in class order. PaddleOCR appends a space
character (`use_space_char`). Pass `--no-space-char` if the recognizer was
trained without it. The exporter checks the recognizer's class count against
the dictionary. Detector tuning (`--limit-side-len`, `--limit-type`,
`--thresh`, `--box-thresh`, `--unclip-ratio`) and recognizer input
(`--rec-imgh`, `--rec-imgw`) go into the metadata.
