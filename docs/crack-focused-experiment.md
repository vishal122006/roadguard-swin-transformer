# Crack-focused RoadGuard Swin experiment

This experiment adds one unified **Crack** category (D00 + D10 + D20) and retains **Pothole** as the second class. It does not replace the deployed four-class app and does not claim training has already improved performance.

## Why do this?

The recorded five-epoch baseline has validation COCO mAP@0.50:0.95 = 9.0% and mAP@0.50 = 24.7%. At confidence 0.50, recorded subtype test recalls were D00 17.24%, D10 0%, and D20 60.57%. The earlier dataset audit counted 49 transverse-crack training boxes and 6 validation boxes. Merging crack types directly matches the goal of detecting any crack rather than also distinguishing its subtype.

## 1. Evaluate the existing model first

Run this from the deployment repository in the same compatible MMDetection environment, with the dataset and checkpoint already available. Change paths if your Colab mount uses different folders.

    python scripts/evaluate_crack_only.py \
      --config configs/roadguard_swin_infer.py \
      --checkpoint /content/drive/MyDrive/RoadGuard/work_dirs/roadguard_swin_tiny_faster_rcnn/best_coco_bbox_mAP_epoch_5.pth \
      --ann /content/RoadGuard_SwinTransformer/data/rdd2022/annotations/instances_val.json \
      --images-dir /content/RoadGuard_SwinTransformer/data/rdd2022/images \
      --device cuda:0 \
      --out /content/drive/MyDrive/RoadGuard/crack_baseline_validation.json

The script maps the model's D00/D10/D20 boxes to one Crack category for evaluation, and compares them with ground truth formed by merging the same three annotations. It produces JSON and CSV containing object-level precision, recall and F1 across confidence thresholds, interpolated AP at IoU=0.50, and image-level binary crack-present accuracy/precision/recall/specificity/F1.

Image-level accuracy answers “did this image contain any crack?”; object-level scores answer “did the model find the annotated crack with a sufficiently overlapping box?” Report both separately. Image accuracy can look high when there are many crack-negative images, so do not use it alone.

## 2. Create merged COCO annotations

The original annotations are only read; converted annotations are saved to a new directory.

    python scripts/prepare_crack_dataset.py \
      --train-ann /content/RoadGuard_SwinTransformer/data/rdd2022/annotations/instances_train.json \
      --val-ann /content/RoadGuard_SwinTransformer/data/rdd2022/annotations/instances_val.json \
      --test-ann /content/RoadGuard_SwinTransformer/data/rdd2022/annotations/instances_test.json \
      --out-dir /content/drive/MyDrive/RoadGuard/crack_binary_annotations

Inspect conversion_summary.json before training. The conversion preserves image records and boxes, maps D00/D10/D20 to category Crack and maps D40 to Pothole.

## 3. Fine-tune the two-class model

This requires a GPU and a working training environment with the original training configuration. The deployed inference config alone is not a full training config. Run from the repository root:

    python scripts/fine_tune_crack_colab.py \
      --config /content/RoadGuard_SwinTransformer/configs/roadguard_swin_tiny_faster_rcnn.py \
      --checkpoint /content/drive/MyDrive/RoadGuard/work_dirs/roadguard_swin_tiny_faster_rcnn/best_coco_bbox_mAP_epoch_5.pth \
      --data-root /content/RoadGuard_SwinTransformer/data/rdd2022 \
      --annotation-dir /content/drive/MyDrive/RoadGuard/crack_binary_annotations \
      --work-dir /content/drive/MyDrive/RoadGuard/work_dirs/roadguard_swin_crack_binary_20ep \
      --epochs 20 --lr 2e-5 --val-interval 1

The class head is initialized by averaging the D00/D10/D20 classifier and box-regression outputs for Crack, copying D40 outputs to Pothole and preserving the background classifier output. The original checkpoint stays unchanged, and this run gets a separate work directory. This transfer strategy is a controlled experiment, not a guarantee of higher accuracy.

## 4. Compare before deploying

Run crack-only evaluation on the baseline and then on the best new checkpoint, using the same validation split and thresholds. The fine-tuned model uses the class order Crack, Pothole, so use a two-class inference config derived from the training config with the same backbone, FPN, and test pipeline. Do not pass the four-class inference config to a two-class checkpoint.

After the fine-tuning completes, evaluate the best checkpoint using the matching two-class inference config and converted validation annotations:

    python scripts/evaluate_crack_only.py \
      --config configs/roadguard_swin_crack_infer.py \
      --checkpoint /content/drive/MyDrive/RoadGuard/work_dirs/roadguard_swin_crack_binary_20ep/best_coco_bbox_mAP_epoch_XX.pth \
      --model-class-mode binary \
      --ann /content/drive/MyDrive/RoadGuard/crack_binary_annotations/instances_val_crack_binary.json \
      --images-dir /content/RoadGuard_SwinTransformer/data/rdd2022/images \
      --device cuda:0 \
      --out /content/drive/MyDrive/RoadGuard/crack_binary_validation.json

Replace the checkpoint filename with the actual best checkpoint saved by training. Compare the crack-level AP, precision, recall and F1 with the baseline evaluation, using the same validation images and IoU threshold.

Keep the current public app/checkpoint unchanged until the new model passes a real inference smoke test and its validation results are better. Use validation to choose the threshold, then freeze it before evaluating test data. The project’s existing test split was inspected previously, so disclose that limitation and use a genuinely new holdout if you need a stronger final research claim.
