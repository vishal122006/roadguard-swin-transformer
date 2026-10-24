## Publish the trained checkpoint for automatic Streamlit loading

The Streamlit app uses a local checkpoint first. If it is absent, it attempts to download the epoch-5 checkpoint from the public GitHub release asset URL.

**Privacy notice:** this upload will make the checkpoint publicly downloadable. Do not run it unless you are comfortable publishing the trained weights.

In Colab, mount Drive and make sure the \`GITHUB_TOKEN\` secret has **Contents: Read and write** access to \`vishal122006/roadguard-swin-transformer\`. Then run:

\`\`\`python
!wget -q https://raw.githubusercontent.com/vishal122006/roadguard-swin-transformer/main/scripts/publish_checkpoint_release_colab.py -O /content/publish_checkpoint_release_colab.py
!python /content/publish_checkpoint_release_colab.py
\`\`\`

The script defaults to \`/content/drive/MyDrive/RoadGuard/work_dirs/roadguard_swin_tiny_faster_rcnn/best_coco_bbox_mAP_epoch_5.pth\`. If needed, set \`ROADGUARD_CHECKPOINT_LOCAL_PATH\` to another absolute path. It asks you to type \`PUBLISH\` before uploading.

After the asset is uploaded, restart or redeploy Streamlit. The first detection downloads the approximately 172 MB checkpoint into a local cache; later detections reuse it. The app does not switch to YOLO or fabricate predictions if loading fails.
