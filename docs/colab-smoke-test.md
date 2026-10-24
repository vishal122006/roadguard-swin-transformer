## Test the actual checkpoint in Colab and launch Streamlit

This test uses the existing local Drive checkpoint and one real image from the RDD2022 **validation** split. It does not publish the weights or use the held-out test set.

After syncing the repository in Colab, run the smoke test with the same Python environment used for training:

\`\`\`bash
/content/roadguard_py311/bin/python scripts/smoke_test_checkpoint_colab.py
\`\`\`

It saves an annotated test image and a JSON record to \`/content/drive/MyDrive/RoadGuard/roadguard_swin_smoke_test.jpg\` and \`roadguard_swin_smoke_test.json\`. This is a visual smoke test, not a new accuracy result.

To launch the UI in the same Colab runtime, set \`ROADGUARD_SWIN_CHECKPOINT\` to the checkpoint path and run \`streamlit run app.py\` with the same environment. Colab's \`output.serve_kernel_port_as_window(8501)\` can display the running app in a browser tab. This temporary runtime is not a persistent public deployment.
