# Externally supplied perception assets

No model, mesh, or dataset is stored in this repository. The `.gitignore` keeps all
contents of this directory private except this file.

Recommended local layout:

```text
models/
  red_cube_yoloseg_best.pt
  red_cube.obj
  foundationpose_weights/
    2023-10-28-18-33-37/
      model_best.pth
      config.yml
    2024-01-11-20-02-45/
      model_best.pth
      config.yml
```

Update a copied YAML configuration to point to the actual locations. FoundationPose
itself may expect its scorer/refiner directories beneath its checkout; use symlinks
or configuration appropriate to the pinned upstream revision rather than duplicating
large files in Git.

## Required provenance record

For every experiment, record outside the weight file:

- asset filename and role;
- source URL, Drive location, or training run identifier;
- download/export date;
- license or redistribution status;
- YOLO architecture and Ultralytics version (`8.4.138` for legacy v7);
- FoundationPose repository commit (`a1b694b83e633c2cb6115b9063d940a687759392`);
- mesh dimensions and units—the reference cube is an ideal 4 cm cube in metres;
- SHA-256 digest.

The historical clean notebook used the Hugging Face repository
`gpue/foundationpose-weights` for FoundationPose weights. Treat that as provenance
information, not permission to redistribute; verify the upstream terms yourself.
The exact red-cube YOLO training-run identifier is currently **UNKNOWN**.

## Integrity verification

On Linux or Colab:

```bash
sha256sum \
  models/red_cube_yoloseg_best.pt \
  models/red_cube.obj \
  models/foundationpose_weights/*/model_best.pth
```

Save the output with the experiment metadata, then verify later with:

```bash
sha256sum --check experiment-assets.sha256
```

Do not commit the manifest if it exposes private storage paths or proprietary asset
names. Never replace missing assets with fabricated files.
