# Utonia_pretrained/

The pretrained Utonia checkpoint goes here; the configs and the launcher look for `Utonia_pretrained/utonia.pth`. Not in git (CC BY-NC 4.0 from the Utonia authors): https://huggingface.co/Pointcept/Utonia

```bash
huggingface-cli download Pointcept/Utonia utonia.pth --local-dir Utonia_pretrained
```

Used for the paper's **Utonia(L)** rows (549 MB; PT-v3m3 backbone). The configs remap its keys with `CheckpointLoader(keywords="module.", replacement="module.backbone.")`. See `docs/REPRODUCING.md`, step 3.
