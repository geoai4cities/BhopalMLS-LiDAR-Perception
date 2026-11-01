# Concerto_pretrained/

The pretrained Concerto checkpoints go here; the configs and the launcher look for `Concerto_pretrained/concerto_large_outdoor.pth` (default) and `Concerto_pretrained/concerto_large.pth`. Not in git (CC BY-NC 4.0 from the Concerto authors): https://huggingface.co/Pointcept/Concerto

```bash
huggingface-cli download Pointcept/Concerto concerto_large.pth concerto_large_outdoor.pth --local-dir Concerto_pretrained
```

| file | paper label | launcher keyword |
|---|---|---|
| `concerto_large_outdoor.pth` (831 MB) | Concerto(O), outdoor-pretrained | `outdoor` (default) |
| `concerto_large.pth` (831 MB) | Concerto(L), general-purpose | `indoor` (run folders get the `_indoor` suffix) |

Both have direct keys and are remapped with `CheckpointLoader(keywords="module.", replacement="module.backbone.")`. See `docs/REPRODUCING.md`, step 3.
