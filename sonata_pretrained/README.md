# sonata_pretrained/

The pretrained Sonata checkpoint goes here; the configs and the launcher look for `sonata_pretrained/pretrain-sonata-v1m1-0-base.pth`. Not in git (CC BY-NC 4.0 from the Sonata authors): https://huggingface.co/facebook/sonata

```bash
huggingface-cli download facebook/sonata pretrain-sonata-v1m1-0-base.pth --local-dir sonata_pretrained
```

Used for the paper's **Sonata(L)** rows (1.95 GB; the configs read the student backbone with `CheckpointLoader(keywords="module.student.backbone", replacement="module.backbone")`). The smaller `sonata.pth` on the same page holds the identical backbone tensors but with direct keys, so it needs the Utonia-style remapping instead. See `docs/REPRODUCING.md`, step 3.
