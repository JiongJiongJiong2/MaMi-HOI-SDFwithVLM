# ARCTIC Core Data Acquisition

Date: 2026-09-21

Status: acquired and checksum-verified on the server

## Purpose

Acquire the ARCTIC metadata and raw annotations needed by the next
external-data gate for bimanual interaction and possible handover event
verification. Full-resolution images are intentionally excluded.

## Sources

```text
project page:
https://arctic.is.tue.mpg.de

official code and download instructions:
https://github.com/zc-alexfan/arctic

source package:
arctic-1.2.zip
SHA-256
14e871bedc960d719dfd85d77b84724363bf0f868db450fc31781631722d7f16
```

The source package is code, not dataset content. Data archives were
authenticated against the official ARCTIC download service.

## Network Finding

The server timed out connecting to `download.is.tue.mpg.de` both
directly and through AutoDL academic acceleration. The authenticated
downloads were therefore staged on the local workstation and then
transferred to the server.

No credentials are stored in this repository or in the acquisition
scripts.

## Downloaded Files

| File | Bytes | SHA-256 |
|---|---:|---|
| `splits_json.zip` | 2,734 | `99af5f6759c727df07ef897db6d293cc866d465e6b3d6095579fcb62ac831b7d` |
| `raw_seqs.zip` | 225,334,351 | `3c74f8cdb5fb4f521d99132faf0471432bab5db97c7653493b01920d2ad48535` |
| `meta.zip` | 95,270,963 | `2ec627bcb8f17be33defc985a79d1dd744ee44b1f1b5732ed163ecef217c0c6e` |
| `backgrounds.zip` | 18,945,694 | `75634e0fecf184c4b8a18c9a0400443396a7e87eaf20d807e516012dfd4fe20b` |

The `splits_json`, `raw_seqs`, and `meta` hashes match
`arctic-1.2/bash/assets/checksum.json`. `backgrounds.zip` is retained
because it is part of the official minimal download set.

## Server Paths

Archives:

```text
/root/autodl-tmp/arctic_downloads
```

Extracted data:

```text
/root/autodl-tmp/arctic_data/data
```

Extracted sizes:

```text
backgrounds:  20 MB
meta:        247 MB
raw_seqs:    248 MB
splits_json:  24 KB
```

The server data disk retained approximately `5.1 GB` free after
extraction.

## Structure Audit

The raw sequence set contains:

```text
subjects:              9
sequences:           301
annotation files:  1,204
```

Each sequence has four annotation files:

```text
*.mano.npy          left/right MANO parameters
*.smplx.npy         SMPL-X body and hand parameters
*.object.npy        object pose trajectory, shape (T, 7)
*.egocam.dist.npy   egocentric camera calibration and trajectory
```

Metadata contains four files:

```text
downsamplers.npy
mano_decimator_195.npy
misc.json
object_meta.json
```

The split package contains the official `protocol_p1.json` and
`protocol_p2.json`.

## Boundary

ARCTIC raw annotations do not contain explicit contact, release, or
handover labels. Contact and role-switch events must be derived from
hand/object geometry.

SMPL-X and MANO body-model files are not yet present on the server.
They are required before mesh-based hand-object contact audit. No images
are needed for the first geometry-only gate.
