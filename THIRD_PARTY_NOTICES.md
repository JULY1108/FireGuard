# Licensing and Third-Party Notices

Updated: 2026-10-16

## Scope of this notice

The MIT notice in [LICENSE](LICENSE) is retained for code supplied under the upstream FireGuard project's MIT terms. It is not a blanket MIT grant for all artifacts and dependencies in this repository.

This document clarifies existing third-party terms. It does not independently relicense the entire repository, claim new ownership, or grant an Ultralytics Enterprise License. The original upstream MIT notice remains preserved below.

## Ultralytics and the published checkpoint

This project uses Ultralytics YOLO26 and publishes:

- `checkpoints/fireguard_s_local_v1/final/best.pt`
- Run: `fireguard_s_local_v1`
- Initialization: COCO-pretrained `yolo26s.pt`
- Fine-tuning classes: fire and smoke

Ultralytics states that its software and trained/fine-tuned YOLO models are covered by AGPL-3.0 by default and offers a separate Enterprise License. See its [licensing page](https://www.ultralytics.com/license) and [license text](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

The checkpoint is not offered under MIT. Its public availability does not remove applicable AGPL obligations. This repository does not grant an Ultralytics Enterprise License or claim that one has been purchased. An Enterprise License, if obtained separately, is governed by its own scope and terms.

When redistributing or deploying covered works, follow the applicable AGPL terms, including corresponding-source requirements and the network-interaction provision where applicable. Commercial activity is not prohibited by AGPL itself; proprietary deployment without complying with applicable AGPL obligations requires appropriate separate permission.

## Upstream FireGuard code: retained MIT notice

Source: [ParsaVictor/FireGuuard_Fire_detection_smoke](https://github.com/ParsaVictor/FireGuuard_Fire_detection_smoke).

The following is the upstream MIT notice, preserved verbatim for code incorporated from that project. It is not a blanket MIT grant for this repository's dependencies, trained weights, datasets, or media.

```text
MIT License

Copyright (c) 2026 FireGuard contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Datasets and video media

Training uses FASDD_CV and D-Fire. Each dataset, its annotations, and any underlying third-party images remain subject to their applicable source licenses and terms. Dataset files are not redistributed in this repository.

The demo videos and their source footage are not relicensed by the software license. This repository does not claim ownership of, or grant additional rights over, underlying third-party footage. Obtain the necessary source permissions before reusing or redistributing that footage.

## Other dependencies

Python packages and other third-party components retain their own licenses. The project's AGPL distribution does not replace their notices or grant rights beyond those licenses.

The previous list of unrelated Hugging Face fire-detection weights has been removed because those weights are not the checkpoint published here.
