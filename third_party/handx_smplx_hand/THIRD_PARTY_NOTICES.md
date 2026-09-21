# Third-Party Notices

HandX includes or interacts with third-party datasets, model assets, and code.
Each third-party component is governed by its own license or terms. This file
provides attribution and pointers to the applicable third-party licenses; it
does not replace those licenses.

The repository-level license for HandX is provided in [`LICENSE`](LICENSE).

## Redistributed Dataset Sources

HandX may redistribute MANO-format hand pose parameters derived from the
following source datasets. Source attribution is recorded in
`data/handx/source_metadata.json` when provided with the data archive.

- **GigaHands**: Licensed under CC BY-NC 4.0. See
  [`licenses/CC-BY-NC-4.0.txt`](licenses/CC-BY-NC-4.0.txt) and
  https://github.com/brown-ivl/GigaHands.
- **HOT3D**: Governed by the HOT3D Dataset License Agreement. HOT3D hand data is
  licensed under CC BY-NC-SA. See
  [`licenses/HOT3D_Dataset_License_Agreement.pdf`](licenses/HOT3D_Dataset_License_Agreement.pdf)
  and https://www.projectaria.com/datasets/hot3d/license/.
- **HoloAssist**: Licensed under the Community Data License Agreement -
  Permissive, Version 2.0. See
  [`licenses/CDLA-Permissive-2.0.txt`](licenses/CDLA-Permissive-2.0.txt) and
  https://cdla.dev/permissive-2-0/.

## Source Datasets Not Redistributed by HandX

ARCTIC and H2O data are not included in the HandX data archive due to their
redistribution policies. HandX provides processing code for users who obtain
these datasets separately and agree to their original terms.

- **ARCTIC**: ARCTIC Data & Software Copyright License for non-commercial
  scientific research purposes. See https://arctic.is.tue.mpg.de/license.html.
- **H2O**: H2O Dataset Terms of Use. See https://h2odataset.ethz.ch/.

## Third-Party Model Assets

The MANO parameters used by HandX are derived from the MANO model. MANO model
files and MANO-derived parameters are subject to the MANO license terms. See
https://mano.is.tue.mpg.de.

## References

Please cite HandX and the applicable source datasets when using data derived
from them.

```bibtex
@inproceedings{zhang2026handx,
    title     = {HandX: Scaling Bimanual Motion and Interaction Generation},
    author    = {Zhang, Zimu and Zhang, Yucheng and Xu, Xiyan and Wang, Ziyin and Xu, Sirui and Zhou, Kai and Zhou, Bing and Guo, Chuan and Wang, Jian and Wang, Yu-Xiong and Gui, Liang-Yan},
    booktitle = {CVPR},
    year      = {2026},
}

@inproceedings{fu2025gigahands,
    title     = {{GigaHands}: A Massive Annotated Dataset of Bimanual Hand Activities},
    author    = {Fu, Rao and Zhang, Dingxi and Jiang, Alex and Fu, Wanjia and Funk, Austin and Ritchie, Daniel and Sridhar, Srinath},
    booktitle = {CVPR},
    year      = {2025},
}

@inproceedings{banerjee2025hot3d,
    title     = {{HOT3D}: Hand and Object Tracking in {3D} from Egocentric Multi-View Videos},
    author    = {Banerjee, Prithviraj and Shkodrani, Sindi and Moulon, Pierre and Hampali, Shreyas and Han, Shangchen and Zhang, Fan and Zhang, Linguang and Fountain, Jade and Miller, Edward and Basol, Selen and others},
    booktitle = {CVPR},
    year      = {2025},
}

@inproceedings{wang2023holoassist,
    title     = {{HoloAssist}: An Egocentric Human Interaction Dataset for Interactive {AI} Assistants in the Real World},
    author    = {Wang, Xin and Kwon, Taein and Rad, Mahdi and Pan, Bowen and Chakraborty, Ishani and Andrist, Sean and Bohus, Dan and Feniello, Ashley and Tekin, Bugra and Frujeri, Felipe Vieira and others},
    booktitle = {ICCV},
    year      = {2023},
}

@inproceedings{fan2023arctic,
    title     = {{ARCTIC}: A Dataset for Dexterous Bimanual Hand-Object Manipulation},
    author    = {Fan, Zicong and Taheri, Omid and Tzionas, Dimitrios and Kocabas, Muhammed and Kaufmann, Manuel and Black, Michael J. and Hilliges, Otmar},
    booktitle = {CVPR},
    year      = {2023},
}

@inproceedings{kwon2021h2o,
    title     = {{H2O}: Two Hands Manipulating Objects for First Person Interaction Recognition},
    author    = {Kwon, Taein and Tekin, Bugra and St{\"u}hmer, Jan and Bogo, Federica and Pollefeys, Marc},
    booktitle = {ICCV},
    year      = {2021},
}
```

