# Delta-Learning for Event Detection

This is the source code for paper "Distilling Discrimination and Generalization Knowledge for Event Detection via Delta-Representation Learning" in ACL 2019.

## Requirements

* Python 2.7
* PyTorch >= 0.4.0
* six
* nltk
* h5py (for pre-computed elmo representation)

## Usage

Train and test the model
* python train_event_detector.py
* python eval_event_detector.py

Hyper-parameters in our paper are saved in option file "base/option.py" and running script "scripts/run_ace2005.sh" or "scripts/run_kbp2017.sh".

## Citation
If this repository helps you, please cite this paper:
* Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun. *Distilling Discrimination and Generalization Knowledge for Event Detection via Delta-Representation Learning*. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.

```
@InProceedings{lu-etal:2019:ACL2019Delta,
  author    = {Lu, Yaojie and Lin, Hongyu and Han, Xianpei and Sun, Le},
  title     = {Distilling Discrimination and Generalization Knowledge for Event Detection via Delta-Representation Learning},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  month     = {July},
  year      = {2019},
  location  = {Florence, Italy},
  publisher = {Association for Computational Linguistics},
  pages     = {4366--4376},
  url       = {https://www.aclweb.org/anthology/P19-1429}
}
```

## Contact
If you have any question or want to request for the preprocessed data (only if you have the license from LDC) and trained models, please contact me by
* yaojie2017@iscas.ac.cn
