# Decoding language spatial relations to 2D spatial arrangements

This repository is the official implementation of "Decoding language spatial relations to 2D spatial arrangements" [Gorjan Radevski](http://gorjanradevski.github.io/), [Guillem Collel](https://sites.google.com/site/guillemct1/), [Tinne Tuytelaars](https://homes.esat.kuleuven.be/~tuytelaa/), [Marie-Francine Moens](https://people.cs.kuleuven.be/~sien.moens/) published at [Findings of EMNLP 2020](https://www.aclweb.org/anthology/volumes/2020.findings-emnlp/).

## Requirements

If you are using [Poetry](https://python-poetry.org/), navigating to the project root directory and running `poetry install` will suffice. Otherwise, a `requirements.txt` file is present so you can install all dependencies by running `pip install -r requirements.txt`. However, if you just want to download the trained models or dataset splits, make sure to have [gdown](https://github.com/wkentaro/gdown) installed. If the project dependencies are installed then `gdown` is already present. Otherwise, run `pip install gdown` to install it.

## Reference

If you found this code useful, or use some of our resources for your work, we will appreciate if you cite our paper.

```tex
@inproceedings{radevski-etal-2020-decoding,
    title = "Decoding language spatial relations to 2{D} spatial arrangements",
    author = "Radevski, Gorjan  and
      Collell, Guillem  and
      Moens, Marie-Francine  and
      Tuytelaars, Tinne",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.408",
    pages = "4549--4560",
    abstract = "We address the problem of multimodal spatial understanding by decoding a set of language-expressed spatial relations to a set of 2D spatial arrangements in a multi-object and multi-relationship setting. We frame the task as arranging a scene of clip-arts given a textual description. We propose a simple and effective model architecture Spatial-Reasoning Bert (SR-Bert), trained to decode text to 2D spatial arrangements in a non-autoregressive manner. SR-Bert can decode both explicit and implicit language to 2D spatial arrangements, generalizes to out-of-sample data to a reasonable extent and can generate complete abstract scenes if paired with a clip-arts predictor. Finally, we qualitatively evaluate our method with a user study, validating that our generated spatial arrangements align with human expectation.",
}
```

## License

Everything is licensed under the [MIT License](https://opensource.org/licenses/MIT).
