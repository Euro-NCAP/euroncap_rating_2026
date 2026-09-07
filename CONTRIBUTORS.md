# Contributors

`euroncap-rating-2026` is developed for Euro NCAP under the Apache License 2.0
(see [LICENSE](LICENSE)).

## Organisations

- **Euro NCAP IVZW** — owner of the Euro NCAP 2026 assessment protocols, the
  master input templates in `data/`, and the copyright on this implementation.
- **IVEX NV** (<https://ivex.ai>) — design, implementation and maintenance of
  the rating calculator.

## Contributing

- Report bugs or request features by
  [opening an issue](https://github.com/Euro-NCAP/euroncap_rating_2026/issues/new/choose).
- Code is formatted with black (see `.pre-commit-config.yaml`); run
  `pre-commit run --all-files` before opening a pull request.
- Run `python -m unittest discover -s tests` before submitting; CI runs the
  suite, `black --check` and a `generate-template` smoke test for all five
  domains on Linux and Windows for Python 3.10–3.13.
