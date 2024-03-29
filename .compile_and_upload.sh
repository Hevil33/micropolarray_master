#!/bin/bash
pytest -v ./ # first test
#pdoc --html --force --output-dir ./docs micropolarray
cd docs/
./create_documentation.sh
cd ..
#sphinx-apidoc -o docs micropolarray/
pip-compile pyproject.toml
python3 -m build
twine upload --verbose --skip-existing dist/*