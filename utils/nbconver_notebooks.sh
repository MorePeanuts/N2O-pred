rm -rf ./nbconvert/*.py
find ./notebooks -name "*.ipynb" -exec jupyter nbconvert --to python --output-dir ./nbconvert {} \;