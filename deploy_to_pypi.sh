echo Installing packages
pip install -r requirements.txt

echo Delete the previous build folders
rm -rf build
rm -rf dist
rm -rf xklearn.egg-info

echo Building the library
python setup.py sdist bdist_wheel

echo Deploy to pypi.org
twine upload dist/*