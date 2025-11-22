import nbformat

nb = nbformat.read("Rasengan-XL-QLoRA/", as_version=4)

# Remove widgets metadata if broken
if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

nbformat.write(nb, "your_notebook_fixed.ipynb")
