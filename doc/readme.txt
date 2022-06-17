To write the autodoc files, in a terminal run:
sphinx-apidoc -f -o source/ ../autotst/

To generate the html files, in a terminal:
make html && firefox ./build/html/index.html

To update the online documentation:
Copy the content of build/html/ to the root of the auto-tst repository when on the gh-pages branch.
