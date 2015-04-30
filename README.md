# paragraph_vector
Paragraph Vector code


Media Wikiをローカルのmwikiに入っていることが必要です。

[usage]

    python make_pv.py --corpus=wiki

[Cython version]

    cython -a inner_paragraph_vector.pyx

    python setup.py build_ext --inplace

