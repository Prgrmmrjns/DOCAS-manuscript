# TeX Live already ships els-cas-templates; keep local BST (model1-num-names.bst).
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error %O %S';
$bibtex_use = 2;
# Do not treat unresolved refs/cites on intermediate passes as a hard stop;
# latexmk still reruns until the .aux/.bbl stabilize.
$force_mode = 0;
