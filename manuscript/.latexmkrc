$ENV{'TEXINPUTS'} = '../els-cas-templates//:' . ($ENV{'TEXINPUTS'} // '');
$ENV{'BSTINPUTS'} = '../els-cas-templates//:' . ($ENV{'BSTINPUTS'} // '');
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode %O %S';
$bibtex_use = 2;
