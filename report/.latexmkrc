# report.tex uses fontspec, so build with XeLaTeX even when an editor passes -pdf.
$pdf_mode = 5;
$pdflatex = 'xelatex %O %S';
