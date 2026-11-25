# CPE342_TermProject_Report.tex uses fontspec, so always build with XeLaTeX cleanly.
use File::Spec;
$pdf_mode = 5;
$pdflatex = 'xelatex %O %S';
$xelatex = 'xelatex -interaction=nonstopmode -halt-on-error %O %S';
$out_dir = 'output/pdf';
$aux_dir = File::Spec->catdir($ENV{'TMPDIR'} // '/tmp', 'cpe342-report-aux');
