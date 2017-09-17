test:
	Rscript -e 'library(covr); codecov(path = ".")'
coverage:
	Rscript -e 'library(covr); package_coverage(path = ".")'
build-site:
	Rscript -e 'library(pkgdown); build_site(pkg = ".", path = "docs/")'
	