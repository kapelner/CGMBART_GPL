#!/bin/bash

## cleanup
cd ~/workspace/CGMBART_GPL/Rob_BayesTree/src/
rm *.o
rm *.so
rm *.tar.gz
cd ~/R
rm -r BayesTree
rm BayesTree*

## now build new
cd ~/workspace/CGMBART_GPL/Rob_BayesTree
#R CMD check ~/workspace/CGMBART_GPL/Rob_BayesTree
R CMD INSTALL -l ~/R --build pkg ~/workspace/CGMBART_GPL/Rob_BayesTree
#make sure rJava is downloaded and unzipped first
#R CMD INSTALL -l ~/R --build pkg rJava	

#fire up an R session
cd ~/workspace/CGMBART_GPL/
R --interactive
library(BayesTree, lib.loc="~/R/")


