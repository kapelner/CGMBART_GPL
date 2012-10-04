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
 	

#fire up an R session
R --interactive
library(BayesTree, lib.loc="~/R/")


