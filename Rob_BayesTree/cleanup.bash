#!/bin/bash

## cleanup
cd /home/alex/workspace/rpart/src/
rm *.o
rm *.so
cd /home/alex/workspace/rpart/src/
rm -r tree.Rcheck
rm -r rpart.Rcheck
rm *.tar.gz
rm -r /home/alex/R/rpart/
cd workspace

## now build new
R CMD check /home/alex/workspace/rpart/
#sudo R CMD INSTALL -l /home/alex/R /home/alex/workspace/rpart/rpart_3.1-51.tar.gz

R CMD INSTALL -l /home/alex/R --build pkg /home/alex/workspace/rpart
 	

#fire up an R session
R --interactive
library(rpart, lib.loc="/home/alex/R/")


