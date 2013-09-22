JAR = jar

all: ../inst/java/weka.jar

## <ARGH>
## Ensure that dynamic class discovery is off: otherwise, the Weka class
## loader is unhappy and writes 
##   [GenericPropertiesCreator]
##   classloader in use is not the system classloader:
##   using static entries in weka/gui/GenericObjectEditor.props rather
##   than dynamic class discovery
## to stdout which in turn makes everyone else unhappy ...
../inst/java/weka.jar: FORCE
	@echo -n "updating $@ ..."
	@(pfile="weka/gui/GenericPropertiesCreator.props"; \
	  $(JAR) xf $@ $${pfile} ; \
	  mv $${pfile} $${pfile}.in ; \
	  sed 's/^UseDynamic=true/UseDynamic=false/' $${pfile}.in \
	    > $${pfile}; \
	  $(JAR) uf $@ $${pfile} ; \
	  rm -rf weka)
	@echo " done"
## </ARGH>

FORCE:
