
#constants -- these need to be moved into our own environmental variable soon
#VERSION = "1.0b"

#PLOTS_DIR = "output_plots"
#JAR_DEPENDENCIES = c("bart_java.jar", "commons-math-2.1.jar", "trove-3.0.3.jar", "junit-4.10.jar")

COLORS = array(NA, 500)
for (i in 1 : 500){
	COLORS[i] = rgb(runif(1, 0, 0.7), runif(1, 0, 0.7), runif(1, 0, 0.7))
}

set_bart_machine_num_cores = function(num_cores){
	assign("BART_NUM_CORES", num_cores, pkg_globals)
}

bart_machine_num_cores = function(){
	get("BART_NUM_CORES", pkg_globals)
}

init_java_for_bart_machine_with_mem_in_mb = function(bart_max_mem){
  JAR_DEPENDENCIES = get("JAR_DEPENDENCIES", pkg_globals)
	jinit_params = paste("-Xmx", bart_max_mem, "m", sep = "")
#	cat("initializing java with parameters", jinit_params, "from directory", getwd(), "\n")
	.jinit(parameters = jinit_params)
	for (dependency in JAR_DEPENDENCIES){
		.jaddClassPath(paste(find.package("bartMachine"), "/java/", dependency, sep = ""))
#		cat("  with dependency", dependency, "\n")
	} 
#	print(.jclassPath())
}
#http://r.789695.n4.nabble.com/Referencing-inst-directory-in-installed-package-td3749659.html

get_var_counts_over_chain = function(bart_machine, type = "splits"){
	if (!(type %in% c("trees", "splits"))){
		stop("type must be \"trees\" or \"splits\"")
	}
	C = t(sapply(.jcall(bart_machine$java_bart_machine, "[[I", "getCountsForAllAttribute", as.integer(BART_NUM_CORES), type), .jevalArray))
	colnames(C) = colnames(bart_machine$model_matrix_training_data)[1 : bart_machine$p]
	C
}

get_var_props_over_chain = function(bart_machine, type = "splits"){
	if (!(type %in% c("trees", "splits"))){
		stop("type must be \"trees\" or \"splits\"")
	}	
	attribute_props = .jcall(bart_machine$java_bart_machine, "[D", "getAttributeProps", as.integer(BART_NUM_CORES), type)
	names(attribute_props) = colnames(bart_machine$model_matrix_training_data)[1 : bart_machine$p]
	attribute_props
}

#
#do all generic functions here
#



sigsq_est = function(bart_machine){
	sigsqs = .jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesSigsqs")	
	sigsqs_after_burnin = sigsqs[(length(sigsqs) - bart_machine$num_iterations_after_burn_in) : length(sigsqs)]	
	mean(sigsqs_after_burnin)
}





#believe it or not... there's no standard R function for this, isn't that pathetic?
sample_mode = function(data){
	as.numeric(names(sort(-table(data)))[1])
}


#set up a logging system
LOG_DIR = "r_log"
log_file_name = "bart_log.txt"
bart_log = matrix(nrow = 0, ncol = 1)


append_to_log = function(text){
	bart_log = rbind(bart_log, paste(Sys.time(), "\t", text)) #log the time and the actual message
	assign("bart_log", bart_log, .GlobalEnv)
	write.table(bart_log, paste(LOG_DIR, "/", log_file_name, sep = ""), quote = FALSE, col.names = FALSE, row.names = FALSE)
}