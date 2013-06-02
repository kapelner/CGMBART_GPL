#libraries and dependencies
tryCatch(library(rJava), error = function(e){install.packages("rJava")}, finally = library(rJava))


#
#if (.Platform$OS.type == "windows"){
#	tryCatch(library(rJava), error = function(e){install.packages("rJava")}, finally = library(rJava))
#	tryCatch(library(BayesTree), error = function(e){install.packages("BayesTree")}, finally = library(BayesTree))
#} else {
#	tryCatch(library(rJava), error = function(e){library(rJava, lib.loc = "~/R/")})	
#	library(BayesTree, lib.loc = "~/R/")
#}

#constants
VERSION = "1.0b"
BART_MAX_MEM_MB = 8000
PLOTS_DIR = "output_plots"
JAR_DEPENDENCIES = c("bart_java.jar", "commons-math-2.1.jar", "jai_codec.jar", "jai_core.jar", "trove-3.0.3.jar")
DEFAULT_ALPHA = 0.95
DEFAULT_BETA = 2
DEFAULT_K = 2
DEFAULT_Q = 0.9
DEFAULT_NU = 3.0
DEFAULT_PROB_STEPS = c(2.5, 2.5, 4) / 9
DEFAULT_PROB_RULE_CLASS = 0.5
COLORS = array(NA, 500)
for (i in 1 : 500){
	COLORS[i] = rgb(runif(1, 0, 0.7), runif(1, 0, 0.7), runif(1, 0, 0.7))
}

set_bart_machine_max_mem = function(max_mem_mbs){
	assign("BART_MAX_MEM_MB", max_mem_mbs, ".GlobalEnv")
}

BART_NUM_CORES = 1

set_bart_machine_num_cores = function(num_cores){
	assign("BART_NUM_CORES", num_cores, ".GlobalEnv")
}

bart_machine_num_cores = function(){
	BART_NUM_CORES
}

init_java_for_bart = function(){
	jinit_params = paste("-Xmx", BART_MAX_MEM_MB, "m", sep = "")
	.jinit(parameters = jinit_params)
	for (dependency in JAR_DEPENDENCIES){
		.jaddClassPath(dependency)
	}	
}


size_of_bart_chisq_cache_inquire = function(){
	init_java_for_bart()
	num_bytes = .jcall("CGM_BART.StatToolbox", "I", "numBytesForInvChisqCache")
	cat(paste(round(num_bytes / 1000000, 2), "MB\n"))
	invisible(num_bytes)
}

delete_bart_chisq_cache = function(){
	init_java_for_bart()
	.jcall("CGM_BART.StatToolbox", "V", "clearInvChisqHash")
}

get_var_counts_over_chain = function(bart_machine, type = "splits"){
	C = t(sapply(.jcall(bart_machine$java_bart_machine, "[[I", "getCountsForAllAttribute", as.integer(BART_NUM_CORES), type), .jevalArray))
	colnames(C) = colnames(bart_machine$model_matrix_training_data)[1 : bart_machine$p]
	C
}

get_var_props_over_chain = function(bart_machine, type = "splits"){
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