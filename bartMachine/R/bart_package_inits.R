##package constants
VERSION = "1.0b"
JAR_DEPENDENCIES = c("bart_java.jar", "commons-math-2.1.jar", "trove-3.0.3.jar", "junit-4.10.jar")

##color array 
COLORS = array(NA, 500)
for (i in 1 : 500){
	COLORS[i] = rgb(runif(1, 0, 0.7), runif(1, 0, 0.7), runif(1, 0, 0.7))
}

##set number of cores in use
set_bart_machine_num_cores = function(num_cores){
	assign("BART_NUM_CORES", num_cores, bartMachine_globals)
}

##get number of cores in use
bart_machine_num_cores = function(){
	if (exists("BART_NUM_CORES", envir = bartMachine_globals)){
		get("BART_NUM_CORES", bartMachine_globals)
	} else {
		stop("Number of cores not set yet. Please use \"set_bart_machine_num_cores.\"")
	}
}

##initialize JVM
init_java_for_bart_machine_with_mem_in_mb = function(bart_max_mem){
	jinit_params = paste("-Xmx", bart_max_mem, "m", sep = "")
	.jinit(parameters = jinit_params)
	for (dependency in JAR_DEPENDENCIES){
		.jaddClassPath(paste(find.package("bartMachine"), "/java/", dependency, sep = ""))
	} 
}

##get variable counts
get_var_counts_over_chain = function(bart_machine, type = "splits"){
	if (!(type %in% c("trees", "splits"))){
		stop("type must be \"trees\" or \"splits\"")
	}
	C = t(sapply(.jcall(bart_machine$java_bart_machine, "[[I", "getCountsForAllAttribute", as.integer(bart_machine_num_cores()), type), .jevalArray))
	colnames(C) = colnames(bart_machine$model_matrix_training_data)[1 : bart_machine$p]
	C
}

#get variable inclusion proportions
get_var_props_over_chain = function(bart_machine, type = "splits"){
	if (!(type %in% c("trees", "splits"))){
		stop("type must be \"trees\" or \"splits\"")
	}	
	attribute_props = .jcall(bart_machine$java_bart_machine, "[D", "getAttributeProps", as.integer(bart_machine_num_cores()), type)
	names(attribute_props) = colnames(bart_machine$model_matrix_training_data)[1 : bart_machine$p]
	attribute_props
}

##private function called in summary() 
sigsq_est = function(bart_machine){
	sigsqs = .jcall(bart_machine$java_bart_machine, "[D", "getGibbsSamplesSigsqs")	
	sigsqs_after_burnin = sigsqs[(length(sigsqs) - bart_machine$num_iterations_after_burn_in) : length(sigsqs)]	
	mean(sigsqs_after_burnin)
}

#There's no standard R function for this.
sample_mode = function(data){
	as.numeric(names(sort(-table(data)))[1])
}

#set up a logging system
LOG_DIR = "r_log"
log_file_name = "bart_log.txt"
bart_log = matrix(nrow = 0, ncol = 1)

##private function for handling logging
append_to_log = function(text){
	bart_log = rbind(bart_log, paste(Sys.time(), "\t", text)) #log the time and the actual message
	assign("bart_log", bart_log, .GlobalEnv)
	write.table(bart_log, paste(LOG_DIR, "/", log_file_name, sep = ""), quote = FALSE, col.names = FALSE, row.names = FALSE)
}