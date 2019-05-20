
import platform
import shutil
import glob
import re
import typing
import os
import sys
import psutil
import argparse
import time
from typing import Union, List
from concurrent import futures
import subprocess

from lasso.utils.ConsoleColoring import ConsoleColoring
from lasso.logging import str_error, str_running, str_success, str_warn, str_info


def print_application_header():
    ''' Prints the header of the command line tool
    '''

    print("")
    print("   ==== D I F F C R A S H ==== ")
    print("")
    print(" a " + ConsoleColoring.blue("LASSO GmbH",
                                       light=True) + " utility script.")
    print("")


def parse_diffcrash_args():
    ''' Parse the arguments from the command line

    Returns
    -------
    args : `argparse.Namespace`
        parsed arguments
    '''

    # print title
    print_application_header()

    parser = argparse.ArgumentParser(
        description="Python utility script for Diffcrash written by LASSO GmbH.")

    parser.add_argument("--reference-run",
                        type=str,
                        required=True,
                        help="filepath of the reference run.")
    parser.add_argument("--crash-code",
                        type=str,
                        required=True,
                        help="Which crash code is used ('dyna', 'pam' or 'radioss').")
    parser.add_argument("--diffcrash-home",
                        type=str,
                        default=os.environ["DIFFCRASHHOME"] if "DIFFCRASHHOME" in os.environ else "",
                        nargs='?',
                        required=False,
                        help="Home directory where Diffcrash is installed. Uses environment variable 'DIFFCRASHHOME' if unspecified.")
    parser.add_argument("--use-id-mapping",
                        type=bool,
                        nargs='?',
                        default=False,
                        help="Whether to use id-based mapping (default is nearest neighbour).")
    parser.add_argument("--project-dir",
                        type=str,
                        nargs='?',
                        default="project",
                        help="Project dir to use for femzip.")
    parser.add_argument("--config-file",
                        type=str,
                        nargs='?',
                        default="",
                        help="Path to the config file.")
    parser.add_argument("--parameter-file",
                        type=str,
                        nargs='?',
                        default="",
                        help="Path to the parameter file.")
    parser.add_argument("--n-processes",
                        type=int,
                        nargs='?',
                        default=max(1, psutil.cpu_count() - 1),
                        help="Number of processes to use (default: max-1)")
    parser.add_argument("simulation_runs",
                        type=str,
                        nargs='*',
                        help="Simulation runs or patterns used to search for simulation runs.")

    if len(sys.argv) < 2:
        parser.print_help()
        exit(0)

    return parser.parse_args(sys.argv[1:])


def run_subprocess(args):
    ''' Run a subprocess with the specified arguments

    Parameters:
    -----------
        args : `list` of `str`

    Returns
    -------
    rc : `int`
        process return code

    Notes
    -----
        Suppresses stderr.
    '''
    devnull = open(os.devnull, 'wb')
    return subprocess.Popen(args, stderr=devnull).wait()



class DiffcrashRun(object):
    ''' Class for handling the settings of a diffcrash run
    '''

    def __init__(self, 
        project_dir:str, 
        crash_code:str, 
        reference_run:str, 
        simulation_runs: typing.Sequence[str], 
        diffcrash_home:str="", 
        use_id_mapping:bool = False, 
        config_file:str = None, 
        parameter_file: str = None, 
        n_processes: int =1, 
        logfile_dir: str = None):
        ''' Object handling a diffcrash run

        Parameters
        ----------
        project_dir : `str`
            directory to put all buffer files etc in
        crash_code : `str`
            crash code to use.
        reference_run : `str`
            filepath to the reference run
        simulation_runs: `list` of `str`
            patterns used to search for simulation runs
        diffcrash_home : `str`
            home directory of diffcrash installation. Uses environment
            variable DIFFCRASHHOME if not set.
        use_id_mapping : `bool`
            whether to use id mapping instead of nearest neighbor mapping
        config_file : `str`
            filepath to a config file
        parameter_file : `str`
            filepath to the parameter file
        n_processes : `int`
            number of processes to spawn for worker pool
        logfile_dir : `str`
            directory to put logfiles in
        '''

        # settings
        self._msg_option = "{:16s}: {}"

        # diffcrash home
        self.diffcrash_home = self._parse_diffcrash_home(diffcrash_home)
        self.diffcrash_home = os.path.join(self.diffcrash_home, 'bin')
        self.diffcrash_lib = os.path.join(os.path.dirname(self.diffcrash_home), "lib")

        if platform.system() == "Linux":
            os.environ['PATH'] = os.environ['PATH'] + \
                ":" + self.diffcrash_home + ":" + self.diffcrash_lib
        if platform.system() == "Windows":
            os.environ['PATH'] = os.environ['PATH'] + \
                ";" + self.diffcrash_home + ";" + self.diffcrash_lib

        # project dir
        self.project_dir = self._parse_project_dir(project_dir)

        # crashcode
        self.crash_code = self._parse_crash_code(crash_code)

        # reference run
        self.reference_run = self._parse_reference_run(reference_run)

        # mapping
        self.use_id_mapping = self._parse_use_id_mapping(use_id_mapping)

        # simulation runs
        self.simulation_runs = self._parse_simulation_runs(simulation_runs, self.reference_run)

        # config file
        self.config_file = self._parse_config_file(config_file)

        # parameter file
        self.parameter_file = self._parse_parameter_file(parameter_file)

        # n processes
        self.n_processes = self._parse_n_processes(n_processes)
        # if pool != None:
        #     self.pool = pool
        # else:
        #     self.pool = futures.ThreadPoolExecutor(max_workers=self.n_processes)

        # logdir
        if logfile_dir != None:
            self.logfile_dir = logfile_dir
        else:
            self.logfile_dir = os.path.join(self.project_dir, "Log")
        

    def _parse_diffcrash_home(self, diffcrash_home) -> str:
 
        diffcrash_home_ok = len(diffcrash_home) != 0

        print(str_info(self._msg_option.format("diffcrash-home", diffcrash_home)))
        if not diffcrash_home_ok:
            raise RuntimeError(str_error("Either specify with the environment variable DIFFCRASHHOME or the optione --diffcrash-home the path to the Diffcrash installation."))

        return diffcrash_home
    
    def _parse_crash_code(self, crash_code) -> str:

        # these guys are allowed
        valid_crash_codes = ["dyna", "radioss", "pam"]

        # do the thing
        crash_code_ok = crash_code in valid_crash_codes

        print(str_info(self._msg_option.format("crash-code", crash_code)))

        if not crash_code_ok:
            raise RuntimeError(str_error("Invalid crash code '{0}'. Please use one of: {1}".format(crash_code, str(valid_crash_codes))))
    
        return crash_code

    def _parse_reference_run(self, reference_run) -> str:

        reference_run_ok = os.path.isfile(reference_run)

        print(str_info(self._msg_option.format("reference-run", reference_run)))

        if not reference_run_ok:
            raise RuntimeError(str_error("Filepath '{0}' is not a file.".format(reference_run)))
            
        return reference_run

    def _parse_use_id_mapping(self, use_id_mapping) -> bool:
        
        print(str_info(self._msg_option.format("use-id-mapping", use_id_mapping)))
        return use_id_mapping

    def _parse_project_dir(self, project_dir) -> str:
        print(str_info(self._msg_option.format("project-dir", project_dir)))
        return project_dir

    def _parse_simulation_runs(self, simulation_run_patterns, reference_run):
        
        simulation_runs = []
        for pattern in simulation_run_patterns:
            simulation_runs += glob.glob(pattern)
        simulation_runs = [filepath for filepath in simulation_runs if os.path.isfile(filepath)]
        if reference_run in simulation_runs:
            simulation_runs.remove(reference_run)
        
        # sort it because we can!
        atoi = lambda text : int(text) if text.isdigit() else text
        natural_keys = lambda text : [atoi(c) for c in re.split(r'(\d+)', text)]
        simulation_runs = sorted(simulation_runs, key=natural_keys)

        # check
        simulation_runs_ok = len(simulation_runs) != 0
        
        print(str_info(self._msg_option.format("# simul.-files", len(simulation_runs))))
        
        if not simulation_runs_ok:
            raise RuntimeError(str_error("No simulation files could be found with the specified patterns. Check the argument 'simulation_runs'."))

        return simulation_runs

    def _parse_config_file(self, config_file) -> Union[str, None]:

        _msg_config_file = ""
        if len(config_file) > 0:

            if not os.path.isfile(config_file):
                config_file = None
                _msg_config_file = str_warn(
                    "Can not find config file '{}'".format(config_file))
            else:
                config_file = config_file

        # missing config file
        else:

            config_file = None
            _msg_config_file = str_warn(
                "Config file missing. Consider specifying the path with the option '--config-file'.")

        print(str_info(self._msg_option.format("config-file", config_file)))

        if _msg_config_file:
            print(_msg_config_file)

        return config_file
    
    def _parse_parameter_file(self, parameter_file) -> Union[None, str]:
        
        _msg_parameter_file = ""
        if len(parameter_file) > 0:

            if not os.path.isfile(parameter_file):
                parameter_file = None
                _msg_parameter_file = str_warn(
                    "Can not find parameter file '{}'".format(parameter_file))
            else:
                parameter_file = parameter_file

        # missing parameter file
        else:

            parameter_file = None
            _msg_parameter_file = str_warn(
                "Parameter file missing. Consider specifying the path with the option '--parameter-file'.")

        print(str_info(self._msg_option.format("parameter-file", parameter_file )))
        
        if _msg_parameter_file:
            print(_msg_parameter_file)

        return parameter_file

    def _parse_n_processes(self, n_processes) -> int:

        print(str_info(self._msg_option.format("n-processes", n_processes)))

        if n_processes <= 0:
            print(str_error("n-processes is '{0}' but must be at least 1.".format(n_processes)))
            exit(1)

        return n_processes

    def create_project_dirs(self):
        ''' Creates all project relevant directores

        Notes
        -----
            Created dirs:
             - logfile_dir
             - project_dir
        '''
        os.makedirs(self.project_dir, exist_ok=True)
        os.makedirs(self.logfile_dir, exist_ok=True)

    def run_setup(self, pool: futures.ThreadPoolExecutor):
        ''' Run diffcrash setup

        Parameters
        ----------
        pool : `concurrent.futures.ThreadPoolExecutor`
            multiprocessing pool
        '''

        # SETUP
        print()
        print(str_running("Running Setup ... ")+'\r', end='', flush='')
        args = []
        if self.config_file is None and self.parameter_file is None:
            args = [os.path.join(self.diffcrash_home, "DFC_Setup_"
                                    + self.crash_code + "_fem"), self.reference_run, self.project_dir]
        elif self.config_file is not None and self.parameter_file is None:
            args = [os.path.join(self.diffcrash_home, "DFC_Setup_" + self.crash_code
                                    + "_fem"), self.reference_run, self.project_dir, "-C", self.config_file]
        elif self.config_file is None and self.parameter_file is not None:
            if ".fz" in self.reference_run:
                args = [os.path.join(self.diffcrash_home, "DFC_Setup_" + self.crash_code
                                        + "_fem"), self.reference_run, self.project_dir, "-P", self.parameter_file]
            else:
                args = [os.path.join(self.diffcrash_home, "DFC_Setup_" + self.crash_code),
                        self.reference_run, self.project_dir, "-P", self.parameter_file]
        elif self.config_file is not None and self.parameter_file is not None:
            if ".fz" in self.reference_run:
                args = [os.path.join(self.diffcrash_home, "DFC_Setup_" + self.crash_code + "_fem"),
                        self.reference_run, self.project_dir, "-C", self.config_file, "-P", self.parameter_file]
            else:
                args = [os.path.join(self.diffcrash_home, "DFC_Setup_" + self.crash_code),
                        self.reference_run, self.project_dir, "-C", self.config_file, "-P", self.parameter_file]
        start_time = time.time()

        # submit task
        return_code_future = pool.submit(run_subprocess, args)
        return_code = return_code_future.result()

        # check return code
        if return_code != 0:
            print(str_error("Running Setup ... done in {0:.2f}s").format(time.time() - start_time))

            raise RuntimeError(str_error("Process somehow failed."))
        
        # check log
        messages = self.check_if_logfiles_show_success("DFC_Setup.log")
        if messages:
            print(str_error("Running Setup ... done in {0:.2f}s").format(time.time() - start_time))
            
            # print failed logs
            for msg in messages:
                print(str_error(msg))
            
            raise RuntimeError(str_error("Setup failed."))

        # print success
        print(str_success("Running Setup ... done in {0:.2f}s").format(time.time() - start_time))

    def run_import(self, pool: futures.ThreadPoolExecutor):
        ''' Run diffcrash import of runs

        Parameters
        ----------
        pool : `concurrent.futures.ThreadPoolExecutor`
            multiprocessing pool
        '''

        # list of arguments to run in the command line
        import_arguments = []

        # id 1 is the reference run
        # id 2 and higher are the imported runs
        counter_offset = 2

        # assemble arguments for running the import 
        # entry 0 is the reference run, thus we start at 1
        for i_filepath in range(len(self.simulation_runs)):

            # parameter file missing
            if self.parameter_file is None:
                if self.use_id_mapping:
                    args = [os.path.join(
                        self.diffcrash_home, "DFC_Import_" + self.crash_code + "_fem"), "-id", self.simulation_runs[i_filepath], self.project_dir, str(i_filepath + counter_offset)]
                else:
                    args = [os.path.join(
                        self.diffcrash_home, "DFC_Import_" + self.crash_code + "_fem"), self.simulation_runs[i_filepath], self.project_dir, str(i_filepath + counter_offset)]
            # indeed there is a parameter file
            else:
                if self.use_id_mapping:
                    args = [os.path.join(
                        self.diffcrash_home, "DFC_Import_" + self.crash_code), "-ID", self.simulation_runs[i_filepath], self.project_dir, str(i_filepath + counter_offset)]
                else:
                    args = [os.path.join(
                        self.diffcrash_home, "DFC_Import_" + self.crash_code), self.simulation_runs[i_filepath], self.project_dir, str(i_filepath + counter_offset)]

            # append args to list
            import_arguments.append(args)

        # do the thing
        print(str_running("Running Imports ...")+'\r', end='', flush=True)
        start_time = time.time()
        return_code_futures = [pool.submit(run_subprocess,args)
                                for args in import_arguments]

        # wait for imports to finish (with a progressbar)
        n_imports_finished = sum(return_code_future.done() for return_code_future in return_code_futures)
        while n_imports_finished != len(return_code_futures):
            
            # check again
            n_new_imports_finished = sum(return_code_future.done() for return_code_future in return_code_futures)

            # print
            percentage = n_new_imports_finished/len(return_code_futures)*100
            print(str_running("Running Imports ... [{0}/{1}] - {2:3.2f}%\r".format(n_new_imports_finished, len(return_code_futures), percentage)), end='', flush=True)
            
            n_imports_finished = n_new_imports_finished
            
            # wait a little bit
            time.sleep(0.25)

        return_codes = [ return_code_future.result() for return_code_future in return_code_futures ]
            
        # print failure
        if any(return_code != 0 for return_code in return_codes):

            n_failed_runs = 0
            for i_run, return_code in enumerate(return_codes):
                if return_code != 0:
                    _err_msg = str_error(
                        "Run {0} failed to import with error code '{1}'.".format(i_run, return_code))
                    print(str_error(_err_msg))
                    n_failed_runs += 1

            print(str_error("Running Imports ... done in {0:.2f}s   ").format(time.time() - start_time))
            raise RuntimeError(
                str_error("Import of {0} runs failed.".format(n_failed_runs)))

        # check log files
        messages = self.check_if_logfiles_show_success('DFC_Import_*.log')
        if messages:
            
            # print failure
            print(str_error("Running Imports ... done in {0:.2f}s   ").format(time.time() - start_time))
            
            # print failed logs
            for msg in messages:
                print(str_error(msg))

            raise RuntimeError(str_error("At least one import failed. Please check the log files in '{0}'.".format(self.logfile_dir)))

        # print success
        print(str_success("Running Imports ... done in {0:.2f}s   ").format(time.time() - start_time))


    def run_math(self, pool: futures.ThreadPoolExecutor):
        ''' Run diffcrash math

        Parameters
        ----------
        pool : `concurrent.futures.ThreadPoolExecutor`
            multiprocessing pool
        '''


        print(str_running("Running Math ... ")+'\r', end='', flush=True)

        start_time = time.time()
        return_code_future = pool.submit( 
            run_subprocess,
            [os.path.join(self.diffcrash_home, "DFC_Math_" + self.crash_code), self.project_dir])
        returnCode = return_code_future.result()
        
        # check return code
        if returnCode != 0:
            print(str_error("Running Math ... done in {0:.2f}s   ").format(time.time() - start_time))
            raise RuntimeError(str_error("Caught a nonzero return code '{0}'".format(returnCode)))

        # check logs 
        messages = self.check_if_logfiles_show_success("DFC_MATH*.log")
        if messages:
            
            # print failure
            print(str_error("Running Math ... done in {0:.2f}s   ").format(time.time() - start_time))

            # print failed logs
            for msg in messages:
                print(str_error(msg))

            raise RuntimeError(str_error("Logfile does indicate a failure. Please check the log files in '{0}'.".format(self.logfile_dir)))

        #print success
        print(str_success("Running Math ... done in {0:.2f}s   ").format(time.time() - start_time))


    def run_export(self, pool: futures.ThreadPoolExecutor):
        ''' Run diffcrash export

        Parameters
        ----------
        pool : `concurrent.futures.ThreadPoolExecutor`
            multiprocessing pool
        '''

        print(str_running("Running Export ... ")+'\r', end='', flush=True)

        if self.config_file is None:
            export_item_list = []

            # check for pdmx
            pdmx_filepath_list = glob.glob(os.path.join(self.project_dir,"*_pdmx"))
            if pdmx_filepath_list:
                export_item_list.append( os.path.basename(pdmx_filepath_list[0]))

            # check for pdij
            pdij_filepath_list = glob.glob(os.path.join(self.project_dir,"*_pdij"))
            if pdij_filepath_list:
                export_item_list.append( os.path.basename(pdij_filepath_list[0]))

        else:
            export_item_list = self.read_config_file(self.config_file)

        # run exports
        start_time = time.time()
        return_code_futures = [ pool.submit(
                run_subprocess,
                [os.path.join(self.diffcrash_home, "DFC_Export_" + self.crash_code), self.project_dir, export_item]) 
                for export_item in export_item_list ]

        return_codes = [result_future.result() for result_future in return_code_futures]

        # check return code
        if any(rc != 0 for rc in return_codes):
            print(str_error("Running Export ... done in {0:.2f}s   ").format(time.time() - start_time))

            for i_export, export_return_code in enumerate(return_codes):
                if export_return_code != 0:
                    print(str_error("Return code of export '{0}' was nonzero: '{1}'".format(export_item_list[i_export], export_return_code)))

            raise RuntimeError(str_error("At least one export process failed."))

        # check logs
        messages = self.check_if_logfiles_show_success("DFC_Export_*")
        if messages:

            # print failure
            print(str_error("Running Export ... done in {0:.2f}s   ").format(time.time() - start_time))

            # print logs
            for msg in messages:
                print(str_error(msg))

            raise RuntimeError(str_error("At least one export failed. Please check the log files in '{0}'.".format(self.logfile_dir)))
        
        # print success
        print(str_success("Running Export ... done in {0:.2f}s   ").format(time.time() - start_time))

    def run_matrix(self, pool: futures.ThreadPoolExecutor):
        ''' Run diffcrash matrix

        Parameters
        ----------
        pool : `concurrent.futures.ThreadPoolExecutor`
            multiprocessing pool
        '''

        print(str_running("Running Matrix ... ")+'\r', end='', flush=True)

        start_time = time.time()

        # create the input file for the process
        matrix_inputfile = self._create_matrix_input_file(self.project_dir)

        # run the thing
        return_code_future = pool.submit(
            run_subprocess,
            [os.path.join(self.diffcrash_home, "DFC_Matrix_" + self.crash_code), self.project_dir, matrix_inputfile])

        # please hold the line ...
        return_code = return_code_future.result()

        # check return code
        if return_code != 0:

            # print failure
            print(str_error("Running Matrix ... done in {0:.2f}s   ").format(time.time() - start_time))

            raise RuntimeError(str_error("The DFC_Matrix process failed somehow."))

        # check log file
        messages = self.check_if_logfiles_show_success("DFC_Matrix_*")
        if messages:

            # print failure
            print(str_error("Running Matrix ... done in {0:.2f}s   ").format(time.time() - start_time))

            # print why
            for msg in messages:
                print(str_error(msg))

            raise RuntimeError(str_error("DFC_Matrix failed. Please check the log files in '{0}'.".format(self.logfile_dir)))
        
        # print success
        print(str_success("Running Matrix ... done in {0:.2f}s   ").format(time.time() - start_time))

    def run_eigen(self, pool: futures.ThreadPoolExecutor):
        ''' Run diffcrash eigen

        Parameters
        ----------
        pool : `concurrent.futures.ThreadPoolExecutor`
            multiprocessing pool
        '''

        print(str_running("Running Eigen ... ")+'\r', end='', flush=True)
            
        # create input file for process
        eigen_inputfile = self._create_eigen_input_file(self.project_dir)

        # run the thing
        start_time = time.time()
        return_code_future = pool.submit(run_subprocess,
                                        [os.path.join(self.diffcrash_home, "DFC_Eigen_" + self.crash_code), self.project_dir, eigen_inputfile])

        # please hold the line ...
        return_code = return_code_future.result()
        
        # check return code
        if return_code != 0:
            print(str_error("Running Eigen ... done in {0:.2f}s   ").format(time.time() - start_time))
            raise RuntimeError(str_error("The process failed somehow."))
        
        # check log file
        messages = self.check_if_logfiles_show_success("DFC_Matrix_*")
        if messages:

            # print failure
            print(str_error("Running Eigen ... done in {0:.2f}s   ").format(time.time() - start_time))

            # print why
            for msg in messages:
                print(str_error(msg))

            raise RuntimeError(str_error("DFC_Eigen failed. Please check the log files in '{0}'.".format(self.logfile_dir)))
        
        # print success
        print(str_success("Running Eigen ... done in {0:.2f}s   ").format(time.time() - start_time))

    def run_merge(self, pool: futures.ThreadPoolExecutor):
        ''' Run diffcrash merge

        Parameters
        ----------
        pool : `concurrent.futures.ThreadPoolExecutor`
            multiprocessing pool
        '''

        print(str_running("Running Merge ... ")+'\r', end='', flush=True)

        # create ionput file for merge
        merge_inputfile = self._create_merge_input_file(self.project_dir)

        # run the thing
        start_time = time.time()
        return_code_future = pool.submit(run_subprocess,
            [os.path.join(self.diffcrash_home, "DFC_Merge_All_" + self.crash_code), self.project_dir, merge_inputfile])
        return_code = return_code_future.result()

        # check return code
        if return_code != 0:
            print(str_error("Running Merge ... done in {0:.2f}s   ").format(time.time() - start_time))
            raise RuntimeError(str_error("The process failed somehow."))

        # check logfiles
        messages = self.check_if_logfiles_show_success("DFC_Merge_All.log")
        if messages:
            print(str_error("Running Merge ... done in {0:.2f}s   ").format(time.time() - start_time))

            for msg in messages:
                print(str_error(msg))

            raise RuntimeError(str_error("DFC_Merge_All failed. Please check the log files in '{0}'.".format(self.logfile_dir)))
        
        # print success
        print(str_success("Running Merge ... done in {0:.2f}s   ").format(time.time() - start_time))

    def is_logfile_successful(self, logfile: str) -> bool:
        ''' Checks if a logfile indicates a success

        Parameters
        ----------
        logfile : `str`
            filepath to the logile

        Returns
        -------
        success : `bool`
        '''

        with open(logfile, "r") as fp:
            for line in fp:
                if "successfully" in line:
                    return True
        return False

    
    def _create_merge_input_file(self, directory: str, dataname='default') -> str:
        ''' Create an input file for the merge executable

        Notes
        -----
            From the official diffcrash docs.
        '''

        # creates default inputfile for DFC_Merge
        merge_input_file = open(os.path.join(directory, "merge_all.txt"), "w")
        #	merge_input_file.write("\"" + directory + "/\"\n")
        merge_input_file.write("eigen_all        ! Name of eigen input file\n")
        merge_input_file.write(
            "mode_            ! Name of Output file (string will be apended with mode information)\n")
        merge_input_file.write("1 1              ! Mode number to be generated\n")
        merge_input_file.write("'d+ d-'          ! Mode type to be generated\n")
        # TIMESTEPSFILE         optional
        merge_input_file.write(
            "                 ! Optional: Timestepfile (specify timesteps used for merge)\n")
        # PARTSFILE             optional
        merge_input_file.write(
            "                 ! Optional: Partlistfile (specify parts used for merge)\n")
        #    merge_input_file.write("1.5 300\n") #pfactor pmax  optional
        merge_input_file.close()

        return os.path.join(directory, "merge_all.txt")

    def _create_eigen_input_file(self, directory: str) -> str:
        ''' Create an input file for the eigen executable

        Notes
        -----
            From the official diffcrash docs.
        '''

        # creates default inputfile for DFC_Eigen
        eigen_input_file = open(os.path.join(directory, "eigen_all.txt"), "w")
        #	eigen_input_file.write("\"" + project_dir + "/\"\n")
        eigen_input_file.write("matrix_all\n")
        eigen_input_file.write("\"\"\n")
        eigen_input_file.write("1 1000\n")
        eigen_input_file.write("\"\"\n")
        eigen_input_file.write("0 0\n")
        eigen_input_file.write("\"\"\n")
        eigen_input_file.write("eigen_all\n")

        eigen_input_file.write("\"\"\n")
        eigen_input_file.write("0 0\n")
        eigen_input_file.close()

        return os.path.join(directory, "eigen_all.txt")

    def _create_matrix_input_file(self, directory: str) -> str:
        ''' Create an input file for the matrix executable

        Notes
        -----
            From the official diffcrash docs.
        '''

        # creates default inputfile for DFC_Matrix
        matrix_input_file = open(os.path.join(directory, "matrix.txt"), "w")
        matrix_input_file.write(
            "0 1000        !    Initial and final time stept to consider\n")
        matrix_input_file.write("\"\"          !    not used\n")
        matrix_input_file.write("\"\"          !    not used\n")
    #	matrix_input_file.write("\"" + project_dir + "/\"\n")
        matrix_input_file.write(
            "matrix_all    !    Name of matrix file set (Output)\n")
        matrix_input_file.close()

        return os.path.join(directory, "matrix.txt")

    def clear_project_dir(self):
        ''' Clears the entire project dir
        '''
        if os.path.exists(self.project_dir):
            shutil.rmtree(self.project_dir)

    
    def read_config_file(self, config_file: str) -> List[str]:
        ''' Read a diffcrash config file

        Parameters
        ----------
        config_file : `str`
            path to the config file

        Notes
        -----
            From the official diffcrash docs ... seriously.
        '''

        # Just to make it clear, this is not code from LASSO
        # ...

        with open(config_file, "r") as conf:
            conf_lines = conf.readlines()
        line = 0

        for i in range(0, len(conf_lines)):
            if conf_lines[i].find("FUNCTION") >= 0:
                line = i + 1
                break

        export_item_list = []
        j = 1
        if line != 0:
            while 1:
                while 1:
                    for i in range(0, len(conf_lines[line])):
                        if conf_lines[line][i] == "<":
                            element_start = i + 1
                        if conf_lines[line][i] == ">":
                            element_end = i
                    ELEM = conf_lines[line][element_start:element_end]
                    check = conf_lines[line + j][:-1]

                    if check.find(ELEM) >= 0:
                        line = line + j + 1
                        j = 1
                        break
                    j += 1
                    items = check.split(' ')
                    pos = -1
                    for n in range(0, len(items)):
                        if items[n].startswith('!'):
                            print("FOUND at ", n)
                            pos = n
                            break
                        else:
                            pos = len(items)
                    for n in range(0, pos):
                        if items[n] == "PDMX" or items[n] == "pdmx":
                            break
                        elif items[n] == "PDXMX" or items[n] == "pdxmx":
                            break
                        elif items[n] == "PDYMX" or items[n] == "pdymx":
                            break
                        elif items[n] == "PDZMX" or items[n] == "pdzmx":
                            break
                        elif items[n] == "PDIJ" or items[n] == "pdij":
                            break
                        elif items[n] == "STDDEV" or items[n] == "stddev":
                            break
                        elif items[n] == "NCOUNT" or items[n] == "ncount":
                            break
                        elif items[n] == "MISES_MX" or items[n] == "mises_mx":
                            break
                        elif items[n] == "MISES_IJ" or items[n] == "mises_ij":
                            break

                    for k in range(n, pos):
                        POSTVAL = None
                        for m in range(0, n):
                            if items[m] == "coordinates":
                                items[m] = "geometry"
                            if POSTVAL is None:
                                POSTVAL = items[m]
                            else:
                                POSTVAL = POSTVAL + "_" + items[m]
                        POSTVAL = POSTVAL.strip("_")

                        items[k] = items[k].strip()

                        if items[k] != "" and items[k] != "\r":
                            if POSTVAL.lower() == "sigma":
                                export_item_list.append(
                                    ELEM + "_" + POSTVAL + "_" + "001_" + items[k].lower())
                                export_item_list.append(
                                    ELEM + "_" + POSTVAL + "_" + "002_" + items[k].lower())
                                export_item_list.append(
                                    ELEM + "_" + POSTVAL + "_" + "003_" + items[k].lower())
                            else:
                                export_item_list.append(
                                    ELEM + "_" + POSTVAL + "_" + items[k].lower())
                        if export_item_list[-1].endswith("\r"):
                            export_item_list[-1] = export_item_list[-1][:-1]

                if conf_lines[line].find("FUNCTION") >= 0:
                    break
        else:
            export_item_list = ["NODE_geometry_pdmx", "NODE_geometry_pdij"]

        return export_item_list

    def check_if_logfiles_show_success(self, pattern: str) -> List[str]:
        ''' Check if a logfiles with given pattern show success

        Parameters
        ----------
        pattern : `str`
            file pattern used to search for logfiles

        Returns
        -------
        messages : `list`
            list with messages of failed log checks
        '''

        _msg_logfile_nok = str_error("Logfile '{0}' reports no success.")
        messages = []

        logfiles = glob.glob(os.path.join(self.logfile_dir, pattern))
        for filepath in logfiles:
            if not self.is_logfile_successful(filepath):
                # logger.warning()
                messages.append(_msg_logfile_nok.format(filepath))

        return messages