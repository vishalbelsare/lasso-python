

from concurrent import futures
from lasso.diffcrash.DiffcrashRun import DiffcrashRun, parse_diffcrash_args


def main():

    # parse command line stuff
    parser = parse_diffcrash_args()

    # parse settings from command line
    diffcrash_run = DiffcrashRun(
        project_dir=parser.project_dir,
        crash_code=parser.crash_code,
        reference_run=parser.reference_run,
        simulation_runs=parser.simulation_runs,
        diffcrash_home=parser.diffcrash_home,
        use_id_mapping=parser.use_id_mapping,
        config_file=parser.config_file,
        parameter_file=parser.parameter_file,
        n_processes=parser.n_processes
    )

    # remove old stuff
    diffcrash_run.clear_project_dir()
    diffcrash_run.create_project_dirs()

    # do the thing
    print()
    print("   ---- Running Routines ----   ")

    # initiate threading pool for handling jobs
    with futures.ThreadPoolExecutor(max_workers=diffcrash_run.n_processes) as pool:

        # setup
        diffcrash_run.run_setup(pool)

        # import
        diffcrash_run.run_import(pool)

        # math
        diffcrash_run.run_math(pool)

        # export
        diffcrash_run.run_export(pool)

        # TODO EXPORT_add
        '''
        if returnCode == 0:
            if len(export_item_list) > 1:
                print("Export add ...")
                exportadd_functionals = [export_item_list[1]]
                if len(export_item_list) > 2:
                    for i in range(2, len(export_item_list)):
                        exportadd_functionals.append(export_item_list[i])
                exportadd_args = [os.path.join(DIFFCRASHHOME, "DFC_Export_add_" + CRASHCODE), project_dir, os.path.join(
                    project_dir, export_item_list[0] + file_extension), os.path.join(project_dir, "EXPORT_ADD") + file_extension]
                for i in range(0, len(exportadd_functionals)):
                    exportadd_args.append(exportadd_functionals[i])
                returnCode = startproc(exportadd_args)
            else:
                for i in range(1, len(export_item_list)):
                    print("Export", export_item_list[i], "...")
                    returnCode = startproc(
                        [os.path.join(DIFFCRASHHOME, "DFC_Export_" + CRASHCODE), project_dir, export_item_list[i]])
        '''

        # matrix
        diffcrash_run.run_matrix(pool)

        # eigen
        diffcrash_run.run_eigen(pool)

        # merge
        diffcrash_run.run_merge(pool)

    # final spacing
    print()


if __name__ == "__main__":
    main()
