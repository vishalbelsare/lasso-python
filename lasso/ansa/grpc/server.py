
import argparse
import glob
import os

from lasso.logging import str_error, str_info
from lasso.utils.ConsoleColoring import ConsoleColoring

from .settings import GRPC_DEFAULT_PORT

# messages
_msg_invalid_port = "Option --port is '{0}' but must be positive."
_msg_no_site_packages = "Could not find a subdirectory 'site-packages' underneath '{0}'"
_msg_multiple_site_packages = ("Found multiple folders 'site-packages' underneath '{0}'. "
                               "Please specify the path more specifically.")
_msg_site_packages_unspecified = ("No '--python33-path' was specified. "
                                  "Assuming that all required packages are installed within ANSA.")
_msg_ansa_filepath_unspecified = "Filepath to ANSA unspecified. Trying '{0}'."


def parse_args() -> argparse.Namespace:
    ''' Parses arguments when being run from the command line

    Returns
    -------
    parser : `argparse.Namespace`
        parsed command line arguments
    '''

    header_title = 'GRPC Server for ANSA Remote Scripting from ' + \
        ConsoleColoring.blue("LASSO GmbH") + '\n' + "-"*53

    parser = argparse.ArgumentParser(
        description=header_title)
    parser.add_argument('--ansa-filepath',
                        type=str,
                        default="",
                        help='Filepath to ANSA.')
    parser.add_argument('--python33-path',
                        type=str,
                        # required=True,
                        help=('Path to the python 3.3 installation whose'
                              ' site-packages contains packages for ANSA.'))
    parser.add_argument('--port',
                        type=int,
                        default=GRPC_DEFAULT_PORT,
                        help='Port on which the remote scripting will be served.')
    parser.add_argument('--interactive',
                        type=bool,
                        default=False,
                        help='Whether to run the server in interactive mode.')
    parser.add_argument('--show-gui',
                        type=bool,
                        default=False,
                        help='Whether to show a gui or run in batch mode only.')
    parser.add_argument('--enable-logging',
                        type=bool,
                        default=False,
                        help='Whether to actively log activities to the console.')

    return parser.parse_args()


def check_args(parser: argparse.Namespace):
    ''' Check the parser arguments

    Parameters
    ----------
    parser : `argparse.Namespace`
        parsed command line arguments
    '''

    if parser.port <= 0:
        raise argparse.ArgumentTypeError(_msg_invalid_port.format(parser.port))


def get_ansa_server_command(
        ansa_filepath: str,
        python33_path: str,
        port: int,
        interactive: bool,
        show_gui: bool,
        enable_logging: bool) -> str:
    ''' Assemble the ansa command for running the server

    Parameters
    ----------
    ansa_filepath : `str`
        path to ansa executable
    python33_path : `str`
        path to python 3.3 whose site-packages shall be used
    port : `int`
        port to run remote scripting on
    interactive : `bool`
        run ansa server in interactive mode
    show_gui : `bool`
        show a GUI of the server
    enable_logging : `bool`
        whether to log actions run in ANSA

    Returns
    -------
    cmd : `list` of `str`
        ansa command for running the server
    '''

    # path to the script, which ansa has to run
    current_dir = os.path.dirname(__file__)
    server_script_filepath = os.path.join(current_dir, "server_ansa.py")

    # ansa filepath
    if not ansa_filepath:
        if os.name == 'nt':
            ansa_filepath = 'ansa64.bat'
        elif os.name == 'posix':
            ansa_filepath = 'ansa64.sh'

        print(str_info(_msg_ansa_filepath_unspecified.format(ansa_filepath)))

    # basic commands
    cmd = [
        ansa_filepath,
        '-nolauncher'
    ]

    # show a gui?
    if not show_gui:
        cmd.append("-nogui")

    # find the site-packages dir
    if python33_path:

        site_packages_dirs = glob.glob(
            os.path.join(python33_path, '**', 'site-packages'),
            recursive=True)

        if len(site_packages_dirs) == 0:
            raise RuntimeError(
                str_error(_msg_no_site_packages.format(python33_path)))

        if len(site_packages_dirs) > 1:
            raise RuntimeError(
                str_error(_msg_multiple_site_packages.format(python33_path)))

        site_packages_path = site_packages_dirs[0]
        os.environ["ANSA_GRPC_SITE_PACKAGES_PATH"] = site_packages_path

        # check if required libs are installed
        # required_libs = ["grpc*", "enum*", "protobuf3*", "google*"]
        # for required_lib_name in required_libs:
        #     if not glob.glob(os.path.join(site_packages_path, required_lib_name)):
        #         raise RuntimeError(str_error(_msg_import_error.format(
        #             _msg_missing_library.format(required_lib_name, site_packages_path)
        #         )))

    # python33 path not specified (print an info)
    elif not python33_path:
        print(str_info(_msg_site_packages_unspecified))
        site_packages_path = ""

    # this function in the script will be run with the following arguments
    cmd.append("-execscript")
    cmd.append(server_script_filepath)

    script_command = "\"serve({port},{interactive},{enable_logging})\""
    cmd.append("-execpy")
    cmd.append(
        script_command.format(
            site_packages_path=site_packages_path,
            port=port,
            interactive=interactive,
            enable_logging=enable_logging
        )
    )

    return cmd


def main():

    # parse arguments
    parser = parse_args()

    # check stuff
    check_args(parser)

    # assemble command
    cmd = get_ansa_server_command(
        ansa_filepath=parser.ansa_filepath,
        python33_path=parser.python33_path,
        port=parser.port,
        show_gui=parser.show_gui,
        interactive=parser.interactive,
        enable_logging=parser.enable_logging,

    )

    # run server
    print(str_info("Running: {0}".format(
        ' '.join(str(entry) for entry in cmd))))
    os.system(' '.join(str(entry) for entry in cmd))
    # process = subprocess.Popen(cmd, shell=True)


if __name__ == "__main__":
    main()
