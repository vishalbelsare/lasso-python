
import platform

from lasso.utils.ConsoleColoring import ConsoleColoring

# settings
MARKER_INFO = '[/]'
MARKER_RUNNING = '[~]'
MARKER_WARNING = '[!]'
MARKER_SUCCESS = '[Y]' if platform.system() == 'Windows' else '[✔]'
MARKER_ERROR = '[X]' if platform.system() == 'Windows' else '[✘]'


def str_info(msg):
    ''' Format a message as stuff is running

            Parameters
            ----------
            msg : `str`
                message to format

            Returns
            -------
            msg_ret : `str`
                formatted message
        '''
    # return ConsoleColoring.blue("[/] {0}".format(msg), light=True)
    return "{0} {1}".format(MARKER_INFO, msg)


def str_running(msg):
    ''' Format a message as stuff is running

    Parameters
    ----------
    msg : `str`
        message to format

    Returns
    -------
    msg_ret : `str`
        formatted message
    '''
    return "{0} {1}".format(MARKER_RUNNING, msg)


def str_success(msg):
    ''' Format a message as successful

    Parameters
    ----------
    msg : `str`
        message to format

    Returns
    -------
    msg_ret : `str`
        formatted message
    '''
    return ConsoleColoring.green("{0} {1}".format(MARKER_SUCCESS, msg))


def str_warn(msg):
    ''' Format a string as a warning

    Parameters
    ----------
    msg : `str`
        message to format

    Returns
    -------
    msg_ret : `str`
        formatted message
    '''
    return ConsoleColoring.yellow("{0} {1}".format(MARKER_WARNING, msg))


def str_error(msg):
    ''' Format a string as an error

    Parameters
    ----------
    msg : `str`
        message to format

    Returns
    -------
    msg_ret : `str`
        formatted message
    '''
    return ConsoleColoring.red("{0} {1}".format(MARKER_ERROR, msg))
