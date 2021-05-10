import functools
# local imports

UI_PLUGINS = {}

class NameConflictError(BaseException):
    """Raise for errors in adding plugins due to the same name."""

def register_plugin(name):
    if name in UI_PLUGINS:
        raise NameConflictError(
            f"Plugin name conflict: '{name}'. Double check" \
            " that all plugins have unique names.")

    def wrapper_register_plugin(func):
        UI_PLUGINS[name] = func

    return wrapper_register_plugin

def input_generic(name, *args, **kwargs):
    return UI_PLUGINS[name](*args, **kwargs)

input_gtm = functools.partial(input_generic, 'input_gtm')
input_analysis_type = functools.partial(input_generic, 'input_analysis_type')
input_main_metric = functools.partial(input_generic, 'input_main_metric')
input_metric_range = functools.partial(input_generic, 'input_metric_range')
input_timeline = functools.partial(input_generic, 'input_timeline')
input_bar_limit = functools.partial(input_generic, 'input_bar_limit')
input_ticker = functools.partial(input_generic, 'input_ticker')


output_table = functools.partial(input_generic, 'expandable_table')



