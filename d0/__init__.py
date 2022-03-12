is_simple_core = True

if is_simple_core:
    from d0.core_simple import Variable
    from d0.core_simple import Function
    from d0.core_simple import using_config
    from d0.core_simple import no_grad
    from d0.core_simple import as_array
    from d0.core_simple import as_variable
    from d0.core_simple import setup_variable
else:
    from d0.core import Variable
    from d0.core import Function
    from d0.core import using_config
    from d0.core import no_grad
    from d0.core import as_array
    from d0.core import as_variable
    from d0.core import setup_variable

setup_variable()
