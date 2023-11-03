from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import types

def init_obj(
    obj_dict:Optional[dict],
    module: "types.ModuleType",
    *args, **kwargs
    ) -> Any :
    """  Initialize the object for given object dict in the module

    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given. When encountering the string argument start with "_eval_(*)", it call eval(*, __globals, __locals) to support the init function with argmuent beyond the basic type

    Args:
        obj_dict (dict | None): the object dict with key "type" for the object type name, key "args" for arguments for __init__
        module (ModuleType): the module to find the class / object type
        __globals (dict): global varialbes dict used as eval calling context
        __locals (dict): local varialbes dict used as eval calling context
        *args Tuple[Any, ...]: other positional arguments for object __init__
        **kwargs Dict[str, Any]:  other keyword arguments for object __init__
    
    Returns:
        the object 

    Examples:: 

        >>> object1 = init_obj(obj_dict, module, a, b=1)
        >>> object2 = module.obj_dict['type'](a, b=1, **obj_dict["args"])
        >>> print(object1 == object2)
        True
    """
    if obj_dict is None:
        return None
    assert isinstance(obj_dict, dict), "invalid init object dict"
        
    module_name = obj_dict['type']
    module_args = dict(obj_dict.get('args', {}))
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)
    