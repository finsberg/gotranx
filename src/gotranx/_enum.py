from enum import Enum, EnumMeta

# https://stackoverflow.com/questions/62299740/how-do-i-detect-and-invoke-a-function-when-a-python-enum-member-is-accessed


class OnAccess(EnumMeta):
    """
    runs a user-specified function whenever member is accessed
    """

    #
    def __getattribute__(cls, name):
        obj = super().__getattribute__(name)
        if isinstance(obj, Enum) and obj._on_access:
            obj._on_access()
        return obj

    #
    def __getitem__(cls, name):
        member = super().__getitem__(name)
        if member._on_access:
            member._on_access()
        return member

    #
    def __call__(cls, value, names=None, *, module=None, qualname=None, type=None, start=1):
        obj = super().__call__(
            value, names, module=module, qualname=qualname, type=type, start=start
        )
        if isinstance(obj, Enum) and obj._on_access:
            obj._on_access()
        return obj
