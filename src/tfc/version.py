__version__ = "1.3.1"


def _version_as_tuple(version_str: str) -> tuple[int, ...]:
    return tuple(int(i) for i in version_str.split(".") if i.isdigit())


__version_info__ = _version_as_tuple(__version__)
