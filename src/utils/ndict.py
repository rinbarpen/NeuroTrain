from typing import Dict, List, Union, Optional, Sequence, TypedDict, Tuple, NamedTuple

class NDict(dict):
    """
    NDict类型支持dict的所有功能，可通过unfold展开其值（或自动展开）
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        # 自动展开子dict为NDict
        if isinstance(value, dict) and not isinstance(value, NDict):
            value = NDict(value)
            super().__setitem__(key, value)
        return value

    def __setitem__(self, key, value):
        # 自动将子dict变为NDict
        if isinstance(value, dict) and not isinstance(value, NDict):
            value = NDict(value)
        super().__setitem__(key, value)

    def unfold(self, recursive: bool = False):
        """
        unfold：将所有的values合并为一个list
        如果recursive=True，则递归展开所有的嵌套NDict
        """
        result = []
        for value in self.values():
            if isinstance(value, NDict) and recursive:
                result.extend(value.unfold(recursive=True))
            else:
                result.append(value)
        return result

    def copy(self):
        return NDict(super().copy())

    def to_dict(self):
        """
        转换成普通的dict，递归转换所有NDict为dict
        """
        result = {}
        for k, v in self.items():
            if isinstance(v, NDict):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    @classmethod
    def fromkeys(cls, seq, value=None):
        return cls(dict.fromkeys(seq, value))

if __name__ == '__main__':
    ndict = NDict({'a': {'b': 1, 'c': 2}, 'd': 3})
    print(ndict.unfold())
    print(ndict.to_dict())
    print(ndict.fromkeys(['a', 'b', 'c'], 0))
    print(ndict.copy())
    print(ndict.get('a'))
    print(ndict.get('a', 'default'))
    print(ndict.get('e'))
    print(ndict.get('e', 'default'))
    print(ndict.get('e', 'default'))