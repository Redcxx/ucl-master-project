import os
import re
from re import Pattern

_line_match = re.compile(r'line|clean', re.IGNORECASE)
_line_sub = re.compile(r'line|clean|_', re.IGNORECASE)

_fill_match = re.compile(r'colour|col|fill', re.IGNORECASE)
_fill_sub = re.compile(r'colour|col|fill|_', re.IGNORECASE)

_tied_match = re.compile(r'td|tiedown', re.IGNORECASE)
_tied_sub = re.compile(r'td|tiedown|_', re.IGNORECASE)


class Identifier:

    def __init__(self, path):
        self.path = path
        self._components = Identifier._find_components(self.path)

    def tag(self):
        return '-'.join(self._components)

    def __hash__(self):
        return hash(self.tag())

    def tied_matched(self):
        return self._match(_tied_match)

    def fill_matched(self):
        return self._match(_fill_match)

    def line_matched(self):
        return self._match(_line_match)

    def _match(self, pattern: Pattern):
        return pattern.search(self._components[-1]) is not None

    def remove_tied_attrs(self):
        return self._remove_attrs(_tied_sub)

    def remove_fill_attrs(self):
        return self._remove_attrs(_fill_sub)

    def remove_line_attrs(self):
        return self._remove_attrs(_line_sub)

    def _remove_attrs(self, pattern: Pattern):
        return '-'.join(self._components[:-1] + [pattern.sub('', self._components[-1])])

    def __str__(self):
        return str(self._components)

    @staticmethod
    def _find_components(p):
        p = os.path.normpath(p)
        p_parts = p.split(os.sep)

        sh_indices = [i for i in range(len(p_parts)) if p_parts[i].startswith('SH')]
        if len(sh_indices) == 0:
            raise ValueError(f'SHOT folder not found: {p_parts}')

        sh_i = sh_indices[-1]  # innermost shot folder

        id_comps = []

        # shot name
        shot_name = p_parts[sh_i]
        id_comps.append(shot_name)

        # character/version name follow after shot name
        char_name = p_parts[sh_i + 1]
        id_comps.append(char_name)

        # may have type name such as cleanup, spotkey, and tiedown after character name
        if len(p_parts) > sh_i + 2:
            type_name = p_parts[sh_i + 2].split('_', 1)[-1]  # remove the number in front if any
            id_comps.append(type_name)

        # may have character parts name such as 01_SHA_Connie_fill, and 06_LINE_CONNIE_UPPERBODY
        if len(p_parts) > (sh_i + 3):
            stage_name = p_parts[sh_i + 3].split('_', 1)[-1]  # remove the number in front if any
            id_comps.append(stage_name)

        return [comp.lower() for comp in id_comps]


class ImagePath:
    def __init__(self, path):
        self.norm_path = os.path.normpath(os.path.abspath(path))
        self.identifier = Identifier(self.norm_path)

    def __hash__(self):
        return hash(self.norm_path)

    def __str__(self):
        return self.norm_path

    def __repr__(self):
        return f'ImagePath({self.norm_path})'
