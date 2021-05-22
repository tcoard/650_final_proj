import os
import io
import pickle
from dataclasses import dataclass, field
from typing import Optional, Literal

# maybe make the chars positionally agnostic
MODE = Literal["w", "wb", "r", "rb", "r+", "r+b", "a", "ab"]


@dataclass
class LargeList:
    file_name: str
    mode: MODE = "r+b"
    chunk_size: int = 0  # only use for write. TODO enforce this
    is_chunked: bool = False  # only use for read. TODO enforce this
    # is_large_list_format: bool = False # only use for read. TODO enforce this
    _chunked_data: list = field(default_factory=list)
    _real_mode: MODE = field(init=False)
    file_obj: io.TextIOWrapper = field(init=False)

    def __post_init__(self):
        if self.mode in ["w", "wb"]:
            self._real_mode = "ab"
            if os.path.exists(self.file_name):
                os.remove(self.file_name)
        else:
            if not self.mode.endswith("b"):
                self._real_mode = self.mode + "b"
            else:
                self._real_mode = self.mode

        self.file_obj = open(self.file_name, self._real_mode)

        # if self._real_mode in ['ab', 'r+b']:
        #     pickle.dump(self.chunk_size, self.file_obj)
        # if self._real_mode in ['rb', 'r+b'] and self.is_large_list_format:
        #     self.chunk_size = pickle.load(self.file_obj)["metadata_chunk_size"]

        # if read and is_chunked, get first line to read chunk size

    def __enter__(self):
        return self#.file_obj

    def __exit__(self, type, value, traceback):
        if self.chunk_size and self._chunked_data:
            pickle.dump(self._chunked_data, self.file_obj)
        self.file_obj.close()

    def write(self, data):
        if self._real_mode == "rb":
            raise ValueError("Cannot read to write only list")
            # if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >= max_mem: #10000000:
            # with open(self.file_name, self.mode) as out:
        if self.chunk_size:
            self._chunked_data.append(data)
            if len(self._chunked_data) == self.chunk_size:
                pickle.dump(self._chunked_data, self.file_obj)
                self._chunked_data = []
        else:
            pickle.dump(data, self.file_obj)

    def read(self):
        if self._real_mode == "ab":
            raise ValueError("Cannot write to read only list")

        try:
            while True:
                if self.is_chunked:
                    for obj in pickle.load(self.file_obj):
                        yield obj
                else:
                    yield pickle.load(self.file_obj)
        except EOFError:
            pass
