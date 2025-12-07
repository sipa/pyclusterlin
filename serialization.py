# Copyright (c) 2024-2025 Pieter Wuille
# Distributed under the MIT software license, see the accompanying
# file LICENSE or http://www.opensource.org/licenses/mit-license.php.

"""Python reimplementation of the Bitcoin Core serialization framework."""

from typing import BinaryIO, Generic, TypeVar
import io

T = TypeVar('T')

class Formatter(Generic[T]):
    """Parent class for all serializers/deserializers."""

    def encode(self, _stream: BinaryIO, _arg: T) -> None:
        """Append the serialization of _arg to _stream. Must be overridden."""
        assert False

    def decode(self, _stream: BinaryIO) -> T | None:
        """Deserialize an object from the serialization in _stream."""
        assert False

    def serialize(self, arg: T) -> bytes:
        """Serialize arg to bytes."""
        stream = io.BytesIO()
        self.encode(stream, arg)
        return stream.getvalue()

    def deserialize(self, enc: bytes) -> T | None:
        """Deserialize an object from enc, and return it, or None in case of failure."""
        stream = io.BytesIO(enc)
        return self.decode(stream)

class VarIntFormatter(Formatter[int]):
    """Formatter class for VARINTs."""

    def encode(self, stream: BinaryIO, arg: int) -> None:
        """Append the VARINT encoding of arg to stream."""
        first = True
        tmp = bytearray()
        while True:
            tmp.append((arg & 0x7F) | (0x00 if first else 0x80))
            if arg <= 0x7F:
                break
            first = False
            arg = (arg >> 7) - 1
        tmp.reverse()
        stream.write(tmp)

    def decode(self, stream: BinaryIO) -> int | None:
        """Pop and decode a VARINT from reversed encoding rstream."""
        ret = 0
        while True:
            bval = stream.read(1)
            if not bval:
                return None
            val = ord(bval)
            ret = (ret << 7) | (val & 0x7F)
            if val & 0x80:
                ret += 1
            else:
                break
        return ret

varint_formatter = VarIntFormatter()

class SignedVarIntFormatter(VarIntFormatter):
    """Formatter class for signed integers, using transformed VARINTs."""

    @staticmethod
    def signed_to_unsigned(arg: int) -> int:
        """Bijectively map signed integers to unsigned integer."""
        if arg < 0:
            return ((-(arg + 1)) << 1) + 1
        return arg << 1

    @staticmethod
    def unsigned_to_signed(arg: int) -> int:
        """Bijectively map unsigned integers to signed integers."""
        if arg & 1:
            return -(arg >> 1) - 1
        return arg >> 1

    def encode(self, stream: BinaryIO, arg: int) -> None:
        super().encode(stream, self.signed_to_unsigned(arg))

    def decode(self, stream: BinaryIO) -> int | None:
        ret = super().decode(stream)
        if ret is None:
            return None
        return self.unsigned_to_signed(ret)

signed_varint_formatter = SignedVarIntFormatter()
