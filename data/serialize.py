from functools import partial
import numpy as np
from dataclasses import dataclass

def vec_num2repr(val, base, prec, max_val):
    """
    Convert numbers to a representation in a specified base with precision.

    Parameters:
    - val (np.array): The numbers to represent.
    - base (int): The base of the representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - max_val (float): The maximum absolute value of the number.

    Returns:
    - tuple: Sign and digits in the specified base representation.
    
    Examples:
        With base=10, prec=2:
            0.5   ->    50
            3.52  ->   352
            12.5  ->  1250
    """
    base = float(base)
    bs = val.shape[0]
    sign = 1 * (val >= 0) - 1 * (val < 0)
    val = np.abs(val)
    max_bit_pos = int(np.ceil(np.log(max_val) / np.log(base)).item())

    before_decimals = []
    for i in range(max_bit_pos):
        digit = (val / base**(max_bit_pos - i - 1)).astype(int)
        before_decimals.append(digit)
        val -= digit * base**(max_bit_pos - i - 1)

    before_decimals = np.stack(before_decimals, axis=-1)

    if prec > 0:
        after_decimals = []
        for i in range(prec):
            digit = (val / base**(-i - 1)).astype(int)
            after_decimals.append(digit)
            val -= digit * base**(-i - 1)

        after_decimals = np.stack(after_decimals, axis=-1)
        digits = np.concatenate([before_decimals, after_decimals], axis=-1)
    else:
        digits = before_decimals
    return sign, digits

def vec_repr2num(sign, digits, base, prec, half_bin_correction=True):
    """
    Convert a string representation in a specified base back to numbers.

    Parameters:
    - sign (np.array): The sign of the numbers.
    - digits (np.array): Digits of the numbers in the specified base.
    - base (int): The base of the representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - half_bin_correction (bool): If True, adds 0.5 of the smallest bin size to the number.

    Returns:
    - np.array: Numbers corresponding to the given base representation.
    """
    base = float(base)
    bs, D = digits.shape
    digits_flipped = np.flip(digits, axis=-1)
    powers = -np.arange(-prec, -prec + D)
    val = np.sum(digits_flipped/base**powers, axis=-1)

    if half_bin_correction:
        val += 0.5/base**prec

    return sign * val

@dataclass
class SerializerSettings:
    """
    Settings for serialization of numbers.

    Attributes:
    - base (int): The base for number representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - signed (bool): If True, allows negative numbers. Default is False.
    - fixed_length (bool): If True, ensures fixed length of serialized string. Default is False.
    - max_val (float): Maximum absolute value of number for serialization.
    - time_sep (str): Separator for different time steps.
    - bit_sep (str): Separator for individual digits.
    - plus_sign (str): String representation for positive sign.
    - minus_sign (str): String representation for negative sign.
    - half_bin_correction (bool): If True, applies half bin correction during deserialization. Default is True.
    - decimal_point (str): String representation for the decimal point.
    """
    base: int = 10
    prec: int = 3
    signed: bool = True
    fixed_length: bool = False
    max_val: float = 1e7
    time_sep: str = ' ,'
    bit_sep: str = ' '
    plus_sign: str = ''
    minus_sign: str = ' -'
    half_bin_correction: bool = True
    decimal_point: str = ''
    missing_str: str = ' Nan'

def serialize_arr(arr, settings: SerializerSettings):
    """
    Serialize an array of numbers (a time series) into a string based on the provided settings.

    Parameters:
    - arr (np.array): Array of numbers to serialize.
    - settings (SerializerSettings): Settings for serialization.

    Returns:
    - str: String representation of the array.
    """
    # max_val is only for fixing the number of bits in nunm2repr so it can be vmapped
    assert np.all(np.abs(arr[~np.isnan(arr)]) <= settings.max_val), f"abs(arr) must be <= max_val,\
         but abs(arr)={np.abs(arr)}, max_val={settings.max_val}"
    
    if not settings.signed:
        assert np.all(arr[~np.isnan(arr)] >= 0), f"unsigned arr must be >= 0"
        plus_sign = minus_sign = ''
    else:
        plus_sign = settings.plus_sign
        minus_sign = settings.minus_sign
    
    vnum2repr = partial(vec_num2repr,base=settings.base,prec=settings.prec,max_val=settings.max_val)
    sign_arr, digits_arr = vnum2repr(np.where(np.isnan(arr),np.zeros_like(arr),arr))
    ismissing = np.isnan(arr)
    
    def tokenize(arr):
        return ''.join([settings.bit_sep+str(b) for b in arr])
    
    bit_strs = []
    for sign, digits,missing in zip(sign_arr, digits_arr, ismissing):
        if not settings.fixed_length:
            # remove leading zeros
            nonzero_indices = np.where(digits != 0)[0]
            if len(nonzero_indices) == 0:
                digits = np.array([0])
            else:
                digits = digits[nonzero_indices[0]:]
            # add a decimal point
            prec = settings.prec
            if len(settings.decimal_point):
                digits = np.concatenate([digits[:-prec], np.array([settings.decimal_point]), digits[-prec:]])
        digits = tokenize(digits)
        sign_sep = plus_sign if sign==1 else minus_sign
        if missing:
            bit_strs.append(settings.missing_str)
        else:
            bit_strs.append(sign_sep + digits)
    bit_str = settings.time_sep.join(bit_strs)
    bit_str += settings.time_sep # otherwise there is ambiguity in number of digits in the last time step
    return bit_str

def deserialize_str(bit_str, settings: SerializerSettings, ignore_last=False, steps=None):
    """
    Deserialize a string into an array of numbers (a time series) based on the provided settings.

    Parameters:
    - bit_str (str): String representation of an array of numbers.
    - settings (SerializerSettings): Settings for deserialization.
    - ignore_last (bool): If True, ignores the last time step in the string (which may be incomplete due to token limit etc.). Default is False.
    - steps (int, optional): Number of steps or entries to deserialize.

    Returns:
    - None if deserialization failed for the very first number, otherwise 
    - np.array: Array of numbers corresponding to the string.
    """
    # ignore_last is for ignoring the last time step in the prediction, which is often a partially generated due to token limit
    orig_bitstring = bit_str
    bit_strs = bit_str.split(settings.time_sep)
    # remove empty strings
    bit_strs = [a for a in bit_strs if len(a) > 0]
    if ignore_last:
        bit_strs = bit_strs[:-1]
    if steps is not None:
        bit_strs = bit_strs[:steps]
    vrepr2num = partial(vec_repr2num,base=settings.base,prec=settings.prec,half_bin_correction=settings.half_bin_correction)
    max_bit_pos = int(np.ceil(np.log(settings.max_val)/np.log(settings.base)).item())
    sign_arr = []
    digits_arr = []
    try:
        for i, bit_str in enumerate(bit_strs):
            if bit_str.startswith(settings.minus_sign):
                sign = -1
            elif bit_str.startswith(settings.plus_sign):
                sign = 1
            else:
                assert settings.signed == False, f"signed bit_str must start with {settings.minus_sign} or {settings.plus_sign}"
            bit_str = bit_str[len(settings.plus_sign):] if sign==1 else bit_str[len(settings.minus_sign):]
            if settings.bit_sep=='':
                bits = [b for b in bit_str.lstrip()]
            else:
                bits = [b[:1] for b in bit_str.lstrip().split(settings.bit_sep)]
            if settings.fixed_length:
                assert len(bits) == max_bit_pos+settings.prec, f"fixed length bit_str must have {max_bit_pos+settings.prec} bits, but has {len(bits)}: '{bit_str}'"
            digits = []
            for b in bits:
                if b==settings.decimal_point:
                    continue
                # check if is a digit
                if b.isdigit():
                    digits.append(int(b))
                else:
                    break
            #digits = [int(b) for b in bits]
            sign_arr.append(sign)
            digits_arr.append(digits)
    except Exception as e:
        print(f"Error deserializing {settings.time_sep.join(bit_strs[i-2:i+5])}{settings.time_sep}\n\t{e}")
        print(f'Got {orig_bitstring}')
        print(f"Bitstr {bit_str}, separator {settings.bit_sep}")
        # At this point, we have already deserialized some of the bit_strs, so we return those below
    if digits_arr:
        # add leading zeros to get to equal lengths
        max_len = max([len(d) for d in digits_arr])
        for i in range(len(digits_arr)):
            digits_arr[i] = [0]*(max_len-len(digits_arr[i])) + digits_arr[i]
        return vrepr2num(np.array(sign_arr), np.array(digits_arr))
    else:
        # errored at first step
        return None
