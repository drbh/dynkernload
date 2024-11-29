import ctypes
import numpy as np
from typing import Dict, Any


def parse_signature(sig: str) -> list:
    type_map = {"float*": np.ctypeslib.ndpointer(dtype=np.float32), "int": ctypes.c_int}
    return [type_map[t] for t in sig.split(",")]


def load_kernel_lib(library_path: str, kernel_name: str) -> Dict[str, Any]:
    lib = ctypes.CDLL(library_path)
    sig_func = getattr(lib, f"get_{kernel_name}_signature")
    sig_func.restype = ctypes.c_char_p
    sig = sig_func().decode()
    kernel_func = getattr(lib, f"{kernel_name}_kernel")
    kernel_func.argtypes = parse_signature(sig)
    return kernel_func


def main():
    kernel_name = "matmul"
    kernel = load_kernel_lib(f"lib{kernel_name}.so", kernel_name)
    N = 10
    
    # make sure a seed is set for reproducibility
    np.random.seed(0)

    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    kernel(A, B, C, N)

    print("C = A @ B")
    # round all elements to 4 decimal places
    print(np.round(C, 4))

    np.testing.assert_allclose(C, np.matmul(A, B), atol=1e-6)
    print("Success!")


if __name__ == "__main__":
    main()
