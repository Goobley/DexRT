from scipy import special
import numpy as np
 
def voigt_H(a, v):
    z = (v + 1j * a)
    return special.wofz(z)

if __name__ == "__main__":
    n_samples = 11
    a = np.linspace(0, 15, n_samples)
    v = np.linspace(-600, 600, n_samples)

    result = np.zeros((n_samples, n_samples), dtype=np.complex128)

    for ia, aa in enumerate(a):
        for iv, vv in enumerate(v):
            result[ia, iv] = voigt_H(aa, vv)

    str_rep = "{"
    for ia in range(a.shape[0]):
        str_rep += "{"
        for iv in range(v.shape[0]):
            str_rep += f"{result[ia, iv].real} + {result[ia, iv].imag}i"
            if iv != v.shape[0] - 1:
                str_rep += ", "
        str_rep += "}"
        if ia != a.shape[0] - 1:
            str_rep += ","
        str_rep += "\n"
    str_rep += "};"