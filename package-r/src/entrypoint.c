#include <R.h>
#include <Rinternals.h>

extern void R_init_perpetual_rust_extendr(DllInfo *dll);

void R_init_perpetual(DllInfo *dll) {
    R_init_perpetual_rust_extendr(dll);
}
