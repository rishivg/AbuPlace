"""GPU port of cong_relax_v2 polish (proxy = wl + 0.5*den + 0.5*cong); serial
across macros, parallelism is intra-macro across the 24 candidate poses;
float64 to track C double precision modulo FP-order drift."""
