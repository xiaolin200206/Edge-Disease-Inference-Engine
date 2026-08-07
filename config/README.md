# Dataset configurations

`data_orig_abs.yaml` and `data_clean.yaml` differ only in the `val`
key: the first points at the original validation partition, the second
at the partition with the 32 leak-implicated images removed (Table 1).

Both originally carried absolute paths from the machine they were
written on. Those have been replaced with paths relative to this
directory. The filename `data_orig_abs.yaml` is kept because
`args.yaml` in the released training logs refers to it by name; the
`_abs` suffix is now historical and the file is relative.
