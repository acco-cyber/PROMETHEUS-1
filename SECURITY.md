# Security policy

Report a vulnerability privately to the repository owner rather than opening a public issue. Do
not include source-data credentials or private series; the registered benchmark requires no secrets.

Model checkpoints and downloaded archives are untrusted binary inputs. Reproduce them only from the
recorded official sources and verify the manifest hashes. The release does not load arbitrary user
checkpoints through its CLI.
