# Archived CTR evaluation

`clean_target_rates_all_configurations_batch256_legacy_threshold.csv` is the
historical batch-size-256 evaluation. Its `status` fields were produced with an
older, tighter clean-accuracy tolerance and are not authoritative under the
current strict `< 0.2` percentage-point rule.

The official result is
`../clean_target_rates_all_configurations.csv`, evaluated with batch size 128.
The archived file is retained only for provenance and batch-size sensitivity
checks; it must not be used to populate the paper tables.
