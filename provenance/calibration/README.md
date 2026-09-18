# Preserved calibration artifacts

These compact files preserve the previously measured gamma/fallback calibration
arms and their checksums. They are historical evidence for choosing
`budget_gamma=1.1`; they are not the new three-replicate matrix.

The historical runner used an external Redis observer to record engine queue
telemetry. The Router launch in every artifact records `router_reads_redis=false`;
Redis was not a routing input. The standalone reproduction runner in this
repository removes that observer and does not start Redis. Raw request ledgers
and logs remain outside the repository; the index records their hashes and
original local paths.
