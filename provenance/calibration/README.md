# Preserved calibration artifacts

These compact files preserve the previously measured gamma/fallback calibration
arms and their checksums. They are historical evidence for choosing
`budget_gamma=1.1`; they are not the new three-replicate matrix.

The frozen PO calibration observations were:

- overload/default: 223.776 ktok/s, 83.093% SLO attainment;
- budget-attention, gamma 1.0: 222.148 ktok/s, 83.857% attainment;
- budget-attention, gamma 1.1: 226.817 ktok/s, 84.233% attainment;
- budget-attention, gamma 1.2: 227.426 ktok/s, 83.665% attainment.

Gamma 1.1 was selected because it was the best balanced common-cohort result;
the higher gamma 1.2 aggregate was partly due to closed-loop admission drift.

The historical runner used an external Redis observer to record engine queue
telemetry. The Router launch in every artifact records `router_reads_redis=false`;
Redis was not a routing input. The standalone reproduction runner in this
repository removes that observer and does not start Redis. Raw request ledgers
and logs remain outside the repository; the index records their hashes and
original local paths.
