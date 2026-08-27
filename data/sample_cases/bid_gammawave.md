# Bid — GammaWave Technologies — RFP-2026-014

> Synthetic bid created for the SDAIA Academy capstone project. GammaWave
> Technologies is a fictional company.

## Company Profile and Certifications

GammaWave Technologies is a Riyadh-based integrator with 210 employees. We
hold a valid ISO/IEC 27001:2022 certification issued by Arabian Standards
Certification, certificate GW-IS-2210, valid until 2026-12-15, attached in
Appendix I, together with ISO 9001 and ISO 20000-1.

## Local Support Presence

GammaWave runs a 24/7 service desk and service center in Riyadh (Al Malaz)
staffed by 18 engineers across three shifts, with spare parts stocked
locally for all proposed models.

## Reference Projects

1. Peninsula Insurance — data center network replacement, 430 endpoints,
   completed 2023, reference: Infrastructure Manager.
2. Qassim Agri-Industries — two server halls, core and access refresh,
   completed 2024, reference: Head of IT.
3. Eastern Media Group — campus network upgrade, 460 endpoints, completed
   2022, reference: CTO office.

## Proposed Technical Solution

### Architecture and Redundancy

We propose a two-tier design with a redundant core pair per hall and
stacked access switches, dual power throughout, and redundant uplinks.
Aggregation is collapsed into the core. A network diagram and failure-mode
table are included in Appendix II; the design has no single point of
failure at the core layer.

### Equipment Performance

Core: 4x GW-CS8400 switches (two per hall) with 40G uplinks upgradable to
100G through optional modules quoted separately in Appendix V. Access: 44x
GW-AS2500 switches with 25G server ports, 3.6 Tbps capacity per core
switch.

### Security and Segmentation

Macro-segmentation via VRFs for production, management and DMZ, with
micro-segmentation delivered through our partner firewall vendor for
east-west control, included in the base price. The policy model is
documented in Appendix III.

### Monitoring and Management

Monitoring and management is provided through our subcontractor NetSight
Arabia using their NS-Vision platform (telemetry, alerting, configuration
management, capacity dashboards, role-based access). NetSight Arabia would
hold the operational monitoring contract under our responsibility.

## Implementation Plan and Migration

A 115-calendar-day plan from contract signature: design freeze (day 1-20),
staging (21-50), hall A migration (51-80), hall B migration (81-105),
acceptance (106-115). Migrations run in 12 waves within 4-hour downtime
windows, each with a documented rollback procedure; a cutover rehearsal is
performed before hall A only.

## Project Team

Delivery team of 9, including two CCIE-certified engineers (CCIE #44120,
#50934) and a PRINCE2-certified project manager who managed the Peninsula
Insurance replacement.

## Support Services and SLA

Three-year support from the Riyadh service center: Priority-1 response
within 1 hour (24/7), on-site within 6 hours, advance replacement within
two business days, semi-annual service reviews. Monitoring-platform support
is provided by NetSight Arabia under back-to-back SLA.

## Warranty

All supplied hardware carries a 36-month manufacturer-backed warranty from
provisional acceptance.

## Financial Proposal

Total cost of ownership for supply, installation, migration and three years
of support: **SAR 7,400,000**, milestones: 25% mobilization, 40% delivery,
20% acceptance, 15% across support. The optional 100G core upgrade modules
are quoted separately at SAR 640,000. Bid validity: 90 days. Bid security
of 2% (SAR 148,000) is attached.
