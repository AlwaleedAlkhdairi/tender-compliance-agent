# Bid — AlphaTech Networks — RFP-2026-014

> Synthetic bid created for the SDAIA Academy capstone project. AlphaTech
> Networks is a fictional company.

## Company Profile and Certifications

AlphaTech Networks is a systems integrator headquartered in Riyadh with 480
employees. We hold a valid ISO/IEC 27001:2022 certification issued by
GulfCert Assurance, certificate AT-27001-0042, valid until 2027-11-30. We
also hold ISO 9001 and are a Tier-1 partner of our proposed switching
vendor. A copy of each certificate is attached in Annex A.

## Local Support Presence

AlphaTech operates a 24/7 Network Operations Center and service center in
Riyadh (Olaya district) staffed by 35 support engineers on rotating shifts,
with a spare-parts depot in the same facility and a secondary depot in
Dammam.

## Reference Projects

1. Saudi Meridian Bank — dual data center network refresh, 620 endpoints,
   completed 2024, reference: Eng. F. Al-Harbi, Head of Infrastructure.
2. Tabuk Health Cluster — campus network and DC upgrade, 540 endpoints,
   completed 2023, reference: IT Director office.
3. Red Sand Logistics — two server halls, leaf-spine rebuild, completed
   2025, reference: CIO office.
4. Central Grid Operator — monitoring platform rollout, completed 2022.
5. Najd University — campus core replacement, completed 2023.

## Proposed Technical Solution

### Architecture and Redundancy

We propose a leaf-spine fabric with redundant spine pairs in each server
hall, dual supervisors on core chassis, dual power feeds (A/B), and
redundant 100G uplinks in active-active pairs. There is no single point of
failure at core or aggregation; the failure-mode analysis and full network
diagrams are provided in Annex B.

### Equipment Performance

Core: 2x AT-CX9500 chassis per hall, 100G uplinks, 12.8 Tbps switching
capacity each. Access: 46x AT-LX4800 leaf switches with 25G server ports
(48x25G + 8x100G per switch). All models are current-generation and covered
by a 48-month manufacturer warranty.

### Security and Segmentation

The design implements macro-segmentation (production / management / DMZ
VRFs) and micro-segmentation through group-based policies enforced at the
leaf layer, with east-west inspection for inter-zone flows and a documented
policy model included in Annex C.

### Monitoring and Management

We include our AT-Fabric Manager platform providing configuration
management, streaming telemetry, alerting, capacity reporting and role-based
access control, deployed on-premises in high availability.

## Implementation Plan and Migration

We propose five phases over 100 calendar days from contract signature:
survey and design freeze (day 1-15), staging and factory tests (16-40),
hall A migration (41-65), hall B migration (66-90), monitoring cutover and
acceptance (91-100). Server migrations run in 14 waves; each wave is
executed inside a maximum 3-hour downtime window with a rehearsed rollback
procedure and a cutover rehearsal before each hall migration.

## Project Team

Delivery team of 12, including three CCIE-certified engineers (CCIE
#41022, #47810, #52288) and a PMP-certified project manager who led the
Saudi Meridian Bank refresh. CVs are provided in Annex D.

## Support Services and SLA

Three-year support with 24/7 coverage from the Riyadh NOC: Priority-1
response within 30 minutes, on-site engineer within 4 hours, advance
hardware replacement next business day, quarterly service reviews, and a
named technical account manager.

## Warranty

All supplied hardware carries a 48-month manufacturer-backed warranty from
provisional acceptance, exceeding the 36-month requirement.

## Financial Proposal

Total cost of ownership for supply, installation, migration and three years
of support: **SAR 8,950,000**, broken into milestones: 20% mobilization, 40%
delivery of hardware, 25% acceptance of both halls, 15% spread across the
support period. Bid validity: 120 days. Bid security of 2% (SAR 179,000) is
attached as a bank guarantee.
