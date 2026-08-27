# Bid — BetaGrid Solutions — RFP-2026-014

> Synthetic bid created for the SDAIA Academy capstone project. BetaGrid
> Solutions is a fictional company.

## Company Profile and Certifications

BetaGrid Solutions is a fast-growing network integrator with 95 employees
headquartered in Jeddah. Our ISO/IEC 27001 certification is currently in
progress with the certification audit scheduled for Q4 2026; a letter from
the certification body confirming the audit booking is attached. We hold
ISO 9001:2015.

## Local Support Presence

BetaGrid operates a support office in Jeddah staffed by 9 engineers,
providing support during business hours (Sunday-Thursday, 8:00-18:00) with
an on-call engineer for emergencies outside these hours.

## Reference Projects

1. Coastal Retail Group — head-office network refresh and one server room,
   180 endpoints, completed 2024, reference: IT Manager.
2. (In progress) Western Logistics Hub — data center network build, 450
   endpoints, expected completion late 2026.

## Proposed Technical Solution

### Architecture and Redundancy

We propose a collapsed-core design with a redundant core switch pair shared
across both halls and single aggregation switches per hall connected by dual
uplinks. Core power is dual-fed; aggregation switches use single power
supplies with a cold spare held in our Jeddah depot.

### Equipment Performance

Core: 2x BG-C7200 switches with 100G uplink modules, 6.4 Tbps capacity.
Access: 40x BG-A3600 switches with 25G server ports. Datasheets are
attached in Annex 2.

### Security and Segmentation

VLAN-based macro-segmentation separating production, management and DMZ
zones with ACL enforcement at the core. Micro-segmentation is offered as an
optional add-on module at additional cost, not included in the base price.

### Monitoring and Management

We offer the open-source LibreView monitoring stack configured for SNMP
polling and syslog alerting, hosted on a single virtual machine provided by
NRUC. Role-based access control is supported through local user groups.

## Implementation Plan and Migration

We propose a 150-calendar-day schedule from contract signature: design
(30 days), procurement and staging (60 days), migration of both halls
(45 days), acceptance (15 days). Server migrations run in 8 waves of up to
6-hour downtime windows each, scheduled on weekends. A rollback plan is
described at a high level in Annex 3.

## Project Team

Delivery team of 6, including one CCIE-certified engineer (CCIE #58455) and
two CCNP-certified engineers. The proposed project manager has delivered
office-network projects for the last three years.

## Support Services and SLA

Three-year support from our Jeddah office: Priority-1 response within 4
business hours, on-site within next business day, spare parts shipped from
Jeddah within 48 hours. Support outside business hours is handled by the
on-call engineer on a best-effort basis.

## Warranty

Supplied hardware carries the manufacturer's standard 24-month warranty. An
extended warranty to 36 months can be purchased as an option at 4% of
hardware value per additional year.

## Financial Proposal

Total cost of ownership for supply, installation, migration and three years
of support: **SAR 6,200,000** (base configuration), milestones: 30%
mobilization, 50% hardware delivery, 20% acceptance. Bid validity: 90 days.
Bid security of 2% is attached.
