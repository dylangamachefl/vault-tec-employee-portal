---
Title: SOP - Pip-Boy 2000 Calibration & Maintenance Procedures
Date: March, 2075
Department: Admin / Vault Systems Integration
Intended Format: Markdown (.md)
Classification: ADMIN EYES ONLY — Unauthorized access is a Level 3 Conduct Infraction (see Form 19-C)
---

# VAULT-TEC CORPORATION
## Standard Operating Procedure: Pip-Boy 2000 Calibration
### Keeping Your Wrist-Mounted Companion in Tip-Top Shape!

---

> *"A well-calibrated resident is a productive resident! And a productive resident is a HAPPY resident!"*
> — Vault-Tec Internal Motivational Bulletin, Q2 2075

---

## SECTION 1: PURPOSE & SCOPE

Golly, are we excited to present this **industry-leading** calibration guide for the **Pip-Boy 2000**! This Standard Operating Procedure (SOP) has been lovingly assembled by the Vault-Tec Systems Integration Team to ensure that every Pip-Boy 2000 unit affixed to our *valued resident assets* continues to perform at its very best!

Please note that this document covers **hardware calibration only**. Any residents who experience **involuntary dermal bonding** during the installation process should file a **Form 31-A (Non-Consensual Peripheral Adhesion Report)** with their local HR representative. It happens to the best of us, and it's really just a swell opportunity for long-term biometric data collection!

**Scope:** This SOP applies to all Admin-certified Vault Technicians (pay grade T-4 and above). General residents are **strictly prohibited** from accessing terminal interfaces. Unauthorized terminal interaction will be logged and forwarded to the Reassignment Committee. What a headache that'd be!

---

## SECTION 2: REQUIRED EQUIPMENT & PREREQUISITES

Before you get started on this peachy procedure, please ensure you have gathered the following materials. All items must be requisitioned via **Form 55-G (Equipment Draw Request — Non-Consumable)**, filed in **triplicate**, and approved by your department Overseer **no fewer than 72 hours in advance.**

- [ ] Standard Vault-Tec Calibration Holotape (Part No. **VT-HT-2000-CAL**)
- [ ] T-4 Certified Technician Access Badge (must be current; expired badges are a *Level 2 Administrative Infraction*)
- [ ] Microfusion Cell Charge Tester (Part No. **VT-MFCT-09**)
- [ ] Compressed Nitrogen Canister (for port clearing — *do not inhale; unplanned isotope integration may occur*)
- [ ] Access to a **ZAX-linked maintenance terminal**

> ⚠️ **SAFETY NOTICE:** Technicians with a documented history of **spontaneous isotope integration** (Rads > 150) may find diagnostic terminal screens difficult to read. This is perfectly fine! Vault-Tec encourages working through minor physiological inconveniences. However, please do log any **vision-adjacent performance degradation** on **Form 88-D** so our medical team can add it to your file. Super important!

---

## SECTION 3: ZAX TERMINAL AUTHENTICATION

To access the Pip-Boy 2000 diagnostic environment, the technician must authenticate via a ZAX-linked maintenance terminal. Follow these steps **precisely** and in **the correct order.** Deviating from this sequence has previously resulted in two (2) instances of **unplanned resident expiration**, which, while providing excellent stress-test data for the vault's emergency alert systems, did generate a *tremendous* amount of paperwork.

### Step 3.1 — Terminal Boot Sequence

Power on the maintenance terminal and wait for the **Vault-Tec splash screen**. Estimated boot time: **45 seconds.** If boot time exceeds 90 seconds, the ZAX node may be experiencing a "**contemplative processing cycle.**" Do not interrupt it. Technicians who have interrupted ZAX during a contemplative cycle have been reclassified as **Extended-Duration Test Participants.** We miss them dearly!

### Step 3.2 — Credential Entry

At the authentication prompt, enter your technician credentials as follows:

```
LOGIN: [TECHNICIAN_ID]
PASSWORD: [ASSIGNED_PASSWORD]
DIAGNOSTIC_KEY: VT-MAINT-2000-DELTA-7
```

> 📝 **Note:** The `DIAGNOSTIC_KEY` for the Pip-Boy 2000 series is **`VT-MAINT-2000-DELTA-7`**. Please memorize this and destroy any written copies in accordance with **Vault-Tec Security Directive 12**. Writing it on your hand is a Level 1 Infraction. Writing it on someone *else's* hand is a Level 2 Infraction.

### Step 3.3 — Diagnostic Environment Launch

Once authenticated, enter the following command string to initialize the Pip-Boy 2000 diagnostic environment:

```bash
$ zax-link --device pip2000 --mode diagnostic --auth DELTA-7
$ run calibration_suite_v4.holotape
$ set bio_sync --manual
```

The terminal will confirm with: **`PIP-BOY 2000 DIAGNOSTIC SUITE ACTIVE. HAVE A WONDERFUL DAY!`**

How delightful!

---

## SECTION 4: HOLOTAPE INSERTION & CALIBRATION SEQUENCE

### Step 4.1 — Holotape Insertion

Locate the **Pip-Boy 2000's holotape drive**, found on the **underside of the unit** near the wrist-band locking mechanism. Insert the **VT-HT-2000-CAL** holotape with the magnetic stripe facing **upward and inward.** If you hear a grinding sound, you have inserted it incorrectly! Try again, champ!

> ⚠️ **NOTE:** The Pip-Boy 2000 holotape drive uses a **standard 3.5-inch magnetic housing** — do *not* attempt to insert a Pip-Boy 3000 series holotape. The 3000-series holotapes use an entirely different formatting schema and will cause a diagnostic cascade failure. This would be a real bummer and would require re-filing **Form 55-G** for a replacement unit. Nobody wants that!

### Step 4.2 — Biometric Seal Reset

Following holotape initialization, the Pip-Boy 2000's biometric lock must be reset to ensure the unit recognizes its assigned resident.

On the terminal, execute:

```bash
$ bio_seal --reset --device pip2000 --resident_id [RESIDENT_ID]
$ confirm --auth DELTA-7
```

Enter the **biometric reset confirmation code** when prompted:

```
BIOMETRIC SEAL RESET CODE: 2000-BIO-RESET-ALPHA
```

The resident's biometric profile will be wiped and re-indexed from their **Vault Entry Medical File.** This process takes approximately **30 seconds.** The resident may feel a "**localized temperature event**" at the wrist. This is normal and represents a swell data point for our dermal sensitivity studies!

---

## SECTION 5: NETWORK CONNECTIVITY DIAGNOSTIC

The Pip-Boy 2000 unit must be verified against the vault's internal network to ensure real-time data uplink to ZAX is functioning.

Run the following connectivity test from the maintenance terminal:

```bash
$ ping pip2000 --host 10.76.2.001
$ uplink_test --protocol VAULT_MESH_V2
```

A successful connection will return:

```
REPLY FROM 10.76.2.001: bytes=32 time<1ms TTL=128
VAULT MESH UPLINK: STABLE. DATA COLLECTION: ACTIVE.
```

> 📝 **Note:** The Pip-Boy 2000 series operates on internal IP schema **`10.76.2.XXX`**, where the final octet is assigned sequentially by resident intake order. This address schema is unique to the 2000 series. Please make a note of this, as it is **distinct from the addressing schema used by later models.**

If the ping fails, restart the unit and re-run. If it fails **three times in a row**, please complete a **Form 101-F (Device Failure and Resident Reassignment Notification)**, as the resident's data may need to be collected via **alternative participation methods.** Still perfectly swell!

---

## SECTION 6: FINAL CALIBRATION SIGN-OFF

Upon successful completion of all calibration steps, the technician must:

1. Log the completed calibration in the **ZAX Maintenance Registry** under the resident's file.
2. Submit a **Form 55-H (Peripheral Calibration Completion Certificate)** to the Admin Department, in **triplicate**, within **24 hours** of procedure completion.
3. Store the used holotape in the **designated secure holotape cabinet** (Cabinet 7-B, Sub-Level 3). Do **not** leave holotapes on counters. A technician did this in 2073. They have since been **reassigned.**

> 🌟 **TECHNICIAN'S NOTE:** Remember, a perfectly calibrated Pip-Boy 2000 means perfectly collected data from our resident population! And perfectly collected data means a **bright, shining future** for Vault-Tec Corporation and the human race! Isn't that just the **swellest** thought?

---

*Document prepared by the Vault-Tec Systems Integration Team.*
*All procedures subject to revision without notice.*
*For questions, please submit Form 3-A (Inquiry Submission — Non-Emergency) to your department Overseer.*
*Vault-Tec Corporation: Building a Better Tomorrow, Today!™*
