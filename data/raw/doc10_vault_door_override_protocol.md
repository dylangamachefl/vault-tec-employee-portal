# VAULT-TEC CORPORATION
## Internal Administration & Facilities Management Division

---

**Title:** Vault Door Override Protocol and Access Logs
**Date:** December, 2076
**Department:** Admin / IT
**Intended Format:** Markdown (.md)
**Classification:** ADMIN-EYES ONLY — Unauthorized access constitutes a *swell* reason for immediate reassignment to Sector 7-G (Experimental)
**Document ID:** VT-ADMIN-4471-DOOR
**Supersedes:** VT-ADMIN-4470-DOOR (Recalled due to "enthusiastic over-use by residents")

---

> *"A locked door is a happy door! And a happy door means happy data!"*
> *— Vault-Tec Facilities Management Motto, circa 2072*

---

## SECTION 1: PURPOSE & CHEERFUL OVERVIEW

Greetings, Authorized Administrator!

What a **swell** responsibility you've been entrusted with! The Vault Gear Door — Vault-Tec Model VGD-9000 "Ol' Reliable" — is not merely a multi-ton, hydraulically-actuated, blast-proof portal. It is the **cornerstone of data integrity**. Every single valued participant sealed within this facility represents an **irreplaceable testing asset**, and it is your *peachy* privilege to ensure that not a single one of them wanders out prematurely and contaminates our longitudinal study parameters.

This document outlines the complete protocol for manual door cycling, emergency overrides, and the critically important access log procedures. Please read it thoroughly! Improper use of the override system has, in **three (3) documented prior cases**, resulted in what our Legal & Liability team cheerfully refers to as an "Unplanned Population Reduction Event." We'd love to keep that number right where it is!

---

## SECTION 2: AUTHORIZED PERSONNEL

Access to Vault Door override terminals is restricted to the following **tip-top** personnel only:

| Clearance Level | Role | Badge Color | Override Access |
|---|---|---|---|
| **ALPHA-1** | Overseer | Platinum | Full cycling, emergency seal, resident lockout |
| **ALPHA-2** | Deputy Overseer | Gold | Full cycling, emergency seal |
| **BETA-1** | Senior IT Administrator | Silver | Diagnostic access, log review |
| **BETA-2** | Facilities & Maintenance Lead | Silver | Manual crank access only (see §4.3) |
| **GAMMA** | General Employee | None — isn't that fun! | **NONE. ABSOLUTELY NONE. EVER.** |

> ⚠️ **IMPORTANT NOTICE:** Any GAMMA-clearance resident found in possession of this document should be greeted with a **warm smile** and immediately reported to HR for "voluntary schedule restructuring." Please submit Form **HR-22-C (Unauthorized Document Awareness Report)** in triplicate within four (4) business hours. It's the right thing to do!

---

## SECTION 3: STANDARD DOOR CYCLING PROCEDURE

### 3.1 — Routine Scheduled Cycling (Quarterly Maintenance)

Routine door cycles are scheduled **once per fiscal quarter** for lubrication, gear inspection, and — per the Behavioral Sciences Division request — to observe resident "door-proximity stress response metrics." The data is *just fantastic.*

**Step-by-step procedure:**

```
1. Navigate to the Primary Override Terminal (Level 1, Corridor A, behind the motivational mural).

2. Authenticate using dual-key biometric protocol:
   - Insert Administrator Keycard (VT-ADM-KEY-GOLD or PLATINUM tier)
   - Confirm thumbprint on BioScan Pad Unit 7
   - Speak the current monthly vocal passphrase clearly into the microphone.

   [ DECEMBER 2076 VOCAL PASSPHRASE: "Vault-Tec: Building a Brighter Tomorrow, Underground!" ]
   *** ROTATE PASSPHRASE FIRST BUSINESS DAY OF EACH MONTH — see §6.1 ***

3. The terminal will prompt: > CONFIRM CYCLING INTENTION (Y/N)
   Input: Y

4. Announce over vault PA system:
   "Attention, valued residents! Your scheduled quarterly door maintenance is underway!
    Please remain at least 50 (fifty) meters from the door threshold. Failure to comply
    may result in an exciting and data-rich participation opportunity!"

5. Engage hydraulic sequence from terminal:
   > RUN CYCLE_ROUTINE --override=ADMIN --log=TRUE --blast_seal=DISENGAGE

6. Confirm green status lights on panel board (all 12 lights must read GREEN).
   - Any RED lights indicate a fault. Do NOT proceed. Submit Form MAINT-7 immediately.
   - Any YELLOW lights are "peachy" — log them and proceed with caution.

7. Upon cycle completion, re-engage blast seal:
   > RUN SEAL_ENGAGE --verify=BIOMETRIC --log=TRUE

8. Log event in the Vault Door Access Ledger (physical binder, Overseer's office,
   shelf 3B) AND in the ZAX Mainframe log system (see Doc 11 for ZAX credentials).
```

---

### 3.2 — Emergency Override (Unscheduled Cycling)

Emergency overrides are authorized **only** under the following *exciting* scenarios:

- **Scenario A:** Authorized exterior personnel (Vault-Tec inspectors, corporate auditors) require entry. *(Note: As of October 2077, the probability of this scenario has been reassessed as "charmingly optimistic." Proceed with enthusiasm regardless.)*
- **Scenario B:** Critical supply delivery is imminent per pre-war procurement schedule.
- **Scenario C:** Overseer-authorized "population adjustment" requires expedited external relocation of one (1) or more residents. *(See HR Doc 5 for warm and friendly relocation procedures.)*
- **Scenario D:** Facilities determines a gear tooth inspection cannot wait until scheduled maintenance. Paperwork heavy. Worth it!

> ❌ **NON-AUTHORIZED scenarios include, but are not limited to:**
> - Resident requests, petitions, or "eerily coordinated group demonstrations"
> - "The surface probably isn't that bad" reasoning
> - A resident claiming to be "pretty sure the war is over by now"
> - Any scenario described by a non-ALPHA clearance individual as an "emergency"

**Emergency override terminal command string:**

```bash
> AUTHENTICATE --key=ALPHA --biometric=TRUE --vocal="[CURRENT PASSPHRASE]"
> OVERRIDE_ENGAGE --type=EMERGENCY --auth_code=[SEE SEALED ENVELOPE, OVERSEER'S DESK DRAWER 2]
> CYCLE_DOOR --direction=[OPEN|CLOSE] --blast_seal=[ENGAGE|DISENGAGE] --log=MANDATORY
> CONFIRM_SEAL_STATUS --verify=ALL_SYSTEMS
```

---

## SECTION 4: MANUAL CRANK PROCEDURE (Power Failure Protocol)

In the *perfectly manageable* event of a total vault power failure, the VGD-9000 can be operated via the **Manual Emergency Crank System (MECS)**. This is a swell backup option!

### 4.1 — Accessing the Manual Crank

The MECS crank housing is located behind Panel **B-7** in the sub-corridor adjacent to the door mechanism room. Access requires:

- BETA-2 clearance or above
- Physical key `VT-MECS-BRAVO-07` (stored in Overseer's key cabinet, hook 14)
- A **minimum of four (4) personnel** rated at standard adult fitness metrics — or six (6) personnel if the resident population has been experiencing "involuntary calorie restriction" for more than thirty (30) days

### 4.2 — Crank Operation

```
Turn clockwise to OPEN.
Turn counter-clockwise to CLOSE.

Required rotations for full open: 1,247 (one thousand, two hundred and forty-seven)
Estimated time to full open with 4 fit personnel: 6.5 hours
Estimated time with calorie-restricted personnel: Please just file Form MAINT-11 and wait.

DO NOT ATTEMPT PARTIAL CYCLING. A door that is 40% open is, from a data-integrity
standpoint, 100% a problem.
```

### 4.3 — Important Crank Safety Notice

> Three (3) incidents of "crank-related enthusiastic over-rotation" have been logged in the past eighteen (18) months, resulting in a combined total of eleven (11) **unplanned extremity modification events** among maintenance staff. Vault-Tec celebrates these participants' contribution to our **Workplace Ergonomics Data Initiative** and reminds all staff: **safety gloves are available via Form MAINT-3B (Supply Requisition — Personal Protective Equipment, Non-Critical)**. Processing time is four to six weeks. Plan ahead — it's the Vault-Tec way!

---

## SECTION 5: ACCESS LOG REQUIREMENTS

Maintaining a **complete and accurate** access log is not merely a suggestion — it is the **lifeblood of our compliance department** and a source of genuine joy for our corporate auditors (when they visit, which they will, absolutely, any day now).

### 5.1 — Mandatory Log Fields

Every door interaction **must** be recorded with the following fields, both in the ZAX Mainframe system and in the physical Overseer ledger:

| Field | Description | Example |
|---|---|---|
| `TIMESTAMP` | Date and time of event | `2076-12-04 / 14:32:00` |
| `INITIATING_OFFICER` | Full name and badge ID | `J. Pemberton / ADM-0042` |
| `CYCLE_TYPE` | `ROUTINE`, `EMERGENCY`, `MANUAL` | `ROUTINE` |
| `AUTHORIZATION_CODE` | Logged system auth code | `ALPHA-VGD-20761204-R` |
| `DOOR_STATUS_FINAL` | `SEALED` or `OPEN` | `SEALED` |
| `POPULATION_COUNT_PRE` | Vault resident headcount before event | `1,000` |
| `POPULATION_COUNT_POST` | Vault resident headcount after event | `1,000` |
| `VARIANCE_JUSTIFICATION` | Required if counts differ | `N/A` |
| `INCIDENT_FORMS_FILED` | List any associated forms submitted | `N/A` |
| `NOTES` | Any additional cheerful observations | — |

> 💡 **PRO TIP:** A variance between `POPULATION_COUNT_PRE` and `POPULATION_COUNT_POST` is a **tremendous data opportunity!** However, it does require submission of the appropriate forms. Please see Section 6.3 for the complete variance documentation checklist. Isn't paperwork *fun?*

---

## SECTION 6: SUPPLEMENTARY PROTOCOLS

### 6.1 — Monthly Passphrase Rotation Schedule

Vocal passphrases are rotated on the **first business day** of each month by the Senior IT Administrator (BETA-1). The new passphrase is delivered in a **sealed envelope** to ALPHA-clearance personnel only. Passphrases must:

- Be between fifteen (15) and thirty (30) words in length
- Contain at least one (1) reference to **Vault-Tec's corporate mission**
- Be **enthusiastically deliverable** under stress conditions
- **Not** contain the words "open," "outside," "surface," "leave," "freedom," or "please"

*Previous passphrases are archived in ZAX Mainframe under directory `/secure/door/passphrases/archive/`. Access requires ALPHA-1 clearance. Yes, they are delightful. No, you may not browse them recreationally.*

---

### 6.2 — Attempted Unauthorized Access: Response Protocol

In the *highly stimulating* event that an unauthorized individual attempts to operate the door, the following response matrix applies:

| Attempt Type | Resident Status | Recommended Response | Forms Required |
|---|---|---|---|
| Terminal tampering | General Employee | Warm verbal correction + HR report | HR-22-C |
| Repeated terminal tampering | General Employee | Enthusiastic escort to Behavioral Sciences | HR-22-C, BSC-7 |
| Physical force on door mechanism | Any non-BETA | Immediate security response + medical standby | SEC-1A, MED-3 |
| Successful unauthorized cycling | Any | **Priority One Incident** — notify Overseer immediately, seal door, file all forms, begin population count, make coffee | SEC-1A, MED-3, HR-22-C, OVR-PRIORITY-1, and Form **74-B (Mass Casualty Disposal Pre-Authorization)** as a *cheerful* precaution |

---

### 6.3 — Population Variance Documentation Checklist

Should a door cycling event result in a change to the resident population count — whether due to **authorized external relocation**, **spontaneous isotope integration events** near the door threshold, or any number of other *swell and scientifically valuable* outcomes — the following forms must be filed in **triplicate** within twenty-four (24) hours:

- [ ] **Form OVR-Delta-1:** Overseer Incident Acknowledgment
- [ ] **Form MED-9B:** Physiological Status Change — Group Filing (for events of 2+ residents)
- [ ] **Form BSC-12:** Behavioral Sciences Data Capture Notification *(the good folks in BSC will want to know!)*
- [ ] **Form 74-B:** Mass Casualty Disposal Pre-Authorization *(precautionary — better to have it and not need it!)*
- [ ] **Form HR-41:** Resident Record Update & Occupancy Reconciliation
- [ ] **Form COMP-2:** Resource Reallocation Notice *(unused ration units must be accounted for — waste not, want not!)*

> All forms are available from the Administrative Supply Closet (Level 2, Room 204). If supply closet is locked, submit **Form ADMIN-REQ-3 (Closet Access Request)**. Processing time is three to five business days. We know, we know — *it's just swell how thorough we are!*

---

## SECTION 7: REVISION HISTORY

| Version | Date | Author | Summary of Changes |
|---|---|---|---|
| 1.0 | March 2072 | B. Howell, Facilities | Initial publication |
| 1.4 | January 2074 | T. Nguyen, Admin IT | Added ZAX integration steps |
| 2.0 | August 2075 | T. Nguyen, Admin IT | Revised passphrase policy after "the incident" |
| 2.3 | February 2076 | P. Carmichael, Admin IT | Updated population count fields following Behavioral Sciences request |
| **3.0** | **December 2076** | **P. Carmichael, Admin IT** | **Major revision: Added manual crank safety notices, Form 74-B to variance checklist, unauthorized access matrix. General tone improvements.** |

---

## DOCUMENT FOOTER

*This document is the exclusive property of **Vault-Tec Corporation**, a wholly-owned subsidiary of optimism. Unauthorized reproduction, distribution, or reading-by-flashlight-under-covers is strictly prohibited and, frankly, a little suspicious. Any questions regarding this document should be directed to the Administrative IT Division, Level 1, Room 108 — right next to the motivational mural of the smiling family that definitely made it to the vault on time.*

*Vault-Tec Corporation: **Because the future belongs to those who prepared — and locked the door behind them!***

---
*End of Document VT-ADMIN-4471-DOOR | Version 3.0 | December 2076*
*Filed in ZAX Mainframe: `/docs/admin/facilities/door_protocol/VT-ADMIN-4471-DOOR_v3.md`*
*Physical copy location: Overseer's Office, Filing Cabinet 2, Drawer C, Folder "DO NOT LOSE THIS"*
