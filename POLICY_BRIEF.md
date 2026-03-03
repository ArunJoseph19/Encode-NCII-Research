# Policy Brief: The Enforcement Paradox
## Reframing NCII Regulation for 2026

**Prepared for:** Parliament | Home Office | Encode London  
**Date:** March 2026  
**Classification:** Open

---

## The Core Problem in Plain Language

In 2025, Parliament made it illegal to create a fake naked image of a real person without their consent. This was the right decision. But there is a growing gap between what the law says and what the police can actually do about it.

Anyone can download a free AI model onto a personal laptop, generate Non-Consensual Intimate Imagery (NCII) in minutes, and never leave a trace that any law enforcement agency can detect. The crime is real. The harm is real. The criminal liability is real. The enforcement is not.

**This brief presents technical evidence that reshapes where policy responsibility should lie — and makes concrete recommendations for closing the gap.**

---

## What the Research Shows

We conducted two technical experiments. The findings have direct policy consequences.

### Finding 1: Victims Cannot Protect Themselves

Tools like Glaze and Fawkes were developed to let people "cloak" their photos before posting online, making it harder for AI systems to learn their appearance. These tools generated significant media coverage and are widely recommended as self-protection measures.

Our research demonstrates they do not work against modern AI architectures. An attacker using freely available fine-tuning techniques can bypass these protections entirely, recovering a person's identity from their "protected" images. Recommending these tools to victims provides false reassurance. **They are a policy mirage.**

**Policy consequence:** We cannot place the burden of technical self-defence on potential victims. The protection must be built into the tools, not bolted onto the outputs.

### Finding 2: Developers Can Build Safer Tools — But Often Choose Not To

We implemented a technique called SafeGrad that prevents an AI model from being adapted to generate NCII, even by a technically sophisticated attacker. A model built with this protection resists nudification fine-tuning while retaining full capability for legitimate creative and professional uses.

This means the distinction between a "safe" and an "unsafe" AI model is not a technical accident — it is a design decision. **Releasing an unlocked model is a choice.**

**Policy consequence:** If we can technically prevent these harms at the model level, and developers choose not to, that is a question of accountability, not of technical feasibility.

---

## The Legislative Landscape (2026)

| Instrument | What It Does | Gap |
|------------|-------------|-----|
| **Data Use and Access Act 2025, s.138** | Criminalises creation of NCII | Cannot detect private, local generation |
| **Online Safety Act 2023** | Forces platforms to proactively block NCII | Does not cover locally-run models |
| **Crime and Policing Bill 2026 (pending)** | Criminalises supply of nudification tools | Depends on definition of "supply"; open-source unclear |

The legislative architecture is sound. The enforcement architecture is not.

---

## Three Policy Recommendations

### 1. Repository Accountability

Platforms like HuggingFace and Civitai host thousands of freely downloadable AI model components, including "nudification" add-ons specifically designed to generate sexual content from clothed photos of real people. These platforms currently operate with minimal oversight.

**We recommend:**
- Extend the Crime and Policing Bill 2026 to explicitly impose liability on repository platforms that host nudification components after receiving a qualified notification
- Require automated screening of new model uploads against known harmful-use signatures
- Mandate cross-referencing with the StopNCII.org hash database before allowing NSFW-flagged uploads

### 2. Product Liability for AI Models

The law already requires cars to have seatbelts and pharmaceuticals to undergo safety testing. An AI model capable of generating photorealistic intimate images of real individuals is an inherently high-risk product.

**We recommend:**
- Classify high-resolution human image generation models as high-risk products under reformed product liability law
- Require manufacturers to demonstrate implementation of safety locks (equivalent to the SafeGrad approach our research validated) as a condition of commercial release
- Create a rebuttable presumption of liability where NCII is created using a model released without safety attestation

This does not ban open-source AI development. It creates accountability for commercial releases and gives victims a civil law remedy against negligent developers.

### 3. A UK Sovereign NCII Hash Database

StopNCII.org already operates a hash-matching system that helps prevent NCII from circulating on major platforms. But its coverage is incomplete, participation is voluntary, and it cannot address AI-generated imagery at scale.

**We recommend:**
- Mandate StopNCII participation for all Category 1 and 2A platforms under the Online Safety Act
- Establish a UK Sovereign NCII Hash Database (SNHD), operated by the National Crime Agency, that extends coverage to AI-generated NCII
- Develop technical standards for perceptual hashing of AI-generated faces to enable proactive matching before upload

---

## The Human Rights Dimension

This research connects directly to the Human Rights Act 1998. Article 8 — the right to private life — imposes a positive obligation on the state to protect individuals from serious third-party interference with their private sphere. When the technical landscape makes individual self-protection practically impossible, the state's positive obligation requires structural intervention.

Our finding that victims cannot protect themselves using available tools is not just a technical result. It is evidence that Article 8 rights are being technologically overwhelmed. The response must be proportionate to the scale of the failure: not individual criminal liability for the user in the basement, but accountability further up the supply chain, where the harm was made possible.

---

## Summary of Recommendations

| Recommendation | Legislative Vehicle | Urgency |
|----------------|--------------------|---------| 
| Repository liability for nudification tools | Crime and Policing Bill 2026 | Immediate |
| Product liability classification for high-risk models | Consumer Protection Act reform | 12-18 months |
| Safety Lock Attestation requirement | Secondary legislation / BSI standard | 12 months |
| Mandatory StopNCII participation | Online Safety Act regulatory update | Immediate |
| UK Sovereign NCII Hash Database | Primary legislation | 18-24 months |

---

*Full technical evidence base available in: The Enforcement Paradox: Technical Research Report (2026)*  
*Prepared in support of Encode London's Human Rights Impact Assessment programme*
