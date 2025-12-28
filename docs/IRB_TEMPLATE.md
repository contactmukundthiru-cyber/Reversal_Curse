# IRB Application Template: Temporal Credit Assignment Study

**Note**: This is a template for IRB/ethics board submission. Fill in the bracketed sections with your specific information.

---

## 1. Study Title

Temporal Credit Assignment Shapes the Direction of Knowledge Retrieval: An Experimental Investigation of Directional Constraints in Human Associative Learning

## 2. Principal Investigator

**Name**: [Your Name]
**Institution**: [Your Institution]
**Department**: [Your Department]
**Email**: [Your Email]
**Phone**: [Your Phone]

## 3. Research Team

| Name | Role | Institution | Training Status |
|------|------|-------------|-----------------|
| [Name] | PI | [Institution] | CITI Certified |
| [Name] | Co-I | [Institution] | CITI Certified |

## 4. Study Purpose and Background

### 4.1 Background

Human associative learning is a fundamental cognitive process, yet the directional properties of learned associations remain poorly understood. Recent work in artificial intelligence (Berglund et al., 2023) has identified a "reversal curse" in large language models where learning "A is B" does not enable retrieval of "B is A." This project investigates whether humans exhibit similar directional constraints in associative learning.

### 4.2 Specific Aims

1. Test whether temporal ordering during learning determines the direction of successful retrieval
2. Determine whether asymmetry direction reverses when presentation order is reversed
3. Compare human learning flexibility to rigid constraints observed in AI systems

### 4.3 Scientific Significance

Understanding directional constraints in human memory has implications for:
- Educational design and knowledge transfer
- Medical communication when guidelines change
- Expert-novice knowledge transmission

## 5. Study Design

### 5.1 Overview

- **Design Type**: 2 × 3 Mixed Factorial Experiment
- **Within-Subjects Factor**: Test Direction (Forward vs. Reverse)
- **Between-Subjects Factor**: Training Condition (A-then-B vs. B-then-A vs. Simultaneous)
- **Randomization**: Participants randomly assigned to conditions; stimulus assignments randomized per participant

### 5.2 Sample Size

- **Target N**: 180 participants (60 per condition)
- **Power Analysis**: With n=60/condition, >95% power to detect medium effect sizes (η²p = 0.06) at α = .05

### 5.3 Study Duration

- **Per Participant**: Approximately 12-14 minutes
- **Total Data Collection**: [Estimated timeline]

## 6. Participant Population

### 6.1 Inclusion Criteria

- Age 18-45 years
- English as native or fluent language
- Normal or corrected-to-normal vision
- Access to computer with internet connection

### 6.2 Exclusion Criteria

- Self-reported learning disabilities affecting memory
- Previous participation in similar symbol-label learning studies
- Technical issues during study completion

### 6.3 Recruitment

- **Platform**: Prolific Academic (online participant pool)
- **Location**: US, UK, Canada, Australia
- **Minimum Requirements**: 95% approval rate, 10+ previous submissions

## 7. Procedures

### 7.1 Consent Process

Participants will be presented with an online consent form that includes:
- Study purpose and procedures
- Duration and compensation
- Risks and benefits
- Confidentiality measures
- Voluntary participation statement
- Contact information for questions

Consent is obtained by checkbox acknowledgment before proceeding.

### 7.2 Study Phases

1. **Consent** (~1 min): Review and acknowledge consent form
2. **Instructions** (~1 min): Condition-specific training instructions
3. **Training** (~6 min): Learn 16 symbol-label pairs with 6 repetitions each
4. **Criterion Check** (~2 min): Verify basic learning before test
5. **Distractor Task** (~1 min): Simple arithmetic to clear working memory
6. **Test Phase** (~3 min): 32 trials testing both directions (4-AFC format)
7. **Survey** (~2 min): Demographics and debriefing questions

### 7.3 Data Collected

- Response accuracy (correct/incorrect)
- Reaction times (milliseconds)
- Confidence ratings (1-5 scale)
- Demographics (age, gender, education, native language)
- Strategy descriptions (free text)

## 8. Risks and Benefits

### 8.1 Potential Risks

**Minimal risk study.** Risks are comparable to everyday life:
- Mild frustration from memory task difficulty
- Minor fatigue from computer-based task (~12 min)
- Potential boredom

### 8.2 Risk Mitigation

- Task is low-stakes (no penalty for errors)
- Duration is short
- Participants can withdraw at any time
- No sensitive questions asked

### 8.3 Potential Benefits

- **Direct Benefits**: None to participants
- **Indirect Benefits**: Contribution to understanding human learning and memory

## 9. Compensation

- **Amount**: $2.50 USD
- **Rate**: ~$12.50/hour
- **Payment Method**: Through Prolific platform
- **Requirement**: Completion of all study phases

## 10. Data Management

### 10.1 Data Collection

- Data stored on secure server with encryption
- Prolific IDs used for payment processing only
- Prolific IDs not linked to research data in publications

### 10.2 Data Storage

- Electronic data stored on [secure server/cloud with encryption]
- Access restricted to research team members
- Data retained for [X years] per institutional policy

### 10.3 Data Sharing

- Anonymized data may be shared in public repositories (OSF) upon publication
- No identifying information in shared datasets
- Compliant with FAIR data principles

## 11. Confidentiality

### 11.1 Identifiers

Only identifier collected is Prolific ID, which:
- Is used solely for compensation processing
- Is not linked to responses in any publication
- Will be removed from dataset after compensation confirmed

### 11.2 Data Security

- All data transmitted via HTTPS
- Server access password-protected
- Database encrypted at rest

## 12. Informed Consent Document

See attached consent form (embedded in study at `experiment/frontend/templates/index.html`).

Key elements included:
- ✓ Study purpose
- ✓ Procedures
- ✓ Duration
- ✓ Compensation
- ✓ Risks and benefits
- ✓ Confidentiality
- ✓ Voluntary participation
- ✓ Right to withdraw
- ✓ Contact information

## 13. Debriefing

Participants receive a brief explanation of the study purpose after completion, including:
- General research question (how temporal order affects memory)
- Appreciation for participation
- Researcher contact for questions

## 14. Supporting Documents

- [ ] Consent form (in study)
- [ ] Study protocol (this document)
- [ ] Pre-registration (docs/pre_registration.md)
- [ ] Data management plan
- [ ] CITI training certificates for all personnel

---

## Appendix A: Consent Form Text

```
INFORMATION SHEET AND CONSENT

Purpose of the Study
This study investigates how people learn and remember associations between symbols and labels. Your participation will help us understand fundamental aspects of human memory.

What You Will Do
- Learn to associate abstract symbols with pronounceable labels
- Complete a brief training phase (~6 minutes)
- Take a short memory test (~3 minutes)
- Answer a few demographic questions

Duration and Compensation
This study takes approximately 12 minutes to complete. You will receive $2.50 for your participation.

Risks and Benefits
There are no known risks beyond those of everyday life. While there are no direct benefits to you, your participation contributes to scientific research on memory and learning.

Confidentiality
Your responses will be kept confidential and stored securely. Your Prolific ID will be used only to process payment and will not be associated with your responses in any publications.

Voluntary Participation
Your participation is voluntary. You may withdraw at any time by closing your browser, though compensation requires completion.

Contact Information
If you have questions about this research, please contact [researcher email].

[ ] I have read and understood the information above. I am at least 18 years old and agree to participate in this study.
```

---

## Appendix B: Pre-Registration Summary

See `docs/pre_registration.md` for full pre-registration document including:

- H1-H4: Primary hypotheses about directional asymmetry
- H5-H6: Mechanistic "killer test" predictions
- H7-H8: Secondary hypotheses (RT, confidence)
- Analysis plan with decision rules
- Exclusion criteria (applied symmetrically)
- Sample size justification

---

**Submission Checklist**:
- [ ] Completed IRB application form
- [ ] This protocol document
- [ ] Consent form
- [ ] Pre-registration document
- [ ] CITI certificates for all personnel
- [ ] Data management plan
- [ ] Letter of support (if required)
