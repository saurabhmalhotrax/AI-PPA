# Duplicate Detector Testing - Findings and Plan

## 1. Current Status (as of last evaluation run)

*   **Evaluation Script:** `scripts/evaluate_duplicates.py` is operational.
*   **Metrics (After Phase 1 Adjustments):**
    *   **F1-Score:** 0.9167 (Meeting target of >= 0.85)
    *   **Precision:** 1.0000 (Excellent, no false positives)
    *   **Recall:** 0.8462
*   **False Negatives:** 6 true duplicate pairs are currently being missed.
*   **Debug Logging:** `src/duplicate_detector.py` includes detailed debug prints.

## 2. Key Findings from False Negative Analysis

The primary factor impacting the remaining missed duplicates is the **quality and accuracy of the input data** extracted by the upstream Vision AI model. The detector often correctly filters pairs based on flawed data provided to it.

**Specific reasons for the 6 remaining missed pairs:**

*   **A. Large Date Difference Exceeds Threshold (4 pairs):**
    *   `rvl_cdip_invoice_125.png` (504 days diff) - *Also noted for poor scan quality.*
    *   `rvl_cdip_invoice_131.png` (91 days diff)
    *   `rvl_cdip_invoice_162.png` (323 days diff)
    *   `rvl_cdip_invoice_191.png` (360 days diff)
    *   *Issue:* Current 7-day threshold is strict. Root cause likely OCR/extraction errors for dates.

*   **B. Large Amount Difference Exceeds Threshold (1 pair):**
    *   `rvl_cdip_invoice_73.png` (Diff: 310.77)
    *   *Issue:* Confirmed Vision AI incorrectly extracted `348.62` (panel number) as `total_amount`.

*   **C. Low Text Similarity Score (1 pair):**
    *   `rvl_cdip_invoice_70.png` (Similarity: 0.6785)
    *   *Issue:* Extracted data fields are significantly different between the original and copy, suggesting major Vision AI error or ground truth issue.

## 3. Comprehensive Plan for Improvement

### Phase 1: Initial Detector Tuning & Diagnostics (Completed)

*   **Actions Taken:**
    *   Implemented robust date parsing (`dateutil`).
    *   Handled `np.nan` float values for dates.
    *   Added detailed debug logging to heuristics.
    *   Implemented handling for `nan` string values in `total_amount` (treating `nan` vs `nan` as a pass).
    *   Increased `DEFAULT_TOP_K_FAISS` from 5 to 20 for better diagnostics.
    *   Lowered `DEFAULT_SIMILARITY_THRESHOLD` from 0.90 to 0.88 to catch near-identical matches with minor extraction differences (e.g., missing `invoice_no`).
*   **Outcome:** Achieved F1 Score of 0.9167, Recall 0.8462, Precision 1.0000. Reduced false negatives significantly.

### Phase 2: Improve Vision AI Extraction Quality (Upstream Focus)

*Goal: Address the root cause of most remaining errors by ensuring cleaner, more accurate data is fed into the detector.*

1.  **Analyze Vision AI Performance:**
    *   **Action:** Review `data/extracted_invoices.csv` for the remaining 6 false negatives (and potentially a wider sample). Pinpoint specific Vision AI extraction errors (wrong field identification, poor OCR on dates/numbers).
2.  **Improve Vision AI Prompting:**
    *   **Action:** Refine prompts sent to the Vision AI. Add specific instructions/examples to disambiguate fields (e.g., `total_amount` vs. other numbers), handle different date formats, and guide behavior for missing fields.
3.  **Investigate Image Pre-processing:**
    *   **Action:** For low-quality scans (e.g., `rvl_cdip_invoice_125.png`), explore applying image pre-processing (denoising, binarization, contrast adjustment, deskewing) *before* sending to Vision AI.
4.  **Consider Vision AI Output Validation/Post-processing:**
    *   **Action:** Implement rules to validate or clean the Vision AI output *after* extraction (e.g., check date plausibility, numeric amounts, potentially flag values like the panel number found in `invoice_no`).

### Phase 3: Implement Data Normalization Layer (Within Detector)

*Goal: Make the detector more robust to minor, semantically irrelevant variations in the extracted data.*

1.  **Define Normalization Rules:**
    *   **Action:** Define specific rules for each key field (`vendor`, `invoice_no`, `date`, `total_amount`).
    *   *Vendor:* Lowercase, strip whitespace. Consider removing common suffixes (Inc, Ltd).
    *   *Invoice No:* Start conservatively (strip whitespace). Evaluate more aggressive options (remove hyphens/slashes, force case) based on analysis of typical formats, ensuring distinct IDs aren't merged.
    *   *Date:* Ensure comparison uses standardized `datetime` objects or `YYYY-MM-DD` format.
    *   *Amount:* Ensure comparison uses cleaned `float` values.
2.  **Refactor `invoice_to_text_representation`:**
    *   **Action:** Modify this function to apply the defined normalization rules to each field *before* concatenating them into the string used for text embedding generation.
3.  **Ensure Heuristics Use Normalized Data:**
    *   **Action:** Verify that the amount and date heuristic checks operate on the consistently normalized/parsed values, not raw strings.
4.  **(Optional) Implement Multi-Scenario Invoice # Comparison:**
    *   **Action:** If simple normalization isn't sufficient, consider modifying the invoice number heuristic check to compare multiple normalized versions (e.g., original vs. no-hyphens version) as discussed.

### Phase 4: Implement Secondary AI Review for Ambiguous Cases

*Goal: Use targeted AI analysis to resolve cases where the initial detection signals are conflicting or borderline.*

1.  **Define Trigger Conditions & Categories:**
    *   **Action:** Establish clear rules for flagging a pair as "Requires Secondary Review" (e.g., high similarity but failed one specific heuristic like date/amount difference; borderline similarity score). Distinguish from "High Confidence Duplicate" and "Likely Not Duplicate".
2.  **Develop Targeted Secondary Prompts:**
    *   **Action:** Craft specific prompts for a secondary AI call (Vision or LLM). Prompts should focus on the specific point of ambiguity (e.g., "Are these the same invoice despite the date difference?", "Verify the total amount on both images.", "Visually compare these two scans for functional identity.").
3.  **Integrate Secondary Review Logic:**
    *   **Action:** Modify the detection workflow. If a pair triggers the secondary review condition, make the targeted API call, parse the response, and use it to make the final duplicate/non-duplicate determination. Account for latency and cost implications.

### Phase 5: Final Evaluation & Threshold Tuning (Iterative)

*Goal: Optimize the complete system after implementing upstream and downstream improvements.*

1.  **Comprehensive Re-evaluation:**
    *   **Action:** After significant changes (esp. Phase 2 & 3), regenerate `data/extracted_invoices.csv` if possible, or use the improved detector on the existing data. Run `scripts/evaluate_duplicates.py` to measure the impact of all improvements.
2.  **Fine-Tune Thresholds:**
    *   **Action:** Based on the new evaluation results (with cleaner data and potentially secondary review), revisit and adjust the heuristic thresholds (`similarity`, `amount`, `date`) and the rules for triggering secondary review for the optimal balance of precision and recall according to business needs and risk appetite.

### Next Steps (Following this Plan):

1.  Begin **Phase 2** by analyzing the Vision AI performance on the remaining false negatives and brainstorming prompt/validation improvements.
2.  Concurrently or subsequently, begin **Phase 3** by implementing basic normalization (whitespace, case) in `invoice_to_text_representation`.
