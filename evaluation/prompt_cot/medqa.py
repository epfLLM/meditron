cot_prompts = [
    {
        "prompt": """Four weeks after starting hydrochlorothiazide, a 49-year-old man with hypertension comes to the physician because of muscle cramps and weakness. His home medications also include amlodipine. His blood pressure today is 176/87 mm Hg. Physical examination shows no abnormalities. The precordial leads of a 12-lead ECG are shown. The addition of which of the following is most likely to have prevented this patient's condition?\n\nOptions:\nA. Torsemide \nB. Nifedipine \nC. Eplerenone \nD. Hydralazine""",
        "gold": "C",
        "steps": [
            "The patient has started hydrochlorothiazide.",
            "He now presents with muscle cramps and weakness and an ECG that supports the diagnosis of hypokalemia.",
            "(A) Torsemide is a loop diuretic and would likely aggravate the hypokalemia.",
            "(B) Nifedipine is a calcium antagonist and would not alleviate the hypocalcemia.",
            "(C) Eplerenone is a potassium-sparing diuretic and would likely decrease the chance of hypokalemia.",
            "(D) Hydralazine is a potent vasodilator and would not decrease the risk of hypokalemia.",
        ]
    },
    {
        "prompt": """A 57-year-old woman comes to the emergency department because of severe pain around her right eye, blurred vision in the same eye, and a headache for the past 4 hours. She is nauseous but has not vomited. She can see colored bright circles when she looks at a light source. She is currently being treated for a urinary tract infection with trimethoprim-sulfamethoxazole. She appears uncomfortable. Vital signs are within normal limits. Examination shows visual acuity of 20/20 in the left eye and counting fingers at 5 feet in the right eye. The right eye shows conjunctival injection and edema of the cornea. The right pupil is dilated and fixed. Intravenous analgesia and antiemetics are administered. Which of the following is the most appropriate next step in management?\n\nOptions:\nA. Perform ultrasound biomicroscopy \nB. Perform gonioscopy \nC. Perform fundoscopy \nD. Administer topical steroids""",
        "gold": "B",
        "steps": [
            "The patient has severe pain, blurred vision, and a headache.",
            "She is also nauseous but has not vomited. She can see bright colored circles when she looks at a light source.",
            "The right eye shows conjunctival injection and edema of the cornea. The right pupil is dilated and fixed.",
            "The tentative diagnosis is acute primary angle-closure glaucoma.",
            "Analgesics have been administered.",
            "The next step is to confirm the diagnosis using the gold standard test.",
            "(A) ultrasound biomicroscopy could be used but are not widely available.",
            "(B) Gonioscopy is the gold-standard test to confirm the diagnosis of acute primary angle-closure glaucoma.",
            "(C) fundoscopy is not as reliable as gonioscopy.",
            "(D) topical steroids is not relevant, the possible medical therapy would be timolol, apraclonidine or pilocarpine.",
            "The most appropriate next step in management is to administer intravenous analgesia and antiemetics."
        ]
    },
    {
        "prompt": """A 55-year-old man with a history of hypertension and benign prostate hyperplasia presents for follow-up 4 days into the treatment of a urinary tract infection with trimethoprim-sulfamethoxazole. His symptoms have resolved, and he reports no problems with urination, with the exception of a weak urine stream and hesitancy, which he has had for the past 2 years. At the time of this visit, the patient is afebrile; the blood pressure is 130/88 mm Hg and the heart rate is 80/min. There is no flank tenderness. A urinalysis reveals no leukocytes and is negative for esterase. The urinalysis reveals 2 red blood cells (RBCs)/ high power field (HPF), and there are no casts on urinary sediment analysis. The physician, however, notices the following abnormality:
        Prior treatment
        BUN 12 mg/dL
        Creatinine 1.2 mg/dL
        Today’s visit
        BUN 13 mg/dL
        Creatinine 2.1 mg/dL\n\nOptions: \nA. Reassure the patient, stop trimethoprim-sulfamethoxazole and repeat the measurement in 1–2 weeks \nB. Schedule an intravenous pyelography for urinary obstruction \nC. Schedule a cystoscopy for urethral obstruction \nD. Admit the patient for further management of acute interstitial nephritis""",
        "gold": "A",
        "steps": [
            "The patient has a history of hypertension and benign prostate hyperplasia, risk factors for urinary tract infections.",
            "He's being treated with trimethoprim-sulfamethoxazole for a urinary tract infection.",
            "The symptoms have resolved, but the patient has a weak urine stream and hesitancy.",
            "The urinalysis reveals 2 red blood cells (RBCs)/ high power field (HPF), and there are no casts on urinary sediment analysis.",
            "BUN and creatinine levels have increased from the prior visit.",
            "This could be due to acute interstitial nephritis, a known but rare side effect of trimethoprim-sulfamethoxazole.",
            "Interstitial nephritis should be suspected if there's an elevated creatinine and a urinalysis that shows white cells, white cell casts, and, in some cases, eosinophiluria.",
            "None of the urinalysis results is present and the patient has no typical symptoms such as fever, rash, suggesting acute interstitial nephritis is an unlikely diagnosis.",
            "The correct answer is (A)."
        ]
    },
    {
        "prompt": """A 32-year-old woman comes to the physician because of weight gain, generalized weakness, and irregular menstrual cycles for the past 16 months. She began having symptoms of insomnia and depression 10 months ago. More recently, she has been having difficulties rising from a chair. She has a 2-year history of hypertension. Current medications include citalopram and hydrochlorothiazide. She is 168 cm (5 ft 6 in) tall and weighs 100 kg (220 lb). BMI is 36 kg/m2. Her pulse is 92/min, respirations are 18/min, and blood pressure is 134/76 mm Hg. She appears tired and has a full, plethoric face and central obesity. Examination of the skin shows violaceous linear striations on her lower abdomen. Two midnight serum cortisol studies show measurements of 288 μg/L and 253 μg/L (N < 90); a 24-hour urinary cortisol measurement was 395 μg (N < 300). Upon follow-up laboratory examination, the patient's serum ACTH levels were also elevated at 136 pg/mL (N = 7–50). Which of the following is the most appropriate next step in evaluation?\n\nOptions: \nA. MRI of the head with contrast \nB. Measure ACTH levels in inferior petrosal sinuses \nC. High-dose dexamethasone suppression test \nD. CT scan of the abdomen with contrast""",
        "gold": "C",
        "steps": [
            "The patient has symptoms suggestive of Cushing's syndrome: weight gain, generalized weakness, irregular menstrual cycles, insomnia, depression, and difficulties rising from a chair.",
            "Physical examination findings, such as a full face, central obesity, and violaceous linear striations on the abdomen, further support the diagnosis.",
            "The two elevated serum cortisol measurements and the elevated 24-hour urinary cortisol confirm the presence of hypercortisolism.",
            "The elevated ACTH levels indicate ACTH-dependent Cushing's syndrome.",
            "The next step is to determine the cause of the ACTH-dependent Cushing's syndrome.",
            "A high-dose dexamethasone suppression test can help differentiate between Cushing's disease (a pituitary cause) and ectopic ACTH syndrome (a non-pituitary cause).",
            "In conclusion, the high-dose dexamethasone suppression test is the most appropriate next step."
        ]
    },
    {
        "prompt": """A 53-year-old man comes to the physician because of a 3-month history of a nonpruritic rash. He has been feeling more tired than usual and occasionally experiences pain in his wrists and ankles. He does not smoke or drink alcohol. His temperature is 37.6°C (99.7°F), pulse is 98/min, respirations are 18/min, and blood pressure is 130/75 mm Hg. Physical examination shows multiple, erythematous, purpuric papules on his trunk and extremities that do not blanch when pressed. The remainder of the examination shows no abnormalities. The patient's hemoglobin is 14 g/dL, leukocyte count is 9,500/mm3, and platelet count is 228,000/mm3. Urinalysis and liver function tests are within normal limits. The test for rheumatoid factor is positive. Serum ANA is negative. Serum complement levels are decreased. Serum protein electrophoresis and immunofixation shows increased gammaglobulins with pronounced polyclonal IgM and IgG bands. Testing for cryoglobulins shows no precipitate after 24 hours. Chest x-ray and ECG show no abnormalities. Which of the following is the most appropriate next step in management?\n\nOptions: \nA. Rapid plasma reagin test \nB. Hepatitis C serology \nC. pANCA assay \nD. Bence Jones protein test""",
        "gold": "B",
        "steps": [
            "The patient presents with a rash, fatigue, joint pain, and has normal blood counts.",
            "His serum complement levels are decreased, suggesting the presence of an immune complex-mediated disease.",
            "Serum protein electrophoresis and immunofixation indicate increased gammaglobulins, specifically polyclonal IgM and IgG bands.",
            "The negative ANA and the absence of cryoglobulins suggest that the condition is not related to Lupus or cryoglobulinemia.",
            "The clinical presentation, combined with the laboratory findings, points toward a possible non-cryoglobulinemic vasculitis.",
            "Hepatitis C is known to be associated with non-cryoglobulinemic vasculitis.",
            "The correct answer is (B) Hepatitis C serology, as testing for the Hepatitis C virus can confirm the diagnosis."
        ]
    }    
]
